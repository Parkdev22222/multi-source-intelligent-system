"""
Feature extraction for the learned instance-pairing head (contribution C2).

The production pipeline decides "is this the same building as last month?" with
a hand-tuned linear blend

    score = 0.7 * clip_cosine + 0.2 * size_similarity + 0.1 * geo_proximity

behind a hard geodesic gate. That blend is brittle: the weights were tuned on a
handful of scenes, the gate discards true matches whenever registration drifts,
and greedy assignment commits to early mistakes.

We keep the same cheap signals but let a small MLP learn how to combine them,
and we add *contextual* features the linear blend cannot express -- mutual-best
flags, candidate ranks and local detection density -- which is where most of
the accuracy comes from.

Everything here is NumPy only; torch is needed for the model, not the features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# feature layout -- keep these names in sync with the paper's feature table
# ---------------------------------------------------------------------------
PAIR_FEATURE_NAMES: Tuple[str, ...] = (
    "clip_sim",           # CLIP cosine between mask-cropped object patches
    "geo_dist_norm",      # geodesic centre distance / match radius
    "geo_iou",            # IoU of the two geo-referenced boxes
    "log_area_ratio",     # log(area_cur / area_past), clipped
    "aspect_diff",        # |log(ar_cur) - log(ar_past)|
    "conf_past",
    "conf_cur",
    "same_class",
    "dx_norm",            # centre offset / sqrt(mean area), signed
    "dy_norm",
    "mutual_best_clip",   # 1 if each is the other's top CLIP candidate
    "mutual_best_geo",
    "rank_for_past",      # normalised rank of this candidate for the past det
    "rank_for_cur",
    "cand_density_past",  # log1p(#candidates) / 3
    "cand_density_cur",
)

UNARY_FEATURE_NAMES: Tuple[str, ...] = (
    "confidence",
    "log_rel_area",       # log(bbox area / image area)
    "aspect",             # log(w/h)
    "pos_x",              # centre position in tile, 0..1
    "pos_y",
    "mask_fill",          # mask area / bbox area -- SAM3 segmentation quality
    "best_iou_other",     # best geo IoU against the other frame
    "best_clip_other",    # best CLIP similarity against the other frame
    "n_overlap_other",    # log1p(#other-frame dets overlapping) / 3
    "border_dist",        # min distance to tile edge, 0..0.5
)

N_PAIR_FEATURES = len(PAIR_FEATURE_NAMES)
N_UNARY_FEATURES = len(UNARY_FEATURE_NAMES)


@dataclass
class Det:
    """Frame-agnostic detection view used by the head.

    Deliberately decoupled from `DetectionResult` / `DetectionRecord` so the
    head can be trained from cached JSONL without importing the pipeline.
    """

    det_id: str
    object_class: str
    class_id: int
    confidence: float
    bbox_px: Tuple[float, float, float, float]     # x1, y1, x2, y2 in tile pixels
    lat: float
    lon: float
    geo_bbox: Tuple[float, float, float, float]    # lat_min, lon_min, lat_max, lon_max
    embedding: Optional[np.ndarray] = None         # L2-normalised CLIP vector
    mask_area: Optional[int] = None

    @property
    def area_px(self) -> float:
        x1, y1, x2, y2 = self.bbox_px
        return max(1e-6, (x2 - x1) * (y2 - y1))

    @property
    def aspect(self) -> float:
        x1, y1, x2, y2 = self.bbox_px
        return max(1e-6, (x2 - x1)) / max(1e-6, (y2 - y1))


# ---------------------------------------------------------------------------
# primitive similarities
# ---------------------------------------------------------------------------
def geo_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two (lat_min, lon_min, lat_max, lon_max) boxes."""
    a_lat1, a_lon1, a_lat2, a_lon2 = a
    b_lat1, b_lon1, b_lat2, b_lon2 = b
    ilat1, ilon1 = max(a_lat1, b_lat1), max(a_lon1, b_lon1)
    ilat2, ilon2 = min(a_lat2, b_lat2), min(a_lon2, b_lon2)
    ih, iw = max(0.0, ilat2 - ilat1), max(0.0, ilon2 - ilon1)
    inter = ih * iw
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a_lat2 - a_lat1) * max(0.0, a_lon2 - a_lon1)
    area_b = max(0.0, b_lat2 - b_lat1) * max(0.0, b_lon2 - b_lon1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def geo_distance_deg(a: Det, b: Det) -> float:
    return math.hypot(a.lat - b.lat, a.lon - b.lon)


# ---------------------------------------------------------------------------
# candidate generation
# ---------------------------------------------------------------------------
def candidate_pairs(
    past: Sequence[Det],
    current: Sequence[Det],
    match_radius_deg: float,
    max_candidates: int = 8,
) -> List[Tuple[int, int]]:
    """(past_idx, cur_idx) candidates within `match_radius_deg`.

    A generous radius is used on purpose: unlike the production hard gate, the
    head is allowed to see borderline candidates and reject them itself. We cap
    the fan-out per detection so the cost stays linear in practice.
    """
    out: List[Tuple[int, int]] = []
    for i, p in enumerate(past):
        scored = []
        for j, c in enumerate(current):
            d = geo_distance_deg(p, c)
            if d <= match_radius_deg:
                scored.append((d, j))
        scored.sort()
        out.extend((i, j) for _, j in scored[:max_candidates])
    return out


# ---------------------------------------------------------------------------
# feature matrices
# ---------------------------------------------------------------------------
def _sim_tables(
    past: Sequence[Det],
    current: Sequence[Det],
    pairs: Sequence[Tuple[int, int]],
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Per-candidate clip / iou lookups plus per-detection bests."""
    clip: Dict[Tuple[int, int], float] = {}
    iou: Dict[Tuple[int, int], float] = {}
    for i, j in pairs:
        clip[(i, j)] = cosine(past[i].embedding, current[j].embedding)
        iou[(i, j)] = geo_iou(past[i].geo_bbox, current[j].geo_bbox)

    best_p: Dict[int, Tuple[float, int]] = {}
    best_c: Dict[int, Tuple[float, int]] = {}
    for (i, j), v in clip.items():
        if v > best_p.get(i, (-2.0, -1))[0]:
            best_p[i] = (v, j)
        if v > best_c.get(j, (-2.0, -1))[0]:
            best_c[j] = (v, i)
    return clip, iou, best_p, best_c


def pair_features(
    past: Sequence[Det],
    current: Sequence[Det],
    pairs: Sequence[Tuple[int, int]],
    match_radius_deg: float,
) -> np.ndarray:
    """(len(pairs), N_PAIR_FEATURES) float32 matrix."""
    if not pairs:
        return np.zeros((0, N_PAIR_FEATURES), dtype=np.float32)

    clip, iou, best_clip_p, best_clip_c = _sim_tables(past, current, pairs)

    # geodesically nearest counterpart per detection, for the mutual-best flags
    near_p: Dict[int, Tuple[float, int]] = {}
    near_c: Dict[int, Tuple[float, int]] = {}
    for i, j in pairs:
        d = geo_distance_deg(past[i], current[j])
        if d < near_p.get(i, (1e9, -1))[0]:
            near_p[i] = (d, j)
        if d < near_c.get(j, (1e9, -1))[0]:
            near_c[j] = (d, i)

    # candidate ranks (by CLIP similarity, descending) and fan-out counts
    by_p: Dict[int, List[int]] = {}
    by_c: Dict[int, List[int]] = {}
    for i, j in pairs:
        by_p.setdefault(i, []).append(j)
        by_c.setdefault(j, []).append(i)
    rank_p = {i: {j: r for r, j in enumerate(sorted(js, key=lambda j: -clip[(i, j)]))}
              for i, js in by_p.items()}
    rank_c = {j: {i: r for r, i in enumerate(sorted(is_, key=lambda i: -clip[(i, j)]))}
              for j, is_ in by_c.items()}

    rows = np.zeros((len(pairs), N_PAIR_FEATURES), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        p, c = past[i], current[j]
        area_p, area_c = p.area_px, c.area_px
        scale = math.sqrt((area_p + area_c) / 2.0)
        n_p, n_c = len(by_p.get(i, ())), len(by_c.get(j, ()))

        rows[k] = (
            clip[(i, j)],
            min(4.0, geo_distance_deg(p, c) / max(1e-9, match_radius_deg)),
            iou[(i, j)],
            float(np.clip(math.log(area_c / area_p), -3.0, 3.0)),
            min(3.0, abs(math.log(c.aspect) - math.log(p.aspect))),
            p.confidence,
            c.confidence,
            float(p.object_class == c.object_class),
            float(np.clip(((c.bbox_px[0] + c.bbox_px[2]) - (p.bbox_px[0] + p.bbox_px[2])) / 2.0 / scale, -5, 5)),
            float(np.clip(((c.bbox_px[1] + c.bbox_px[3]) - (p.bbox_px[1] + p.bbox_px[3])) / 2.0 / scale, -5, 5)),
            float(best_clip_p.get(i, (0, -1))[1] == j and best_clip_c.get(j, (0, -1))[1] == i),
            float(near_p.get(i, (0, -1))[1] == j and near_c.get(j, (0, -1))[1] == i),
            rank_p[i][j] / max(1, n_p - 1) if n_p > 1 else 0.0,
            rank_c[j][i] / max(1, n_c - 1) if n_c > 1 else 0.0,
            math.log1p(n_p) / 3.0,
            math.log1p(n_c) / 3.0,
        )
    return rows


def unary_features(
    dets: Sequence[Det],
    other: Sequence[Det],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """(len(dets), N_UNARY_FEATURES) matrix for the change-instance verifier.

    `other` is the opposite frame: a building that has a near-identical
    counterpart in the other frame did *not* change, and `best_iou_other` /
    `best_clip_other` carry exactly that evidence.
    """
    w, h = image_size
    img_area = float(max(1, w * h))
    rows = np.zeros((len(dets), N_UNARY_FEATURES), dtype=np.float32)

    for k, d in enumerate(dets):
        best_iou = 0.0
        best_clip = 0.0
        n_overlap = 0
        for o in other:
            v = geo_iou(d.geo_bbox, o.geo_bbox)
            if v > best_iou:
                best_iou = v
            if v > 0.1:
                n_overlap += 1
            s = cosine(d.embedding, o.embedding)
            if s > best_clip:
                best_clip = s

        cx = (d.bbox_px[0] + d.bbox_px[2]) / 2.0 / max(1, w)
        cy = (d.bbox_px[1] + d.bbox_px[3]) / 2.0 / max(1, h)
        fill = (d.mask_area / d.area_px) if d.mask_area else 1.0

        rows[k] = (
            d.confidence,
            float(np.clip(math.log(d.area_px / img_area), -12.0, 0.0)),
            float(np.clip(math.log(d.aspect), -3.0, 3.0)),
            cx, cy,
            float(np.clip(fill, 0.0, 1.0)),
            best_iou,
            best_clip,
            math.log1p(n_overlap) / 3.0,
            min(cx, cy, 1.0 - cx, 1.0 - cy),
        )
    return rows

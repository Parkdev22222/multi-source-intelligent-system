"""
Detection cache: run SAM3 + CLIP once, train the head a hundred times.

SAM3 inference over LEVIR-CD dominates the experiment budget, and the head we
are training is 20k parameters. Decoupling them means a full training sweep
(thresholds, ablations, seeds) costs minutes on cached tensors instead of hours
of GPU time -- which is what makes the 12-day schedule feasible.

Layout on disk, one directory per (dataset, split):

    <cache_dir>/
        samples.jsonl        one JSON object per benchmark pair (no embeddings)
        embeddings.npz       {"<pair_id>|past": (N,D), "<pair_id>|cur": (M,D)} fp16

Self-supervised labels are baked in at cache time from the GT change mask:
    coverage(det) = |det_mask AND change_mask| / |det_mask|
A detection with coverage >= `change_coverage_thr` sits inside an annotated
change; one below `same_coverage_thr` is a stable object that must be matched
across the two frames. Everything in between is left unlabelled and excluded
from the loss, so ambiguous boundary detections do not inject noise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from icce.pairing_head.features import Det

logger = logging.getLogger(__name__)

CHANGE_COVERAGE_THR = 0.5
SAME_COVERAGE_THR = 0.25


@dataclass
class CachedDet:
    det_id: str
    object_class: str
    class_id: int
    confidence: float
    bbox_px: List[float]
    lat: float
    lon: float
    geo_bbox: List[float]
    mask_area: Optional[int] = None
    coverage: float = 0.0          # fraction of this detection inside the GT change mask

    def to_det(self, embedding: Optional[np.ndarray] = None) -> Det:
        return Det(
            det_id=self.det_id,
            object_class=self.object_class,
            class_id=self.class_id,
            confidence=self.confidence,
            bbox_px=tuple(self.bbox_px),
            lat=self.lat,
            lon=self.lon,
            geo_bbox=tuple(self.geo_bbox),
            embedding=embedding,
            mask_area=self.mask_area,
        )


@dataclass
class CachedSample:
    pair_id: str
    dataset: str
    split: str
    image_size: Tuple[int, int]
    past: List[CachedDet] = field(default_factory=list)
    current: List[CachedDet] = field(default_factory=list)
    gt_instances: List[List[float]] = field(default_factory=list)   # GT change bboxes (px)
    gt_change_present: Optional[bool] = None
    captions: List[str] = field(default_factory=list)
    parent_scene: str = ""

    def to_json(self) -> Dict:
        d = asdict(self)
        d["image_size"] = list(self.image_size)
        return d

    @classmethod
    def from_json(cls, d: Dict) -> "CachedSample":
        return cls(
            pair_id=d["pair_id"],
            dataset=d["dataset"],
            split=d["split"],
            image_size=tuple(d["image_size"]),
            past=[CachedDet(**x) for x in d.get("past", [])],
            current=[CachedDet(**x) for x in d.get("current", [])],
            gt_instances=[list(b) for b in d.get("gt_instances", [])],
            gt_change_present=d.get("gt_change_present"),
            captions=list(d.get("captions", [])),
            parent_scene=d.get("parent_scene", ""),
        )


class CacheWriter:
    def __init__(self, cache_dir: Path) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh = (self.dir / "samples.jsonl").open("w", encoding="utf-8")
        self._emb: Dict[str, np.ndarray] = {}
        self.n = 0

    def add(
        self,
        sample: CachedSample,
        past_emb: Optional[np.ndarray] = None,
        cur_emb: Optional[np.ndarray] = None,
    ) -> None:
        self._fh.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")
        if past_emb is not None and past_emb.size:
            self._emb[f"{sample.pair_id}|past"] = past_emb.astype(np.float16)
        if cur_emb is not None and cur_emb.size:
            self._emb[f"{sample.pair_id}|cur"] = cur_emb.astype(np.float16)
        self.n += 1

    def close(self) -> Path:
        self._fh.close()
        np.savez_compressed(self.dir / "embeddings.npz", **self._emb)
        logger.info("cached %d samples -> %s", self.n, self.dir)
        return self.dir


def load_cache(cache_dir: Path) -> Tuple[List[CachedSample], Dict[str, np.ndarray]]:
    cache_dir = Path(cache_dir)
    jsonl = cache_dir / "samples.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(
            f"no detection cache at {cache_dir}. "
            f"Run: python -m icce.eval.cache_detections --dataset <name> --split <split>"
        )
    samples = [CachedSample.from_json(json.loads(l)) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]

    emb_path = cache_dir / "embeddings.npz"
    emb: Dict[str, np.ndarray] = {}
    if emb_path.is_file():
        with np.load(emb_path) as z:
            emb = {k: z[k].astype(np.float32) for k in z.files}
    else:
        logger.warning("no embeddings.npz in %s -- CLIP features will be zero", cache_dir)
    return samples, emb


def sample_dets(
    sample: CachedSample,
    emb: Dict[str, np.ndarray],
) -> Tuple[List[Det], List[Det]]:
    """Rehydrate a cached sample into feature-ready `Det` lists."""
    pe = emb.get(f"{sample.pair_id}|past")
    ce = emb.get(f"{sample.pair_id}|cur")
    past = [d.to_det(pe[i] if pe is not None and i < len(pe) else None)
            for i, d in enumerate(sample.past)]
    cur = [d.to_det(ce[i] if ce is not None and i < len(ce) else None)
           for i, d in enumerate(sample.current)]
    return past, cur


# ---------------------------------------------------------------------------
# self-supervised labels
# ---------------------------------------------------------------------------
def px_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + bb - inter) if (aa + bb - inter) > 0 else 0.0


def pair_labels(
    sample: CachedSample,
    candidates: Sequence[Tuple[int, int]],
    iou_match_thr: float = 0.5,
    iou_reject_thr: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels for the candidate pairs.

    Returns (match_label, state_label, valid_mask):
      match_label 1 -> the same stable object seen in both frames
      state_label  0 stationary / 1 moved / 2 modified (only where match == 1)
      valid_mask   0 -> ambiguous, excluded from the loss
    """
    n = len(candidates)
    match = np.zeros(n, dtype=np.float32)
    state = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=np.float32)

    for k, (i, j) in enumerate(candidates):
        p, c = sample.past[i], sample.current[j]
        iou = px_iou(p.bbox_px, c.bbox_px)
        stable = p.coverage < SAME_COVERAGE_THR and c.coverage < SAME_COVERAGE_THR
        changed = p.coverage >= CHANGE_COVERAGE_THR or c.coverage >= CHANGE_COVERAGE_THR

        if iou >= iou_match_thr and stable:
            match[k] = 1.0
            # buildings do not translate; a large centroid shift at high IoU is
            # detector jitter, so only a genuine footprint change is "modified"
            shift = abs((p.bbox_px[0] + p.bbox_px[2]) - (c.bbox_px[0] + c.bbox_px[2])) / 2.0
            scale = max(1.0, (p.bbox_px[2] - p.bbox_px[0]))
            state[k] = 1 if shift > 0.5 * scale else 0
        elif iou >= iou_match_thr and changed:
            match[k] = 1.0
            state[k] = 2                       # overlapping but annotated as changed
        elif iou < iou_reject_thr:
            match[k] = 0.0
        else:
            valid[k] = 0.0                     # 0.3 <= IoU < 0.5: too ambiguous to label

    return match, state, valid


def verify_labels(dets: Sequence[CachedDet]) -> Tuple[np.ndarray, np.ndarray]:
    """(label, valid) for the change-instance verifier."""
    cov = np.array([d.coverage for d in dets], dtype=np.float32)
    label = (cov >= CHANGE_COVERAGE_THR).astype(np.float32)
    valid = ((cov >= CHANGE_COVERAGE_THR) | (cov < SAME_COVERAGE_THR)).astype(np.float32)
    return label, valid

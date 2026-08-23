"""
Cross-frame co-located evidence.

The detect-then-pair design has one dominant error mode: an object the detector
missed in the past frame reappears as a *false new building* in the current
frame. Pairing cannot fix this, because there is nothing to pair with -- the
evidence that the object was already there lives in the past frame's *pixels*,
not in its detections.

This module extracts that evidence directly. For each detection we crop its
footprint from its own frame and the identical footprint from the other frame,
then measure four things:

  xf_clip_sim     CLIP cosine between the two crops. High -> same thing was
                  already there.
  xf_pixel_diff   Mean absolute intensity difference, normalised. Low -> the
                  ground did not change.
  xf_pixel_corr   Normalised cross-correlation, robust to global illumination
                  and seasonal shifts that fool a raw difference.
  xf_edge_delta   Signed change in edge density. A new roof adds strong straight
                  edges to what was bare land; a demolition removes them. This
                  is the single most discriminative cue for building change and
                  it costs one Sobel pass.

All four are cheap and computed during the caching pass, where the images are
already in memory.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

N_FEATURES = 4
_EPS = 1e-8


def _crop(img: np.ndarray, bbox: Sequence[float], pad: float = 0.0) -> Optional[np.ndarray]:
    """Crop `bbox` (x1,y1,x2,y2) from `img`, optionally padded by a fraction."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if pad > 0:
        dw, dh = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1, x2, y2 = x1 - dw, y1 - dh, x2 + dw, y2 + dh
    xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
    xi2, yi2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if xi2 - xi1 < 2 or yi2 - yi1 < 2:
        return None
    return img[yi1:yi2, xi1:xi2]


def _gray(patch: np.ndarray) -> np.ndarray:
    a = patch.astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return a / 255.0


# Gradient magnitude (in 0..1 intensity units) above which a pixel counts as a
# structural edge. Sensor noise at sigma ~ 8/255 produces gradients around
# 0.04, while a roofline against bare ground is a step of 0.2 or more, so this
# threshold separates structure from grain.
_EDGE_THRESHOLD = 0.10


def _edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels sitting on a strong edge.

    Deliberately *not* the mean gradient: mean gradient is dominated by
    high-frequency sensor noise and vegetation texture, which vary between two
    acquisitions of an unchanged scene and would swamp the building signal. A
    thresholded count responds to roof outlines and ignores grain.

    Forward differences rather than `np.gradient`: the central difference the
    latter uses averages over two pixels and so halves the magnitude of a clean
    one-pixel step, which is exactly what a roofline is.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    dx = np.abs(np.diff(gray, axis=1))[:-1, :]
    dy = np.abs(np.diff(gray, axis=0))[:, :-1]
    mag = np.sqrt(dx * dx + dy * dy)
    return float((mag > _EDGE_THRESHOLD).mean())


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation mapped to 0..1."""
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if denom < _EPS:
        return 0.5
    return float(np.clip((a * b).sum() / denom, -1.0, 1.0) * 0.5 + 0.5)


# The footprint boundary carries the strongest evidence -- a roofline against
# bare ground -- but it sits exactly on the bbox edge. Crops are padded so that
# boundary falls inside the compared region rather than being cut away.
_FOOTPRINT_PAD = 0.15


def pixel_features(
    own_img: np.ndarray,
    other_img: np.ndarray,
    bbox: Sequence[float],
) -> Tuple[float, float, float]:
    """(xf_pixel_diff, xf_pixel_corr, xf_edge_delta) for one detection."""
    a = _crop(own_img, bbox, pad=_FOOTPRINT_PAD)
    b = _crop(other_img, bbox, pad=_FOOTPRINT_PAD)
    if a is None or b is None or a.shape[:2] != b.shape[:2]:
        return 0.0, 0.5, 0.0

    ga, gb = _gray(a), _gray(b)
    diff = float(np.abs(ga - gb).mean())
    corr = _ncc(ga, gb)
    # Signed and squashed: positive means the own frame carries more built
    # structure than the other frame -- construction seen from the current
    # frame, demolition seen from the past frame.
    edge_delta = float(np.tanh((_edge_density(ga) - _edge_density(gb)) * 6.0))
    return diff, corr, edge_delta


def compute(
    own_img: np.ndarray,
    other_img: np.ndarray,
    bboxes: Sequence[Sequence[float]],
    own_embeddings: Optional[np.ndarray] = None,
    other_embedder=None,
) -> np.ndarray:
    """(N, 4) cross-frame feature matrix for `bboxes` taken in `own_img`.

    `other_embedder(image, bboxes) -> (N, D)` supplies CLIP vectors for the
    *same footprints* in the other frame. When it is unavailable the CLIP
    column is left at 0 and the three pixel statistics still carry the signal.
    """
    n = len(bboxes)
    out = np.zeros((n, N_FEATURES), dtype=np.float32)
    if n == 0:
        return out

    other_emb = None
    if other_embedder is not None and own_embeddings is not None and len(own_embeddings) == n:
        try:
            other_emb = other_embedder(other_img, bboxes)
        except Exception as exc:
            logger.warning("cross-frame embedding failed (%s); CLIP column left empty", exc)

    for i, bbox in enumerate(bboxes):
        if other_emb is not None and i < len(other_emb):
            a, b = own_embeddings[i], other_emb[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            out[i, 0] = float(np.dot(a, b) / (na * nb)) if na > _EPS and nb > _EPS else 0.0
        out[i, 1], out[i, 2], out[i, 3] = pixel_features(own_img, other_img, bbox)
    return out

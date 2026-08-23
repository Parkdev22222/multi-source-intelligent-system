"""
Instance-level change metrics.

Pixel F1 rewards blob overlap but says nothing about whether the system
identified the right *number of distinct changed objects* — which is exactly
what a written report claims ("3 new buildings appeared"). Contribution C2
(the learned pairing head) operates on instances, so we score instances too.

Protocol (COCO-style, single class):
  * a prediction matches a GT instance when IoU >= `iou_thr` (default 0.5)
  * greedy assignment in descending prediction score, one-to-one
  * dataset-level micro precision / recall / F1
  * additionally, change-type accuracy over matched instances
    (appeared / disappeared / modified)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]   # x1, y1, x2, y2


@dataclass
class ChangeInstance:
    bbox: BBox
    change_type: str = "changed"      # appeared | disappeared | modified | changed
    score: float = 1.0
    object_class: str = "building"
    mask: Optional[np.ndarray] = None  # optional binary mask for mask-IoU


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def _iou(p: ChangeInstance, g: ChangeInstance, use_mask: bool) -> float:
    if use_mask and p.mask is not None and g.mask is not None and p.mask.shape == g.mask.shape:
        return mask_iou(p.mask, g.mask)
    return bbox_iou(p.bbox, g.bbox)


@dataclass
class InstanceScores:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    n_pred: int
    n_gt: int
    type_accuracy: Optional[float] = None
    count_mae: Optional[float] = None       # mean |#pred - #gt| per image
    count_rmse: Optional[float] = None
    per_type: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return asdict(self)

    def as_row(self, name: str) -> str:
        base = (f"{name}  P={self.precision*100:.2f}  R={self.recall*100:.2f}  "
                f"F1={self.f1*100:.2f}")
        if self.type_accuracy is not None:
            base += f"  TypeAcc={self.type_accuracy*100:.2f}"
        if self.count_mae is not None:
            base += f"  CountMAE={self.count_mae:.3f}"
        return base


class InstanceEvaluator:
    """Accumulates instance matches across the test set."""

    def __init__(self, iou_thr: float = 0.5, use_mask: bool = False,
                 score_types: bool = True) -> None:
        self.iou_thr = iou_thr
        self.use_mask = use_mask
        self.score_types = score_types
        self.tp = self.fp = self.fn = 0
        self.n_pred = self.n_gt = 0
        self.type_hits = 0
        self.type_total = 0
        self._count_errs: List[int] = []
        self._type_stats: Dict[str, List[int]] = {}

    def update(self, preds: Sequence[ChangeInstance], gts: Sequence[ChangeInstance]) -> None:
        preds = sorted(preds, key=lambda d: -d.score)
        used = [False] * len(gts)

        img_tp = 0
        for p in preds:
            best_j, best_iou = -1, self.iou_thr
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                v = _iou(p, g, self.use_mask)
                if v >= best_iou:
                    best_j, best_iou = j, v
            if best_j >= 0:
                used[best_j] = True
                img_tp += 1
                if self.score_types:
                    self.type_total += 1
                    hit = int(p.change_type == gts[best_j].change_type)
                    self.type_hits += hit
                    st = self._type_stats.setdefault(gts[best_j].change_type, [0, 0])
                    st[0] += hit
                    st[1] += 1
            else:
                self.fp += 1

        self.tp += img_tp
        self.fn += len(gts) - img_tp
        self.n_pred += len(preds)
        self.n_gt += len(gts)
        self._count_errs.append(len(preds) - len(gts))

    def compute(self) -> InstanceScores:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        type_acc = (self.type_hits / self.type_total) if self.type_total else None
        errs = np.asarray(self._count_errs, dtype=float) if self._count_errs else None
        mae = float(np.abs(errs).mean()) if errs is not None and errs.size else None
        rmse = float(np.sqrt((errs ** 2).mean())) if errs is not None and errs.size else None

        per_type = {
            k: {"accuracy": (v[0] / v[1]) if v[1] else 0.0, "support": v[1]}
            for k, v in sorted(self._type_stats.items())
        }

        return InstanceScores(
            precision=precision, recall=recall, f1=f1,
            tp=self.tp, fp=self.fp, fn=self.fn,
            n_pred=self.n_pred, n_gt=self.n_gt,
            type_accuracy=type_acc, count_mae=mae, count_rmse=rmse,
            per_type=per_type,
        )

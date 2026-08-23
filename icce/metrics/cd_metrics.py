"""
Pixel-level change-detection metrics (LEVIR-CD / WHU-CD / S2Looking protocol).

The literature reports precision, recall, F1 and IoU **of the change class**
plus overall accuracy and Cohen's kappa, all accumulated over the whole test
set (dataset-level confusion matrix), not averaged per image. `ConfusionMatrix`
below follows that convention so our numbers are directly comparable with the
published baselines we cite.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np


@dataclass
class CDScores:
    precision: float
    recall: float
    f1: float
    iou: float
    oa: float
    kappa: float
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def as_row(self, name: str) -> str:
        return (
            f"{name}  P={self.precision*100:.2f}  R={self.recall*100:.2f}  "
            f"F1={self.f1*100:.2f}  IoU={self.iou*100:.2f}  "
            f"OA={self.oa*100:.2f}  K={self.kappa*100:.2f}"
        )


class ConfusionMatrix:
    """Streaming binary confusion matrix over an arbitrary number of images."""

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """`pred` / `gt` are 2-D arrays; anything non-zero counts as 'changed'."""
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch: pred{pred.shape} vs gt{gt.shape}")
        p = pred.astype(bool).ravel()
        g = gt.astype(bool).ravel()
        self.tp += int(np.count_nonzero(p & g))
        self.fp += int(np.count_nonzero(p & ~g))
        self.fn += int(np.count_nonzero(~p & g))
        self.tn += int(np.count_nonzero(~p & ~g))

    def merge(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn
        return self

    def compute(self) -> CDScores:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = tp + fp + fn + tn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        oa = (tp + tn) / total if total else 0.0

        # Cohen's kappa
        if total:
            pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / float(total * total)
            kappa = (oa - pe) / (1 - pe) if (1 - pe) > 1e-12 else 0.0
        else:
            kappa = 0.0

        return CDScores(precision, recall, f1, iou, oa, kappa, tp, fp, fn, tn)


def binarize(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """GT masks ship as 0/255 uint8 (sometimes 0/1). Normalise both."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == bool:
        return arr
    if arr.max() <= 1:
        return arr.astype(bool)
    return arr > threshold


def score_pair(pred: np.ndarray, gt: np.ndarray) -> CDScores:
    """Convenience single-image score (used for per-image error analysis)."""
    cm = ConfusionMatrix()
    cm.update(binarize(pred), binarize(gt))
    return cm.compute()

"""Does cross-frame co-located evidence actually raise accuracy?

Trains the identical head twice on identical scenes -- once with the
cross-frame block zeroed out, once with the real extractor output -- and
compares instance F1. If zeroing the block costs nothing, the feature is dead
weight and should not be in the paper.
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from icce.metrics.instance_metrics import ChangeInstance, InstanceEvaluator
from icce.pairing_head.cache import CacheWriter, load_cache, sample_dets
from icce.pairing_head.infer import LearnedPairer
from icce.pairing_head.train import build_tensors, select_thresholds, train
from test_pairing_head import make_sample


def build(root: Path, split: str, n: int) -> Path:
    w = CacheWriter(root / split)
    for i in range(n):
        s, pe, ce = make_sample(f"{split}_{i//4}_{i%4}_0", split)
        w.add(s, pe, ce)
    w.close()
    return root / split


def strip_cross_frame(samples):
    """Copy of the samples with the cross-frame block removed."""
    out = copy.deepcopy(samples)
    for s in out:
        for d in list(s.past) + list(s.current):
            d.xf = None
    return out


def instance_f1(pairer, samples, emb) -> float:
    ev = InstanceEvaluator(iou_thr=0.5, score_types=False)
    for s in samples:
        past, cur = sample_dets(s, emb)
        res = pairer.pair(past, cur, image_size=s.image_size)
        ev.update(res.change_instances(),
                  [ChangeInstance(bbox=tuple(b)) for b in s.gt_instances])
    return ev.compute().f1


def run(tr_s, tr_e, va_s, va_e, te_s, te_e, label: str) -> float:
    model, _ = train(build_tensors(tr_s, tr_e, 0.001),
                     build_tensors(va_s, va_e, 0.001),
                     epochs=40, device="cpu", seed=0)
    thr = select_thresholds(model, va_s, va_e, 0.001, "cpu")
    f1 = instance_f1(
        LearnedPairer(head=model, match_radius_deg=0.001,
                      match_threshold=thr["match_threshold"],
                      verify_threshold=thr["verify_threshold"]),
        te_s, te_e)
    print(f"  {label:<34} instance F1 = {f1:.4f}")
    return f1


def test_cross_frame_helps():
    root = Path(tempfile.mkdtemp())
    tr_s, tr_e = load_cache(build(root, "train", 80))
    va_s, va_e = load_cache(build(root, "val", 24))
    te_s, te_e = load_cache(build(root, "test", 32))

    heur = instance_f1(LearnedPairer.from_checkpoint(None, match_radius_deg=0.001),
                       te_s, te_e)
    print(f"\n  {'heuristic (production)':<34} instance F1 = {heur:.4f}")

    without = run(strip_cross_frame(tr_s), tr_e, strip_cross_frame(va_s), va_e,
                  strip_cross_frame(te_s), te_e, "learned head, no cross-frame")
    with_xf = run(tr_s, tr_e, va_s, va_e, te_s, te_e, "learned head + cross-frame")

    print(f"\n  cross-frame delta: {(with_xf - without) * 100:+.2f} F1 points")
    assert with_xf > heur, "the head must beat the production heuristic"
    assert with_xf >= without, "cross-frame evidence must not hurt"
    return heur, without, with_xf


if __name__ == "__main__":
    test_cross_frame_helps()
    print("\nOK")

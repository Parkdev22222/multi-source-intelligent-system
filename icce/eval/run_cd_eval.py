"""
Experiment E1/E2: change detection on a public benchmark.

    python -m icce.eval.run_cd_eval \
        --cache data/cache/levir_cd_test \
        --checkpoint data/checkpoints/pairing_head.pt \
        --dataset levir_cd --out results/levir_cd_test

Runs every ablation variant over the *same* cached detections, so the rows
differ only in how detections are paired and verified -- not in what SAM3 saw.
Emits pixel-level metrics (comparable with the published CD literature),
instance-level metrics (the natural unit for a report that counts objects),
and a LaTeX table.

Variants
  geo-only          geodesic proximity alone, no appearance
  heuristic         the production CLIP+geo greedy matcher
  learned-greedy    learned head, greedy assignment (isolates the assignment)
  learned-noverify  learned head + Hungarian, verifier off (isolates the verifier)
  learned (ours)    learned head + Hungarian + verifier
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from icce.convert.mask_to_instances import instances_to_mask
from icce.metrics.cd_metrics import ConfusionMatrix, binarize
from icce.metrics.instance_metrics import ChangeInstance, InstanceEvaluator
from icce.pairing_head.cache import load_cache, sample_dets
from icce.pairing_head.infer import LearnedPairer
from icce.pairing_head.model import HeuristicHead

logger = logging.getLogger(__name__)


def build_variants(checkpoint: Optional[Path], radius: float, device: str,
                   thresholds: Dict[str, float]) -> List[Dict]:
    mt = thresholds.get("match_threshold", 0.5)
    vt = thresholds.get("verify_threshold", 0.5)

    variants = [
        {"name": "geo-only",
         "pairer": LearnedPairer(head=HeuristicHead(w_clip=0.0, w_size=0.0),
                                 match_radius_deg=radius, match_threshold=0.5,
                                 assignment="greedy")},
        {"name": "heuristic (production)",
         "pairer": LearnedPairer(head=HeuristicHead(w_clip=0.7, w_size=0.2),
                                 match_radius_deg=radius, match_threshold=0.5,
                                 assignment="greedy")},
    ]
    if checkpoint is not None:
        from icce.pairing_head.model import PairingHead
        model, _ = PairingHead.load(checkpoint, map_location=device)
        model.to(device)
        variants += [
            {"name": "learned head + greedy",
             "pairer": LearnedPairer(head=model, match_radius_deg=radius,
                                     match_threshold=mt, verify_threshold=vt,
                                     assignment="greedy", device=device)},
            {"name": "learned head, no verifier",
             "pairer": LearnedPairer(head=model, match_radius_deg=radius,
                                     match_threshold=mt, verify_threshold=0.0,
                                     assignment="hungarian", device=device)},
            {"name": "learned head (ours)",
             "pairer": LearnedPairer(head=model, match_radius_deg=radius,
                                     match_threshold=mt, verify_threshold=vt,
                                     assignment="hungarian", device=device)},
        ]
    return variants


def _pred_mask(result, sample, use_masks: bool) -> np.ndarray:
    """Rasterise predicted change instances, preferring real SAM3 masks."""
    w, h = sample.image_size
    if not use_masks:
        return instances_to_mask(result.change_instances(), (h, w))

    from icce.eval.cache_detections import decode_mask

    out = np.zeros((h, w), dtype=bool)
    for o in result.outcomes:
        if o.status in ("matched", "moved"):
            continue
        src = sample.current if o.current_idx is not None else sample.past
        idx = o.current_idx if o.current_idx is not None else o.past_idx
        m = decode_mask(src[idx].mask_rle, (h, w)) if idx is not None else None
        if m is not None:
            out |= m
        else:
            x1, y1, x2, y2 = [int(round(v)) for v in o.bbox_px]
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                out[y1:y2, x1:x2] = True
    return out


def evaluate_variant(variant: Dict, samples, emb, gt_masks: Dict[str, np.ndarray],
                     use_masks: bool) -> Dict:
    pairer = variant["pairer"]
    cm = ConfusionMatrix()
    ev = InstanceEvaluator(iou_thr=0.5, score_types=False)
    n_suppressed = 0
    t0 = time.time()

    for s in samples:
        past, cur = sample_dets(s, emb)
        res = pairer.pair(past, cur, image_size=s.image_size)
        n_suppressed += res.n_suppressed

        ev.update(res.change_instances(), [ChangeInstance(bbox=tuple(b)) for b in s.gt_instances])

        gt = gt_masks.get(s.pair_id)
        if gt is not None:
            cm.update(_pred_mask(res, s, use_masks), binarize(gt))

    pixel = cm.compute()
    inst = ev.compute()
    return {
        "name": variant["name"],
        "pixel": pixel.as_dict(),
        "instance": inst.as_dict(),
        "n_suppressed": n_suppressed,
        "seconds": time.time() - t0,
        # flattened for the table writers
        "precision": pixel.precision, "recall": pixel.recall,
        "f1": pixel.f1, "iou": pixel.iou,
        "inst_precision": inst.precision, "inst_recall": inst.recall, "inst_f1": inst.f1,
    }


def load_gt_masks(dataset: str, split: str, samples) -> Dict[str, np.ndarray]:
    """Reload GT change masks by pair id (masks are not kept in the cache)."""
    from icce.datasets.registry import load as load_dataset
    from icce.eval.cache_detections import load_mask

    wanted = {s.pair_id: s.image_size for s in samples}
    out: Dict[str, np.ndarray] = {}
    try:
        pairs = load_dataset(dataset, split=split)
    except Exception as exc:
        logger.warning("GT masks unavailable (%s); pixel metrics will be skipped", exc)
        return out

    for p in pairs:
        if p.pair_id in wanted and p.mask is not None:
            w, h = wanted[p.pair_id]
            m = load_mask(p.mask, (h, w))
            if m is not None:
                out[p.pair_id] = m
    logger.info("loaded %d/%d GT masks", len(out), len(wanted))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Change-detection evaluation")
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--match-radius", type=float, default=0.001)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bbox-pixels", action="store_true",
                    help="rasterise bounding boxes instead of SAM3 masks")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    samples, emb = load_cache(args.cache)
    logger.info("%d cached pairs from %s", len(samples), args.cache)

    thresholds: Dict[str, float] = {}
    if args.checkpoint and Path(args.checkpoint).is_file():
        from icce.pairing_head.model import PairingHead
        _, extra = PairingHead.load(args.checkpoint)
        thresholds = {k: extra[k] for k in ("match_threshold", "verify_threshold") if k in extra}
        logger.info("thresholds from checkpoint: %s", thresholds)

    gt_masks = load_gt_masks(args.dataset, args.split, samples)
    use_masks = not args.bbox_pixels and any(d.mask_rle for s in samples for d in s.current)
    if not use_masks:
        logger.warning("no cached instance masks -- pixel metrics use bbox rasterisation "
                       "and are a lower bound")

    rows = []
    for v in build_variants(args.checkpoint, args.match_radius, args.device, thresholds):
        r = evaluate_variant(v, samples, emb, gt_masks, use_masks)
        logger.info("%-28s pixelF1=%.4f IoU=%.4f instF1=%.4f (%.1fs)",
                    r["name"], r["f1"], r["iou"], r["inst_f1"], r["seconds"])
        rows.append(r)

    from icce.eval.tables import latex_table, print_console_table, save_json, save_latex

    out = Path(args.out)
    save_json({
        "dataset": args.dataset, "split": args.split, "n_pairs": len(samples),
        "pixel_scoring": "sam3_masks" if use_masks else "bbox_rasterisation",
        "match_radius_deg": args.match_radius, "thresholds": thresholds,
        "results": rows,
    }, out / "cd_results.json")

    print_console_table(f"{args.dataset}/{args.split} -- pixel level",
                        ["precision", "recall", "f1", "iou"], rows)
    print_console_table(f"{args.dataset}/{args.split} -- instance level (IoU>=0.5)",
                        ["inst_precision", "inst_recall", "inst_f1"], rows)

    save_latex(latex_table(
        f"Pixel-level change detection on {args.dataset.upper().replace('_','-')}. "
        "Baselines are fully supervised; ours is open-vocabulary with a "
        "20k-parameter pairing head.",
        f"tab:{args.dataset}_pixel", ["P", "R", "F1", "IoU"], rows,
        baseline_group=args.dataset,
        metric_map={"P": "precision", "R": "recall", "F1": "f1", "IoU": "iou"},
    ), out / f"table_{args.dataset}_pixel.tex")

    save_latex(latex_table(
        f"Instance-level change detection on {args.dataset.upper().replace('_','-')} "
        f"(IoU $\\geq$ 0.5).",
        f"tab:{args.dataset}_instance", ["P", "R", "F1"], rows,
        metric_map={"P": "inst_precision", "R": "inst_recall", "F1": "inst_f1"},
    ), out / f"table_{args.dataset}_instance.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())

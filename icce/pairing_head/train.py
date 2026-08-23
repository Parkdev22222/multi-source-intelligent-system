"""
Train the pairing head on cached detections.

    python -m icce.pairing_head.train \
        --train-cache data/cache/levir_cd_train \
        --val-cache   data/cache/levir_cd_val \
        --out         data/checkpoints/pairing_head.pt

Three losses are optimised jointly:
  * BCE on match, with `pos_weight` compensating the candidate imbalance
  * cross-entropy on state, masked to positive pairs
  * BCE on the change verifier

After training, the match and verify thresholds are chosen on the *validation*
split by maximising instance-level F1 -- never on test. Those thresholds travel
inside the checkpoint so evaluation cannot accidentally retune them.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from icce.pairing_head.cache import (
    CachedSample,
    load_cache,
    pair_labels,
    sample_dets,
    verify_labels,
)
from icce.pairing_head.features import candidate_pairs, pair_features, unary_features
from icce.pairing_head.model import HeadConfig, PairingHead

logger = logging.getLogger(__name__)


@dataclass
class Tensors:
    pair_x: np.ndarray
    pair_cls_past: np.ndarray
    pair_cls_cur: np.ndarray
    match_y: np.ndarray
    state_y: np.ndarray
    pair_valid: np.ndarray
    unary_x: np.ndarray
    unary_cls: np.ndarray
    verify_y: np.ndarray
    verify_valid: np.ndarray
    n_samples: int

    def summary(self) -> str:
        pos = float(self.match_y[self.pair_valid > 0].mean()) if self.pair_valid.any() else 0.0
        chg = float(self.verify_y[self.verify_valid > 0].mean()) if self.verify_valid.any() else 0.0
        return (f"{self.n_samples} pairs | {len(self.pair_x)} candidates "
                f"({pos*100:.1f}% positive) | {len(self.unary_x)} detections "
                f"({chg*100:.1f}% change)")


def build_tensors(
    samples: Sequence[CachedSample],
    emb: Dict[str, np.ndarray],
    match_radius_deg: float,
    max_candidates: int = 8,
) -> Tensors:
    P, CP, CC, MY, SY, PV = [], [], [], [], [], []
    UX, UC, VY, VV = [], [], [], []

    for s in samples:
        past, cur = sample_dets(s, emb)
        cands = candidate_pairs(past, cur, match_radius_deg, max_candidates)
        if cands:
            P.append(pair_features(past, cur, cands, match_radius_deg))
            CP.append(np.array([past[i].class_id for i, _ in cands], np.int64))
            CC.append(np.array([cur[j].class_id for _, j in cands], np.int64))
            m, st, v = pair_labels(s, cands)
            MY.append(m); SY.append(st); PV.append(v)

        for dets, cached, other in ((past, s.past, cur), (cur, s.current, past)):
            if not dets:
                continue
            UX.append(unary_features(dets, other, s.image_size))
            UC.append(np.array([d.class_id for d in dets], np.int64))
            y, v = verify_labels(cached)
            VY.append(y); VV.append(v)

    def cat(xs, shape_if_empty):
        return np.concatenate(xs, axis=0) if xs else np.zeros(shape_if_empty, np.float32)

    return Tensors(
        pair_x=cat(P, (0, 16)),
        pair_cls_past=np.concatenate(CP) if CP else np.zeros(0, np.int64),
        pair_cls_cur=np.concatenate(CC) if CC else np.zeros(0, np.int64),
        match_y=cat(MY, (0,)),
        state_y=np.concatenate(SY) if SY else np.zeros(0, np.int64),
        pair_valid=cat(PV, (0,)),
        unary_x=cat(UX, (0, 10)),
        unary_cls=np.concatenate(UC) if UC else np.zeros(0, np.int64),
        verify_y=cat(VY, (0,)),
        verify_valid=cat(VV, (0,)),
        n_samples=len(samples),
    )


def _pos_weight(y: np.ndarray, valid: np.ndarray) -> float:
    m = valid > 0
    if not m.any():
        return 1.0
    pos = float(y[m].sum())
    neg = float(m.sum() - pos)
    return float(np.clip(neg / max(1.0, pos), 0.2, 20.0))


def train(
    train_t: Tensors,
    val_t: Optional[Tensors],
    epochs: int = 60,
    batch_size: int = 4096,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    state_weight: float = 0.3,
    verify_weight: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[PairingHead, Dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PairingHead(HeadConfig()).to(device)
    model.fit_normalisation(train_t.pair_x, train_t.unary_x)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    bce_match = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(_pos_weight(train_t.match_y, train_t.pair_valid), device=device),
    )
    bce_verify = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(_pos_weight(train_t.verify_y, train_t.verify_valid), device=device),
    )
    ce_state = nn.CrossEntropyLoss(reduction="none")

    def T(a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=device)

    px, pcp, pcc = T(train_t.pair_x), T(train_t.pair_cls_past, torch.long), T(train_t.pair_cls_cur, torch.long)
    my, sy, pv = T(train_t.match_y), T(train_t.state_y, torch.long), T(train_t.pair_valid)
    ux, uc, vy, vv = T(train_t.unary_x), T(train_t.unary_cls, torch.long), T(train_t.verify_y), T(train_t.verify_valid)

    n_pair, n_unary = len(px), len(ux)
    history: List[Dict] = []
    best = {"val_loss": float("inf"), "epoch": -1, "state": None}

    for ep in range(epochs):
        model.train()
        perm_p = torch.randperm(n_pair, device=device) if n_pair else torch.zeros(0, dtype=torch.long)
        perm_u = torch.randperm(n_unary, device=device) if n_unary else torch.zeros(0, dtype=torch.long)
        n_steps = max(1, max(n_pair, n_unary) // batch_size + 1)
        tot = 0.0

        for step in range(n_steps):
            opt.zero_grad()
            loss = torch.zeros((), device=device)

            if n_pair:
                idx = perm_p[step * batch_size:(step + 1) * batch_size]
                if len(idx):
                    logit, state = model.forward_pair(px[idx], pcp[idx], pcc[idx])
                    w = pv[idx]
                    denom = w.sum().clamp_min(1.0)
                    loss = loss + (bce_match(logit, my[idx]) * w).sum() / denom
                    sw = w * my[idx]
                    if sw.sum() > 0:
                        loss = loss + state_weight * (ce_state(state, sy[idx]) * sw).sum() / sw.sum().clamp_min(1.0)

            if n_unary:
                idx = perm_u[step * batch_size:(step + 1) * batch_size]
                if len(idx):
                    vlogit = model.forward_unary(ux[idx], uc[idx])
                    w = vv[idx]
                    loss = loss + verify_weight * (bce_verify(vlogit, vy[idx]) * w).sum() / w.sum().clamp_min(1.0)

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += float(loss.item())

        sched.step()
        rec = {"epoch": ep, "train_loss": tot / n_steps}
        if val_t is not None:
            rec.update(evaluate(model, val_t, device))
            if rec["val_loss"] < best["val_loss"]:
                best = {"val_loss": rec["val_loss"], "epoch": ep,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        history.append(rec)
        if ep % 10 == 0 or ep == epochs - 1:
            logger.info("epoch %3d  %s", ep, {k: round(v, 4) for k, v in rec.items() if isinstance(v, float)})

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        logger.info("restored best epoch %d (val_loss %.4f)", best["epoch"], best["val_loss"])

    model.eval()
    return model, {"history": history, "best_epoch": best["epoch"]}


@torch.no_grad()
def evaluate(model: PairingHead, t: Tensors, device: str = "cpu") -> Dict[str, float]:
    model.eval()

    def T(a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=device)

    out: Dict[str, float] = {}
    losses = []

    if len(t.pair_x):
        logit, state = model.forward_pair(T(t.pair_x), T(t.pair_cls_past, torch.long), T(t.pair_cls_cur, torch.long))
        y, v = T(t.match_y), T(t.pair_valid)
        l = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
        losses.append(float((l * v).sum() / v.sum().clamp_min(1.0)))
        pred = (torch.sigmoid(logit) >= 0.5).float()
        m = v > 0
        out["match_acc"] = float((pred[m] == y[m]).float().mean()) if m.any() else 0.0
        sm = m & (y > 0)
        if sm.any():
            out["state_acc"] = float((state[sm].argmax(-1) == T(t.state_y, torch.long)[sm]).float().mean())

    if len(t.unary_x):
        vlogit = model.forward_unary(T(t.unary_x), T(t.unary_cls, torch.long))
        y, v = T(t.verify_y), T(t.verify_valid)
        l = nn.functional.binary_cross_entropy_with_logits(vlogit, y, reduction="none")
        losses.append(float((l * v).sum() / v.sum().clamp_min(1.0)))
        pred = (torch.sigmoid(vlogit) >= 0.5).float()
        m = v > 0
        out["verify_acc"] = float((pred[m] == y[m]).float().mean()) if m.any() else 0.0

    out["val_loss"] = float(np.mean(losses)) if losses else 0.0
    return out


@torch.no_grad()
def select_thresholds(
    model: PairingHead,
    samples: Sequence[CachedSample],
    emb: Dict[str, np.ndarray],
    match_radius_deg: float,
    device: str = "cpu",
    grid: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
) -> Dict[str, float]:
    """Pick (match_thr, verify_thr) maximising instance F1 on the validation set."""
    from icce.metrics.instance_metrics import ChangeInstance, InstanceEvaluator
    from icce.pairing_head.infer import LearnedPairer

    best = {"match_threshold": 0.5, "verify_threshold": 0.5, "val_instance_f1": -1.0}
    for mt in grid:
        for vt in grid:
            pairer = LearnedPairer(
                head=model, match_radius_deg=match_radius_deg,
                match_threshold=mt, verify_threshold=vt, device=device,
            )
            ev = InstanceEvaluator(iou_thr=0.5, score_types=False)
            for s in samples:
                past, cur = sample_dets(s, emb)
                res = pairer.pair(past, cur, image_size=s.image_size)
                gt = [ChangeInstance(bbox=tuple(b)) for b in s.gt_instances]
                ev.update(res.change_instances(), gt)
            f1 = ev.compute().f1
            if f1 > best["val_instance_f1"]:
                best = {"match_threshold": mt, "verify_threshold": vt, "val_instance_f1": f1}
    logger.info("selected thresholds: %s", best)
    return best


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train the MSIS learned pairing head")
    ap.add_argument("--train-cache", required=True, type=Path)
    ap.add_argument("--val-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/checkpoints/pairing_head.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--match-radius", type=float, default=0.001)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-threshold-search", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t_start = time.time()

    tr_s, tr_e = load_cache(args.train_cache)
    train_t = build_tensors(tr_s, tr_e, args.match_radius, args.max_candidates)
    logger.info("train: %s", train_t.summary())

    val_t = None
    va_s: List[CachedSample] = []
    va_e: Dict[str, np.ndarray] = {}
    if args.val_cache:
        va_s, va_e = load_cache(args.val_cache)
        val_t = build_tensors(va_s, va_e, args.match_radius, args.max_candidates)
        logger.info("val:   %s", val_t.summary())

    model, info = train(
        train_t, val_t, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device, seed=args.seed,
    )

    extra = {
        "match_radius_deg": args.match_radius,
        "max_candidates": args.max_candidates,
        "n_parameters": model.n_parameters(),
        "train_summary": train_t.summary(),
        "best_epoch": info["best_epoch"],
        "seed": args.seed,
        "match_threshold": 0.5,
        "verify_threshold": 0.5,
    }
    if va_s and not args.skip_threshold_search:
        extra.update(select_thresholds(model, va_s, va_e, args.match_radius, args.device))

    model.cpu().save(args.out, extra)
    (args.out.with_suffix(".history.json")).write_text(
        json.dumps({"history": info["history"], "extra": extra}, indent=2), encoding="utf-8"
    )
    logger.info("saved %s (%.1fs, %d params)", args.out, time.time() - t_start, model.n_parameters())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Render one LEVIR-CC crop as a slide/paper figure: before, after, and the change
inventory each pairing method builds from the SAME SAM 3 detections.

    Before (t0) | After (t1) | Hand-tuned rule | Learned head (ours)

Nothing is re-detected: detections, embeddings and cross-frame features come
from the cached pass the paper's numbers were computed on, and both pairers are
built exactly as `icce.eval.run_report_eval` builds them. What the figure shows
is therefore the inventory each E5 row handed to the report LLM.

Usage
-----
    # 1) find good examples (references say "no change" but the rule invents
    #    some; or crops where the head's count is right and the rule's is not)
    python scripts/render_change_example.py --suggest 10

    # 2) render the chosen crops
    python scripts/render_change_example.py --pairs test_002021 test_000004

    # images not where the cache recorded them? point at the LEVIR-CC root
    #   (expects <root>/<split>/A/<id>.png and <root>/<split>/B/<id>.png)
    python scripts/render_change_example.py --pairs test_002021 \
        --image-root data/benchmarks/LEVIR-CC/images

Outputs per crop, in --out (default docs/examples/):
    <id>.png    the 4-panel figure, 300 dpi
    <id>.json   counts per status, references, and every condition's report text
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from icce.pairing_head.cache import load_cache, sample_dets  # noqa: E402
from icce.pairing_head.infer import (  # noqa: E402
    STATUS_CHANGED,
    STATUS_DISAPPEARED,
    STATUS_MATCHED,
    STATUS_MOVED,
    STATUS_NEW,
    LearnedPairer,
)

# Colours chosen to stay legible on aerial imagery and to differ in lightness,
# not hue alone (blue vs. orange), matching the slide palette.
C_APPEARED = "#3D8BFF"
C_DISAPPEARED = "#FF9F2E"
C_MODIFIED = "#C77DFF"
C_MATCHED = "#FFFFFF"
C_SUPPRESSED = "#B8B8B8"
C_GT = "#FFE45C"

CHANGE_STATUSES = (STATUS_NEW, STATUS_DISAPPEARED, STATUS_CHANGED)

REPORT_SOURCES = {
    # label -> generations file (relative to repo root)
    "vlm_direct": "results/levir_cc_caption_vlm/gen_vlm_direct_caption.jsonl",
    "llm_graphrag (rule pairing)":
        "results/levir_cc_caption_heuristic_pairing/gen_llm_graphrag_caption.jsonl",
    "llm_graphrag (learned head)": "results/levir_cc_caption/gen_llm_graphrag_caption.jsonl",
}


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------
def build_pairers(checkpoint: Path, radius: float, device: str):
    rule = LearnedPairer.from_checkpoint(None, match_radius_deg=radius)
    head = LearnedPairer.from_checkpoint(checkpoint, device=device, match_radius_deg=radius)
    return rule, head


def summarize(res, n_past: int, n_cur: int) -> Dict:
    counts = {s: len(res.by_status(s)) for s in
              (STATUS_NEW, STATUS_DISAPPEARED, STATUS_CHANGED, STATUS_MATCHED, STATUS_MOVED)}
    used_p = {o.past_idx for o in res.outcomes if o.past_idx is not None}
    used_c = {o.current_idx for o in res.outcomes if o.current_idx is not None}
    return {
        "appeared": counts[STATUS_NEW],
        "disappeared": counts[STATUS_DISAPPEARED],
        "modified": counts[STATUS_CHANGED],
        "unchanged": counts[STATUS_MATCHED] + counts[STATUS_MOVED],
        "changes": counts[STATUS_NEW] + counts[STATUS_DISAPPEARED] + counts[STATUS_CHANGED],
        "suppressed_past": sorted(set(range(n_past)) - used_p),
        "suppressed_cur": sorted(set(range(n_cur)) - used_c),
    }


def load_reports() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for label, rel in REPORT_SOURCES.items():
        p = ROOT / rel
        if not p.is_file():
            continue
        rows = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["key"]] = r["text"]
        out[label] = rows
    return out


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------
def resolve_image(recorded: Optional[str], image_root: Optional[Path],
                  split: str, sub: str, pair_id: str) -> Path:
    tried = []
    for cand in ([Path(recorded), ROOT / recorded] if recorded else []):
        tried.append(cand)
        if cand.is_file():
            return cand
    if image_root is not None:
        for ext in (".png", ".jpg", ".tif"):
            cand = image_root / split / sub / f"{pair_id}{ext}"
            tried.append(cand)
            if cand.is_file():
                return cand
    raise FileNotFoundError(
        f"{pair_id}: image {sub} not found; tried " + ", ".join(map(str, tried))
        + ". Pass --image-root <LEVIR-CC images dir>.")


def mask_of(det, image_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if not det.mask_rle:
        return None
    try:
        from icce.eval.cache_detections import decode_mask
    except Exception:
        return None
    w, h = image_size
    return decode_mask(det.mask_rle, (h, w))


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def draw_det(ax, det, image_size, color, lw=2.0, ls="-", use_mask=True):
    import matplotlib.patches as mpatches

    m = mask_of(det, image_size) if use_mask else None
    if m is not None and m.any():
        ax.contour(m.astype(float), levels=[0.5], colors=[color],
                   linewidths=lw, linestyles=ls)
        return
    x1, y1, x2, y2 = det.bbox_px
    ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                    edgecolor=color, linewidth=lw, linestyle=ls))


def draw_inventory(ax, res, sample, summary, show_suppressed: bool):
    size = sample.image_size
    for o in res.outcomes:
        if o.status in (STATUS_MATCHED, STATUS_MOVED):
            draw_det(ax, sample.current[o.current_idx], size, C_MATCHED, lw=0.8)
    if show_suppressed:
        for j in summary["suppressed_cur"]:
            draw_det(ax, sample.current[j], size, C_SUPPRESSED, lw=1.2, ls="--")
        for i in summary["suppressed_past"]:
            draw_det(ax, sample.past[i], size, C_SUPPRESSED, lw=1.2, ls="--")
    for o in res.outcomes:
        if o.status == STATUS_NEW:
            draw_det(ax, sample.current[o.current_idx], size, C_APPEARED, lw=2.4)
        elif o.status == STATUS_DISAPPEARED:
            draw_det(ax, sample.past[o.past_idx], size, C_DISAPPEARED, lw=2.4)
        elif o.status == STATUS_CHANGED:
            draw_det(ax, sample.current[o.current_idx], size, C_MODIFIED, lw=2.4)


def caption_line(s: Dict) -> str:
    if s["changes"] == 0:
        return f"no change reported · {s['unchanged']} unchanged"
    parts = []
    if s["appeared"]:
        parts.append(f"+{s['appeared']} appeared")
    if s["disappeared"]:
        parts.append(f"−{s['disappeared']} disappeared")
    if s["modified"]:
        parts.append(f"{s['modified']} modified")
    return " · ".join(parts)


def render(sample, rule_res, head_res, rule_sum, head_sum, img_a, img_b, out_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from PIL import Image

    a = np.asarray(Image.open(img_a).convert("RGB"))
    b = np.asarray(Image.open(img_b).convert("RGB"))
    size = sample.image_size

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.9), dpi=300)
    titles = [
        ("Before (t0)", f"{len(sample.past)} SAM 3 detections"),
        ("After (t1)", f"{len(sample.current)} SAM 3 detections"),
        ("Hand-tuned rule", caption_line(rule_sum)),
        ("Learned head (ours)", caption_line(head_sum)),
    ]
    for ax, img, (t, sub) in zip(axes, [a, b, b, b], titles):
        ax.imshow(img, interpolation="nearest")
        ax.set_xlim(0, size[0])
        ax.set_ylim(size[1], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{t}\n", fontsize=15, fontweight="bold", loc="left", pad=2)
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=11.5,
                color="#4A5568", va="bottom", ha="left")

    for d in sample.past:
        draw_det(axes[0], d, size, C_MATCHED, lw=0.9)
    for d in sample.current:
        draw_det(axes[1], d, size, C_MATCHED, lw=0.9)
    for (x1, y1, x2, y2) in sample.gt_instances:
        axes[1].add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                             edgecolor=C_GT, linewidth=1.6, linestyle=":"))
    draw_inventory(axes[2], rule_res, sample, rule_sum, show_suppressed=False)
    draw_inventory(axes[3], head_res, sample, head_sum, show_suppressed=True)

    handles = [
        mlines.Line2D([], [], color=C_APPEARED, lw=3, label="appeared"),
        mlines.Line2D([], [], color=C_DISAPPEARED, lw=3, label="disappeared (t0 outline)"),
        mlines.Line2D([], [], color=C_MODIFIED, lw=3, label="modified"),
        mlines.Line2D([], [], color="#9AA3B2", lw=1.2, label="detection / unchanged"),
        mlines.Line2D([], [], color=C_SUPPRESSED, lw=1.5, ls="--", label="rejected by verify"),
    ]
    if sample.gt_instances:
        handles.append(mlines.Line2D([], [], color="#C9A800", lw=1.8, ls=":",
                                     label="ground-truth change"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                     frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.01))
    for t in leg.get_texts():
        t.set_color("#14213D")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.10, wspace=0.04)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# suggest
# ---------------------------------------------------------------------------
def suggest(samples, emb, rule, head, reports, n: int) -> None:
    vlm = reports.get("vlm_direct", {})
    nochange, change = [], []
    for s in samples:
        past, cur = sample_dets(s, emb)
        r = summarize(rule.pair(past, cur, image_size=s.image_size), len(past), len(cur))
        h = summarize(head.pair(past, cur, image_size=s.image_size), len(past), len(cur))
        row = (s.pair_id, len(s.gt_instances), r["changes"], h["changes"],
               vlm.get(s.pair_id, "")[:60])
        if s.gt_change_present is False:
            if h["changes"] == 0 and r["changes"] > 0:
                nochange.append((r["changes"], row))
        elif s.gt_instances:
            err_h = abs(h["changes"] - len(s.gt_instances))
            err_r = abs(r["changes"] - len(s.gt_instances))
            if err_h <= 1 and err_r >= 3:
                change.append((err_r - err_h, row))

    def show(title, rows):
        print(f"\n{title}")
        print(f"{'pair':<14}{'GT':>4}{'rule':>6}{'head':>6}  vlm_direct")
        for _, (pid, gt, rc, hc, v) in sorted(rows, key=lambda t: -t[0])[:n]:
            print(f"{pid:<14}{gt:>4}{rc:>6}{hc:>6}  {v}")

    show("No-change crops: rule invents changes, head reports none", nochange)
    show("Change crops: head's count within 1 of GT, rule off by >= 3", change)


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=ROOT / "data/cache/levir_cc_test")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "data/checkpoints/pairing_head.pt")
    ap.add_argument("--match-radius", type=float, default=0.001)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pairs", nargs="*", default=["test_002021"])
    ap.add_argument("--image-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/examples")
    ap.add_argument("--suggest", type=int, default=0,
                    help="print the N strongest candidate crops of each kind and exit")
    args = ap.parse_args(argv)

    samples, emb = load_cache(args.cache)
    by_id = {s.pair_id: s for s in samples}
    rule, head = build_pairers(args.checkpoint, args.match_radius, args.device)
    reports = load_reports()

    if args.suggest:
        suggest(samples, emb, rule, head, reports, args.suggest)
        return 0

    for pid in args.pairs:
        s = by_id.get(pid)
        if s is None:
            print(f"{pid}: not in {args.cache}", file=sys.stderr)
            continue
        past, cur = sample_dets(s, emb)
        rule_res = rule.pair(past, cur, image_size=s.image_size)
        head_res = head.pair(past, cur, image_size=s.image_size)
        rule_sum = summarize(rule_res, len(past), len(cur))
        head_sum = summarize(head_res, len(past), len(cur))
        img_a = resolve_image(s.image_a, args.image_root, s.split, "A", pid)
        img_b = resolve_image(s.image_b, args.image_root, s.split, "B", pid)

        png = args.out / f"{pid}.png"
        render(s, rule_res, head_res, rule_sum, head_sum, img_a, img_b, png)

        info = {
            "pair_id": pid,
            "n_past": len(past), "n_current": len(cur),
            "n_gt_instances": len(s.gt_instances),
            "gt_change_present": s.gt_change_present,
            "references": s.captions,
            "rule": {k: v for k, v in rule_sum.items() if not k.startswith("suppressed")},
            "learned_head": {k: (len(v) if k.startswith("suppressed") else v)
                             for k, v in head_sum.items()},
            "reports": {label: rows.get(pid) for label, rows in reports.items()},
        }
        (args.out / f"{pid}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
        print(f"{pid}: rule {caption_line(rule_sum)} | head {caption_line(head_sum)} -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

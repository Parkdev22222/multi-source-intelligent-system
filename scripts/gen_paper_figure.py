#!/usr/bin/env python3
"""
Build the paper's method figure.

The previous figure (docs/architecture.png) was the deployment diagram of the
original military MSIS system: databases, a Flask API, a Leaflet dashboard, a
doctrine RAG store. It showed none of this paper's contributions, showed the
pairing heuristics the paper *beats* as if they were the method, and carried
military content into a consumer-technology submission. This replaces it.

Design constraints, in order:

  * It has to survive `\\columnwidth` in a two-column IEEE paper, which is about
    3.5 inches. Everything is drawn at that size rather than shrunk into it, so
    the type is set at a size that is actually legible in print.
  * It has to be no taller than the figure it replaces. The draft is exactly
    six pages with no slack, and a first attempt at a tall vertical chain cost
    a seventh. The pipeline tail is therefore a horizontal strip rather than a
    stack, which buys back the height the emphasised head needs.
  * Light background. A near-black figure prints as a solid block.
  * Vector output (PDF) so it stays sharp at any zoom.
  * It shows the mechanism and where the paper's gain comes from, not the
    software architecture.

The report gain and latency callout is read from the result JSON, not typed, so
the figure cannot drift from the tables the way a hand-drawn one would.

    python scripts/gen_paper_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT_PDF = ROOT / "docs" / "method_figure.pdf"
OUT_PNG = ROOT / "docs" / "method_figure.png"

# --- palette: light, print-safe, distinguishable in greyscale -------------
INK = "#1a1d23"
MUTED = "#6b7280"
LINE = "#9aa2af"
FILL_STAGE = "#f4f6f9"
EDGE_STAGE = "#b8c0cc"
FILL_HEAD = "#e8eefb"
EDGE_HEAD = "#3b6cc4"
FILL_EVID = "#fdf1e3"
EDGE_EVID = "#c8802a"
FILL_EVAL = "#ffffff"
EDGE_EVAL = "#7a8492"

FIG_W, FIG_H = 3.5, 2.42


def _load(rel: str) -> dict | None:
    p = ROOT / rel
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def read_callout() -> str:
    """Controlled pairing-to-report gain and incremental pairing latency."""
    learned = _load("results/levir_cc_caption/report_results_caption.json")
    heuristic = _load(
        "results/levir_cc_caption_heuristic_pairing/report_results_caption.json"
    )
    efficiency = _load("results/efficiency/efficiency.json")

    gain = None
    if learned and heuristic:
        def graph_f1(blob):
            rows = {r["name"]: r for r in blob["results"]}
            return rows["llm_graphrag"]["cfs_f1"]
        gain = (graph_f1(learned) - graph_f1(heuristic)) * 100

    latency = None
    if efficiency:
        stages = {r["stage"]: r for r in efficiency["stages"]}
        latency = (stages["pairing: learned head (ours)"]["per_item_ms"]
                   - stages["pairing: heuristic (production)"]["per_item_ms"])

    if gain is None or latency is None:
        print("warning: incomplete E5/E6 results; callout left unquantified")
        return "pairing changes report factuality at negligible cost"
    return f"Pairing swap: +{gain:.2f} CFS-F1  ·  +{latency:.2f} ms/tile"


def box(ax, x, y, w, h, label, sub=None, fill=FILL_STAGE, edge=EDGE_STAGE,
        lw=0.9, fs=7.0, ls="-", badge=None, badge_color=None):
    """A stage box. Sub-label is placed *relative to the box height* so it
    cannot spill past the border into whatever sits underneath."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.018",
                                linewidth=lw, facecolor=fill, edgecolor=edge,
                                linestyle=ls, mutation_aspect=1.0, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.63, label, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=fs - 1.4, color=MUTED, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)
    if badge:
        ax.text(x + w - 0.018, y + h - 0.017, badge, ha="right", va="center",
                fontsize=6.2, color=badge_color or EDGE_HEAD,
                fontweight="bold", zorder=4)


def arrow(ax, x1, y1, x2, y2, color=LINE, lw=0.9, ls="-", zorder=1, shrink=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=lw, color=color,
                                 linestyle=ls, shrinkA=shrink, shrinkB=shrink,
                                 zorder=zorder))


def build() -> None:
    result_callout = read_callout()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    L, R = 0.03, 0.97
    W = R - L
    cx = (L + R) / 2

    # ---- inputs and feature preparation ----------------------------------
    y_in, h_in = 0.910, 0.060
    box(ax, 0.18, y_in, 0.64, h_in, "TWO CO-REGISTERED REVISITS",
        sub="past image $t_0$  ·  current image $t_1$", fs=5.8)

    y_feat, h_feat = 0.770, 0.095
    left_x, left_w = L, 0.545
    right_x, right_w = 0.605, R - 0.605
    box(ax, left_x, y_feat, left_w, h_feat,
        "SAM3 DETECTIONS + PAIR FEATURES",
        sub="past $P$ · current $C$  |  CLIP · geometry · context",
        fill=FILL_STAGE, edge=EDGE_STAGE, fs=5.2)
    box(ax, right_x, y_feat, right_w, h_feat,
        "CO-LOCATED CROSS-FRAME EVIDENCE",
        sub="same footprint in both images",
        fill=FILL_EVID, edge=EDGE_EVID, fs=4.8)
    arrow(ax, 0.41, y_in, 0.30, y_feat + h_feat)
    arrow(ax, 0.59, y_in, 0.79, y_feat + h_feat)

    # ---- the learned head, emphasised ------------------------------------
    hy, hh = 0.305, 0.420
    ax.add_patch(FancyBboxPatch((L, hy), W, hh,
                                boxstyle="round,pad=0.008,rounding_size=0.018",
                                linewidth=1.6, facecolor=FILL_HEAD,
                                edgecolor=EDGE_HEAD, zorder=2))
    ax.text(cx, hy + hh - 0.040, "LEARNED PAIRING HEAD", ha="center",
            va="center", fontsize=7.3, color=EDGE_HEAD, fontweight="bold",
            zorder=3)
    ax.text(cx, hy + hh - 0.078, "19.8k parameters  ·  trained from change masks",
            ha="center", va="center", fontsize=5.8, color=MUTED, zorder=3)

    iw, ih = 0.520, 0.092
    ix = L + 0.045
    r_match = hy + 0.205
    r_verify = hy + 0.065
    box(ax, ix, r_match, iw, ih, "MATCH + AUXILIARY STATE",
        sub="same object? · stationary / moved / modified",
        fill="#ffffff", edge=EDGE_HEAD, fs=5.8, lw=0.8)
    box(ax, ix, r_verify, iw, ih, "VERIFY UNMATCHED INSTANCES",
        sub="real change or detection miss?",
        fill="#ffffff", edge=EDGE_HEAD, fs=5.8, lw=0.8)
    arrow(ax, ix + iw / 2, r_match, ix + iw / 2, r_verify + ih,
          color=LINE, lw=0.9, zorder=5)
    ax.text(ix + iw / 2 + 0.020, (r_match + r_verify + ih) / 2,
            "unmatched only", ha="left", va="center", fontsize=4.6,
            color=MUTED, style="italic", zorder=6)

    # Pair features feed matching; image-derived co-located evidence feeds
    # only the unmatched verifier.
    arrow(ax, left_x + left_w / 2, y_feat, ix + iw / 2, r_match + ih,
          color=LINE, lw=0.9, zorder=5)
    elbow_x = R - 0.045
    target_y = r_verify + ih / 2
    ax.plot([right_x + right_w / 2, elbow_x, elbow_x],
            [y_feat, y_feat - 0.025, target_y], color=EDGE_EVID,
            linewidth=1.0, zorder=4)
    arrow(ax, elbow_x, target_y, ix + iw, target_y,
          color=EDGE_EVID, lw=1.0, zorder=5)

    ax.text(cx + 0.10, hy + 0.019, result_callout,
            ha="center", va="center", fontsize=5.0, color=EDGE_HEAD,
            fontweight="bold", zorder=4,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff",
                      edgecolor=EDGE_HEAD, linewidth=0.7))

    # ---- tail as a strip, not a stack ------------------------------------
    ty, th = 0.070, 0.145
    n = 4
    gap = 0.030
    bw = (W - gap * (n - 1)) / n
    tail = [("PAIRING RESULTS", "matched · moved · modified\nappeared · disappeared",
             FILL_STAGE, EDGE_STAGE, "-"),
            ("GROUNDED\nEVIDENCE", "current inventory\noptional graph context",
             FILL_STAGE, EDGE_STAGE, "-"),
            ("GROUNDED LLM\nREPORT", None, FILL_STAGE, EDGE_STAGE, "-"),
            ("DETERMINISTIC\nEVALUATION", "references + GT masks\nCFS · ChgAcc · CountMAE",
             FILL_EVAL, EDGE_EVAL, (0, (2.5, 2)))]
    for i, (label, sub, fill, edge, ls) in enumerate(tail):
        x = L + i * (bw + gap)
        box(ax, x, ty, bw, th, label, sub=sub, fill=fill, edge=edge,
            fs=4.6, lw=0.9, ls=ls)
        if i:
            arrow(ax, x - gap, ty + th / 2, x, ty + th / 2,
                  ls=(0, (2.5, 2)) if i == n - 1 else "-", zorder=5)

    arrow(ax, L + bw / 2, hy, L + bw / 2, ty + th)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT_PNG, format="png", dpi=400, bbox_inches="tight",
                pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {OUT_PDF.relative_to(ROOT)}")
    print(f"wrote {OUT_PNG.relative_to(ROOT)}  (preview only; the paper uses the PDF)")
    print(f"callout: {result_callout!r}")


if __name__ == "__main__":
    build()
    sys.exit(0)

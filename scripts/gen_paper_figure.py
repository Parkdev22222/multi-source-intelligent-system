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

The two callouts are read from the result JSON, not typed, so the figure cannot
drift from the tables the way a hand-drawn one would.

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


def read_callouts() -> tuple[str, str]:
    """Gain at the pairing head, and spread across grounding conditions."""
    cd = _load("results/levir_cd_test/cd_results.json")
    if cd:
        by = {r["name"]: r for r in cd["results"]}
        gain = (by["learned head (ours)"]["f1"]
                - by["heuristic (production)"]["f1"]) * 100
        head = f"+{gain:.1f} pixel F1"
    else:
        head = "pixel F1"
        print("warning: no LEVIR-CD results; head callout left unquantified")

    rp = _load("results/levir_cc_caption/report_results_caption.json")
    if rp:
        f1 = {r["name"]: r["cfs_f1"] for r in rp["results"]}
        ladder = [f1[k] for k in ("llm_struct", "llm_flat_rag", "llm_graphrag")
                  if k in f1]
        spread = (max(ladder) - min(ladder)) * 100
        ground = f"$<${spread + 0.05:.1f} CFS-F1"
    else:
        ground = "CFS-F1"
        print("warning: no report results; grounding callout left unquantified")
    return head, ground


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
    head_gain, ground_spread = read_callouts()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    L, R = 0.03, 0.97
    W = R - L
    cx = (L + R) / 2

    # ---- detection --------------------------------------------------------
    y_det, h_det = 0.868, 0.098
    box(ax, L, y_det, W, h_det, "SAM3  open-vocabulary detection  ($t_0$, $t_1$)",
        sub="never trained on either benchmark  ·  geo-referenced", fs=7.0)

    ax.text(cx + 0.035, 0.828, "detections  $P$, $C$", ha="left", va="center",
            fontsize=6.0, color=MUTED, style="italic")

    # ---- the learned head, emphasised ------------------------------------
    hy, hh = 0.372, 0.394
    ax.add_patch(FancyBboxPatch((L, hy), W, hh,
                                boxstyle="round,pad=0.008,rounding_size=0.018",
                                linewidth=1.6, facecolor=FILL_HEAD,
                                edgecolor=EDGE_HEAD, zorder=2))
    ax.text(cx, hy + hh - 0.040, "LEARNED PAIRING HEAD", ha="center",
            va="center", fontsize=7.5, color=EDGE_HEAD, fontweight="bold",
            zorder=3)
    ax.text(cx, hy + hh - 0.078, "20k parameters  ·  supervised from CD masks",
            ha="center", va="center", fontsize=6.0, color=MUTED, zorder=3)
    ax.text(R - 0.020, hy + 0.028, head_gain, ha="right", va="center",
            fontsize=6.8, color=EDGE_HEAD, fontweight="bold", zorder=4)

    iw, ih = 0.415, 0.092
    r1 = hy + 0.176
    r2 = hy + 0.068
    box(ax, L + 0.022, r1, iw, ih, "match", sub="one-to-one assignment",
        fill="#ffffff", edge=EDGE_HEAD, fs=6.9, lw=0.8)
    box(ax, R - 0.022 - iw, r1, iw, ih, "state", sub="stationary / moved",
        fill="#ffffff", edge=EDGE_HEAD, fs=6.9, lw=0.8)
    box(ax, L + 0.022, r2, iw, ih, "verify", sub="is it real change?",
        fill="#ffffff", edge=EDGE_HEAD, fs=6.9, lw=0.8)
    box(ax, R - 0.022 - iw, r2, iw, ih, "cross-frame evidence",
        sub="same footprint, both frames", fill=FILL_EVID, edge=EDGE_EVID,
        fs=6.9, lw=0.8)
    arrow(ax, R - 0.024 - iw, r2 + ih / 2, L + 0.024 + iw, r2 + ih / 2,
          color=EDGE_EVID, lw=1.1, zorder=5)

    arrow(ax, cx, y_det, cx, hy + hh)

    # ---- tail as a strip, not a stack ------------------------------------
    ty, th = 0.150, 0.132
    n = 4
    gap = 0.030
    bw = (W - gap * (n - 1)) / n
    tail = [("change\ninstances", FILL_STAGE, EDGE_STAGE, "-"),
            ("knowledge\ngraph", FILL_STAGE, EDGE_STAGE, "-"),
            ("LLM\nreport", FILL_STAGE, EDGE_STAGE, "-"),
            ("Change-\nFact-Score", FILL_EVAL, EDGE_EVAL, (0, (2.5, 2)))]
    for i, (label, fill, edge, ls) in enumerate(tail):
        x = L + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, ty), bw, th,
                                    boxstyle="round,pad=0.008,rounding_size=0.018",
                                    linewidth=0.9, facecolor=fill,
                                    edgecolor=edge, linestyle=ls, zorder=2))
        ax.text(x + bw / 2, ty + th / 2, label, ha="center", va="center",
                fontsize=6.5, color=INK, zorder=3, linespacing=1.35)
        if i:
            arrow(ax, x - gap, ty + th / 2, x, ty + th / 2,
                  ls=(0, (2.5, 2)) if i == n - 1 else "-", zorder=5)

    arrow(ax, cx, hy, cx, ty + th)
    ax.text(R - 0.010, ty - 0.055, ground_spread, ha="right", va="center",
            fontsize=6.5, color=MUTED, fontweight="bold")

    ax.text(L, ty - 0.055, "the accuracy lever is upstream of the LLM",
            ha="left", va="center", fontsize=6.4, color=INK, style="italic")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT_PNG, format="png", dpi=400, bbox_inches="tight",
                pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {OUT_PDF.relative_to(ROOT)}")
    print(f"wrote {OUT_PNG.relative_to(ROOT)}  (preview only; the paper uses the PDF)")
    print(f"callouts: head={head_gain!r}  grounding={ground_spread!r}")


if __name__ == "__main__":
    build()
    sys.exit(0)

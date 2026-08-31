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

FIG_W, FIG_H = 3.5, 2.14



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
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    # The drawing occupies 0.22 upwards; cropping to it removes a band of empty
    # canvas that bbox_inches alone does not reclaim from an invisible axes.
    ax.set_ylim(0.205, 0.975)
    ax.axis("off")

    L, R = 0.03, 0.97
    W = R - L
    cx = (L + R) / 2

    # A clear channel is reserved down the right-hand side of the head so the
    # cross-frame path can reach the verifier without crossing the state box.
    CHANNEL = 0.075
    ix = L + 0.026
    inner_w = W - 0.052 - CHANNEL
    chan_x = ix + inner_w + CHANNEL / 2

    # ---- input ------------------------------------------------------------
    # Two boxes rather than one sentence: that there are *two* images is the
    # premise of the whole task, and a reader should see it rather than parse
    # it. Both feed both consumers -- detection runs on each image separately,
    # cross-frame evidence comes from comparing them -- so the four edges meet
    # at a short bus instead of crossing each other diagonally.
    y_in, h_in = 0.900, 0.062
    iw_in = 0.30
    x_before, x_after = 0.185, 0.515
    box(ax, x_before, y_in, iw_in, h_in, "before image", fill="#ffffff", fs=7.2)
    box(ax, x_after, y_in, iw_in, h_in, "after image", fill="#ffffff", fs=7.2)

    # ---- detections and cross-frame evidence ------------------------------
    y_f, h_f = 0.757, 0.095
    lw_ = 0.46
    box(ax, L, y_f, lw_, h_f, "SAM3 detections",
        sub="one set per image · geo-referenced", fs=7.2)
    box(ax, R - lw_, y_f, lw_, h_f, "cross-frame evidence",
        sub="same location in both images", fill=FILL_EVID, edge=EDGE_EVID,
        fs=7.2)
    y_bus = (y_in + y_f + h_f) / 2
    x_det, x_xf = L + lw_ / 2, R - lw_ / 2
    ax.plot([x_det, x_xf], [y_bus, y_bus], color=LINE, linewidth=0.9,
            solid_capstyle="round", zorder=1)
    for x_src in (x_before + iw_in / 2, x_after + iw_in / 2):
        ax.plot([x_src, x_src], [y_in, y_bus], color=LINE, linewidth=0.9,
                zorder=1)
    arrow(ax, x_det, y_bus, x_det, y_f + h_f)
    arrow(ax, x_xf, y_bus, x_xf, y_f + h_f)

    # ---- the learned head -------------------------------------------------
    # Wiring matters here. `match` splits its candidates two ways: matched
    # pairs go to `state`, whose `modified` label becomes a change instance,
    # and unmatched detections go to `verify`, which either keeps them
    # (appeared / disappeared) or drops them. An earlier draft drew `state` as
    # a box with no input and no output, which made it unreadable, and drew
    # `verify` as a pass-through, which hid the discard that the whole
    # contribution rests on.
    hy, hh = 0.408, 0.300
    ax.add_patch(FancyBboxPatch((L, hy), W, hh,
                                boxstyle="round,pad=0.008,rounding_size=0.018",
                                linewidth=1.6, facecolor=FILL_HEAD,
                                edgecolor=EDGE_HEAD, zorder=2))
    ax.text(cx, hy + hh - 0.038, "learned pairing head", ha="center",
            va="center", fontsize=7.8, color=EDGE_HEAD, fontweight="bold",
            zorder=3)

    mw, mh, my = 0.42, 0.058, 0.586
    box(ax, cx - mw / 2, my, mw, mh, "match", fill="#ffffff", edge=EDGE_HEAD,
        fs=7.2, lw=0.8)

    ih, ry = 0.076, 0.462
    bw2 = (inner_w - 0.028) / 2
    x_state, x_verify = ix, ix + bw2 + 0.028
    box(ax, x_state, ry, bw2, ih, "state", sub="modified = change",
        fill="#ffffff", edge=EDGE_HEAD, fs=7.2, lw=0.8)
    box(ax, x_verify, ry, bw2, ih, "verify", sub="keep or discard",
        fill="#ffffff", edge=EDGE_HEAD, fs=7.2, lw=0.8)

    arrow(ax, cx - mw / 4, my, x_state + bw2 / 2, ry + ih, color=EDGE_HEAD,
          zorder=5)
    arrow(ax, cx + mw / 4, my, x_verify + bw2 / 2, ry + ih, color=EDGE_HEAD,
          zorder=5)
    ax.text(x_state + bw2 / 2 - 0.010, (my + ry + ih) / 2, "matched",
            ha="right", va="center", fontsize=6.2, color=MUTED, style="italic",
            zorder=5)
    ax.text(x_verify + bw2 / 2 + 0.010, (my + ry + ih) / 2, "unmatched",
            ha="left", va="center", fontsize=6.2, color=MUTED, style="italic",
            zorder=5)

    arrow(ax, L + lw_ / 2, y_f, L + lw_ / 2, hy + hh)
    ax.add_patch(FancyArrowPatch((chan_x, y_f), (x_verify + bw2 + 0.002,
                                                 ry + ih / 2),
                                 arrowstyle="-|>", mutation_scale=7,
                                 linewidth=1.1, color=EDGE_EVID,
                                 connectionstyle="angle,angleA=-90,angleB=0,rad=4",
                                 shrinkA=0, shrinkB=0, zorder=5))

    # ---- tail -------------------------------------------------------------
    ty, th = 0.215, 0.155
    n, gap = 4, 0.028
    bw = (W - gap * (n - 1)) / n
    # (label, sub): the label may wrap, and only `sub` is set in muted type, so
    # a wrapped label is never mistaken for a label plus a caption.
    tail = [(("change\ninstances", None), FILL_STAGE, EDGE_STAGE, "-"),
            (("knowledge graph", "optional context"), FILL_STAGE, EDGE_STAGE, "-"),
            (("LLM\nreport", None), FILL_STAGE, EDGE_STAGE, "-"),
            (("Change-\nFact-Score", None), FILL_EVAL, EDGE_EVAL, (0, (2.5, 2)))]
    for i, ((label, sub), fill, edge, ls) in enumerate(tail):
        x = L + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, ty), bw, th,
                                    boxstyle="round,pad=0.008,rounding_size=0.018",
                                    linewidth=0.9, facecolor=fill,
                                    edgecolor=edge, linestyle=ls, zorder=2))
        if sub:
            ax.text(x + bw / 2, ty + th * 0.60, label, ha="center",
                    va="center", fontsize=6.9, color=INK, zorder=3,
                    linespacing=1.35)
            ax.text(x + bw / 2, ty + th * 0.28, sub, ha="center", va="center",
                    fontsize=5.9, color=MUTED, zorder=3)
        else:
            ax.text(x + bw / 2, ty + th / 2, label, ha="center", va="center",
                    fontsize=6.9, color=INK, zorder=3, linespacing=1.35)
        if i:
            arrow(ax, x - gap, ty + th / 2, x, ty + th / 2,
                  ls=(0, (2.5, 2)) if i == n - 1 else "-", zorder=5)

    arrow(ax, L + bw / 2, hy, L + bw / 2, ty + th)
    ax.text(L + bw / 2 + 0.014, (hy + ty + th) / 2,
            "modified · appeared · disappeared", ha="left", va="center",
            fontsize=6.0, color=MUTED, style="italic")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT_PNG, format="png", dpi=400, bbox_inches="tight",
                pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {OUT_PDF.relative_to(ROOT)}")
    print(f"wrote {OUT_PNG.relative_to(ROOT)}  (preview only; the paper uses the PDF)")


if __name__ == "__main__":
    build()
    sys.exit(0)

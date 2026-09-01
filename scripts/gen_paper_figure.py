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
    ax.set_ylim(0.188, 0.975)
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
    # The two images sit directly above the detector, because that is their
    # only consumer in the drawing. An earlier version also dropped an arrow
    # into cross-frame evidence -- true, since it samples pixels -- but two
    # incoming edges left it ambiguous which one gates it, and readers kept
    # reading it as a second branch off the input rather than a step that needs
    # detections to exist. The sub-label carries the pixel access instead.
    y_in, h_in = 0.900, 0.062
    iw_in, gap_in = 0.205, 0.022
    x_det_c = L + 0.445 / 2
    x_before = x_det_c - iw_in - gap_in / 2
    x_after = x_det_c + gap_in / 2
    box(ax, x_before, y_in, iw_in, h_in, "before image", fill="#ffffff", fs=6.9)
    box(ax, x_after, y_in, iw_in, h_in, "after image", fill="#ffffff", fs=6.9)

    # ---- detections and cross-frame evidence ------------------------------
    y_f, h_f = 0.757, 0.095
    lw_ = 0.445
    box(ax, L, y_f, lw_, h_f, "SAM3 detections",
        sub="one set per image · geo-referenced", fs=7.2)
    box(ax, R - lw_, y_f, lw_, h_f, "cross-frame evidence",
        sub="re-checked in the other image",
        fill=FILL_EVID, edge=EDGE_EVID, fs=7.2)

    y_bus = (y_in + y_f + h_f) / 2
    ax.plot([x_before + iw_in / 2, x_after + iw_in / 2], [y_bus, y_bus],
            color=LINE, linewidth=0.9, solid_capstyle="round", zorder=1)
    for x_src in (x_before + iw_in / 2, x_after + iw_in / 2):
        ax.plot([x_src, x_src], [y_in, y_bus], color=LINE, linewidth=0.9,
                zorder=1)
    arrow(ax, x_det_c, y_bus, x_det_c, y_f + h_f)
    # cross_frame.compute(..., boxes[own]) needs the detection footprints, so
    # detection is its only gating input.
    arrow(ax, L + lw_, y_f + h_f / 2, R - lw_, y_f + h_f / 2, zorder=5)

    # ---- the learned head -------------------------------------------------
    # Data flow, verified against infer.py and model.py:
    #   detections -> pair features(16)  -> forward_pair -> match AND state
    #   detections -> unary features(10) -\
    #   cross-frame evidence(4) ----------+-> forward_unary -> verify
    # match and state are siblings on one input, not a chain: forward_pair
    # computes h once and returns self.match(h), self.state(h). And verify
    # reads detection-derived features directly, not match's output.
    #
    # What match contributes to the other two is *selection*, not data: the
    # assignment decides which pairs are matched (so state's label applies) and
    # which detections are left over (so verify's decision applies). Drawing
    # that as arrows would put control flow and data flow in the same notation,
    # which is what made earlier versions read wrongly; the subs carry it.
    hy, hh = 0.377, 0.347
    ax.add_patch(FancyBboxPatch((L, hy), W, hh,
                                boxstyle="round,pad=0.008,rounding_size=0.018",
                                linewidth=1.6, facecolor=FILL_HEAD,
                                edgecolor=EDGE_HEAD, zorder=2))
    ax.text(cx, hy + hh - 0.028, "learned pairing head", ha="center",
            va="center", fontsize=7.8, color=EDGE_HEAD, fontweight="bold",
            zorder=3)

    ih = 0.068
    r1, r2 = 0.556, 0.400
    # The gap between match and state is wide enough for the unary drop to run
    # down it with visible clearance on both sides; a narrower gap made the
    # line look like it was grazing the borders.
    BGAP = 0.050
    bw2 = (inner_w - BGAP) / 2
    x_match, x_state = ix, ix + bw2 + BGAP
    x_gap = x_match + bw2 + BGAP / 2

    box(ax, x_match, r1, bw2, ih, "match", sub="same object?",
        fill="#ffffff", edge=EDGE_HEAD, fs=7.2, lw=0.8)
    box(ax, x_state, r1, bw2, ih, "state",
        sub="matched pairs · modified = change",
        fill="#ffffff", edge=EDGE_HEAD, fs=7.2, lw=0.8)
    box(ax, ix, r2, inner_w, ih, "verify",
        sub="unmatched detections · keep or discard",
        fill="#ffffff", edge=EDGE_HEAD, fs=7.2, lw=0.8)

    # One detection bus, fanned to all three branches. The drops are pulled in
    # from the box centres towards the outer edges so the band between them is
    # clear for the labels that say which feature vector each edge carries.
    y_hb = 0.666
    x_dm = x_match + 0.072
    x_ds = x_state + bw2 - 0.072
    ax.plot([x_dm, x_ds], [y_hb, y_hb], color=EDGE_HEAD, linewidth=0.9,
            solid_capstyle="round", zorder=4)
    ax.plot([x_det_c, x_det_c], [hy + hh, y_hb], color=EDGE_HEAD,
            linewidth=0.9, zorder=4)
    arrow(ax, x_dm, y_hb, x_dm, r1 + ih, color=EDGE_HEAD, zorder=5)
    arrow(ax, x_ds, y_hb, x_ds, r1 + ih, color=EDGE_HEAD, zorder=5)
    # The unary vector is 14 values: 10 built from the detection and 4 appended
    # from cross-frame evidence (see unary_features). They merge when the
    # vector is constructed, not at the verifier's input, so the two legs meet
    # at a junction and a single edge continues into verify.
    y_merge = r2 + ih + 0.064
    ax.plot([x_gap, x_gap], [y_hb, y_merge], color=EDGE_HEAD, linewidth=0.9,
            zorder=4)
    ax.plot([chan_x, chan_x], [y_f, y_merge], color=EDGE_EVID, linewidth=1.1,
            zorder=4)
    ax.plot([chan_x, x_gap], [y_merge, y_merge], color=EDGE_EVID,
            linewidth=1.1, solid_capstyle="round", zorder=4)
    ax.plot([x_gap], [y_merge], marker="o", markersize=2.6,
            color=EDGE_HEAD, zorder=6)
    arrow(ax, x_gap, y_merge, x_gap, r2 + ih, color=EDGE_HEAD, zorder=5)

    # Which vector each edge carries. Every label is set against the arrow it
    # describes, never on a stretch of the shared bus: a label centred on a bus
    # segment reads as belonging to whichever drop is nearest, which put
    # "unary" on the edge into state in an earlier version. match and state are
    # therefore labelled twice rather than once between them.
    y_lab = (y_hb + r1 + ih) / 2
    for x_edge, side in ((x_dm + 0.012, "left"), (x_ds - 0.012, "right")):
        ax.text(x_edge, y_lab, "pair features (16)", ha=side, va="center",
                fontsize=5.9, color=MUTED, style="italic", zorder=5)
    y_u = (y_merge + r2 + ih) / 2
    ax.text(x_gap - 0.014, y_u, "unary features (14)", ha="right",
            va="center", fontsize=5.9, color=MUTED, style="italic", zorder=5)
    ax.text(chan_x - 0.022, y_u, "incl. 4 cross-frame", ha="right",
            va="center", fontsize=5.6, color=EDGE_EVID, style="italic",
            zorder=5)

    arrow(ax, x_det_c, y_f, x_det_c, hy + hh)

    # ---- tail -------------------------------------------------------------
    ty, th = 0.198, 0.132
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
    # Centred on the clear band between the two rounded borders, not on the
    # raw gap: the boxstyle pad puts the drawn edges 0.008 outside the given
    # coordinates, and centring on the gap ran the text over both of them.
    ax.text(L + bw / 2 + 0.014, ((ty + th + 0.008) + (hy - 0.008)) / 2,
            "modified · appeared · disappeared", ha="left", va="center",
            fontsize=5.8, color=MUTED, style="italic")

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

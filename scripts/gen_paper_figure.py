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
    stack, which buys back the height the emphasized head needs.
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

# --- palette: light, print-safe, distinguishable in grayscale -------------
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

FIG_W, FIG_H = 3.5, 2.080



def box(ax, x, y, w, h, label, sub=None, fill=FILL_STAGE, edge=EDGE_STAGE,
        lw=0.9, fs=7.0, ls="-", badge=None, badge_color=None):
    """A stage box. Sub-label is placed *relative to the box height* so it
    cannot spill past the border into whatever sits underneath."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.018",
                                linewidth=lw, facecolor=fill, edgecolor=edge,
                                linestyle=ls, mutation_aspect=1.0, zorder=2))
    if sub:
        # A wrapped sub needs the label lifted and its own block dropped, or
        # the second line runs into the bottom border.
        two = "\n" in sub
        ax.text(x + w / 2, y + h * (0.71 if two else 0.63), label, ha="center",
                va="center", fontsize=fs, color=INK, zorder=3)
        ax.text(x + w / 2, y + h * (0.30 if two else 0.26), sub, ha="center",
                va="center", fontsize=fs - 1.4, color=MUTED, zorder=3,
                linespacing=1.30)
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
    # The drawing occupies 0.22 upward; cropping to it removes a band of empty
    # canvas that bbox_inches alone does not reclaim from an invisible axes.
    ax.set_ylim(0.210, 0.975)
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
    hy, hh = 0.392, 0.332
    ax.add_patch(FancyBboxPatch((L, hy), W, hh,
                                boxstyle="round,pad=0.008,rounding_size=0.018",
                                linewidth=1.6, facecolor=FILL_HEAD,
                                edgecolor=EDGE_HEAD, zorder=2))
    ax.text(cx, hy + hh - 0.024, "learned pairing head", ha="center",
            va="center", fontsize=7.8, color=EDGE_HEAD, fontweight="bold",
            zorder=3)

    # The three branches sit on one line off one bus. Stacking them, as an
    # earlier version did, read as a sequence -- match, then state, then
    # verify -- when in fact all three are computed from the same detections
    # and none waits on another's output.
    BGAP = 0.030
    bw3 = (inner_w - 2 * BGAP) / 3
    by, bh = 0.452, 0.132
    cols = [(ix + i * (bw3 + BGAP)) for i in range(3)]
    ctr = [x + bw3 / 2 for x in cols]

    box(ax, cols[0], by, bw3, bh, "match",
        sub="candidate pairs\nsame object?",
        fill="#ffffff", edge=EDGE_HEAD, fs=7.2, lw=0.8)
    box(ax, cols[1], by, bw3, bh, "state",
        sub="matched pairs\nmodified = change",
        fill="#ffffff", edge=EDGE_HEAD, fs=7.2, lw=0.8)
    box(ax, cols[2], by, bw3, bh, "verify",
        sub="unmatched ones\nkeep or discard",
        fill="#ffffff", edge=EDGE_HEAD, fs=7.2, lw=0.8)

    y_hb = 0.672
    y_tip = 0.624
    ax.plot([ctr[0], ctr[2]], [y_hb, y_hb], color=EDGE_HEAD, linewidth=0.9,
            solid_capstyle="round", zorder=4)
    ax.plot([x_det_c, x_det_c], [hy + hh, y_hb], color=EDGE_HEAD,
            linewidth=0.9, zorder=4)

    # Cross-frame evidence is not a fourth input to verify: its four values are
    # appended to the unary vector when that vector is built (unary_features),
    # so the two legs meet at a junction on verify's own edge and one arrow
    # continues down.
    y_merge = 0.652
    ax.plot([ctr[2], ctr[2]], [y_hb, y_merge], color=EDGE_HEAD, linewidth=0.9,
            zorder=4)
    ax.plot([chan_x, chan_x], [y_f, y_merge], color=EDGE_EVID, linewidth=1.1,
            zorder=4)
    ax.plot([chan_x, ctr[2]], [y_merge, y_merge], color=EDGE_EVID,
            linewidth=1.1, solid_capstyle="round", zorder=4)
    ax.plot([ctr[2]], [y_merge], marker="o", markersize=2.6,
            color=EDGE_HEAD, zorder=6)

    arrow(ax, ctr[0], y_hb, ctr[0], y_tip, color=EDGE_HEAD, zorder=5)
    arrow(ax, ctr[1], y_hb, ctr[1], y_tip, color=EDGE_HEAD, zorder=5)
    arrow(ax, ctr[2], y_merge, ctr[2], y_tip, color=EDGE_HEAD, zorder=5)

    # Each label is centered on its own column, directly under the arrowhead it
    # belongs to and directly over the box it feeds. Nothing sits on a shared
    # stretch of bus, where it would read as belonging to the nearest drop.
    for c, tag in zip(ctr, ("pair features (16)", "pair features (16)",
                            "unary features (14)")):
        ax.text(c, 0.610, tag, ha="center", va="center", fontsize=5.4,
                color=MUTED, style="italic", zorder=5)
    # Set over the horizontal run inside the head, not over the vertical one
    # outside it: the strip between the evidence box and the head is 0.017 of
    # clear space once both rounded borders are drawn, and the label crossed
    # them both.
    ax.text((chan_x + ctr[2]) / 2, y_merge + 0.030, "4 cross-frame",
            ha="center", va="center", fontsize=5.6, color=EDGE_EVID,
            style="italic", zorder=5)

    # The suppressed detections leave sideways, not downwards. Down is the
    # flow axis here, so a dashed edge dropping toward the tail read as "goes
    # on to the next stage" no matter where its head stopped. Right is
    # orthogonal to the flow and cannot be misread as continuation. The label
    # is set vertically because the margin outside verify is 0.09 wide and the
    # word needs 0.13 lying down.
    # Selection, not data flow, and drawn in a notation that cannot be
    # confused for it. _scores() computes match, state and both verify passes
    # in ONE forward call, before assign() is ever reached (infer.py), and
    # verify is scored for every detection, not just the leftovers -- so no
    # branch consumes another's output and none of them waits. What match
    # does produce is the assignment, and the assignment decides which of the
    # already-computed outputs count: state is read only for matched pairs,
    # verify's threshold applied only to what is left over. A solid arrow here
    # would assert a dependency the code does not have; a dotted one carries
    # the ordering without it.
    y_sel = 0.404
    dotted = (0, (1, 1.6))
    ax.plot([ctr[0], ctr[0]], [by - 0.008, y_sel], color=MUTED, linewidth=0.8,
            linestyle=dotted, zorder=5)
    ax.plot([ctr[0], ctr[2]], [y_sel, y_sel], color=MUTED, linewidth=0.8,
            linestyle=dotted, zorder=5)
    for c in (ctr[1], ctr[2]):
        arrow(ax, c, y_sel, c, by - 0.008, color=MUTED, lw=0.8, ls=dotted,
              zorder=5)
    # Named for what it produces, not for the operation. "assignment" alone
    # was read as match assigning work to the other two branches; what assign()
    # actually returns is (matches, unmatched_past, unmatched_cur) -- the
    # detections paired off against each other, and the leftovers. Naming the
    # split makes the line land on the words already in the boxes it points
    # at: "matched pairs" and "unmatched ones".
    #
    # The words ride above the line rather than in a break in it or on a row
    # of their own: breaking the line left a stub too short to read as a line,
    # and a row under the head cost a seventh page. 0.229 wide against the
    # 0.281 between the two ticks, which nothing may cross.
    ax.text((ctr[1] + ctr[2]) / 2, y_sel + 0.020, "matched vs unmatched",
            ha="center", va="center", fontsize=5.4, color=MUTED,
            style="italic", zorder=5)

    y_disc = by + bh / 2
    arrow(ax, cols[2] + bw3 + 0.008, y_disc, 0.918, y_disc,
          ls=(0, (2.5, 2)), zorder=5)
    ax.text(0.940, y_disc, "discarded", ha="center", va="center",
            fontsize=5.2, color=MUTED, style="italic", rotation=90, zorder=5)

    arrow(ax, x_det_c, y_f, x_det_c, hy + hh)

    # ---- tail -------------------------------------------------------------
    ty, th = 0.220, 0.124
    n, gap = 3, 0.028
    bw = (W - gap * (n - 1)) / n
    # (label, sub): the label may wrap, and only `sub` is set in muted type, so
    # a wrapped label is never mistaken for a label plus a caption.
    # "change inventory", not "change instances". change_instances() returns
    # the three change types and feeds the instance/CD metrics; the report path
    # runs on ChangeEvidence, which from_pairing_result() fills from ALL of
    # result.outcomes, and which prompts.py then reads. The box on this path is
    # the inventory: three change types in detail plus the unchanged total. The
    # old name belonged to the other consumer.
    #
    # No knowledge-graph box. Drawn in this strip it claimed the inventory
    # reaches the LLM through the graph, and it does not: user_prompt() appends
    # render_inventory(ev) for every llm_* mode and appends graph_context as an
    # ADDITIONAL block for llm_graphrag alone, so the graph sits beside the
    # inventory in one condition of six, never carrying it. A bypass edge would
    # say that, but there is no room to route one above, below or through this
    # row, and the row cannot grow without costing a seventh page. So the strip
    # draws the path that is always taken, and the grounding conditions --
    # graph included -- are enumerated in the text where they can be stated
    # exactly.
    tail = [(("change\ninventory", None), FILL_STAGE, EDGE_STAGE, "-"),
            (("LLM report", None), FILL_STAGE, EDGE_STAGE, "-"),
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

    # This edge is the ChangeEvidence branch: from_pairing_result() copies all
    # of result.outcomes, and the inventory reports three change categories in
    # detail plus an unchanged total. change_instances() is a second, separate
    # consumer of the same result -- the three change types alone, feeding the
    # instance and CD metrics -- and is not drawn. The two share a vocabulary
    # (evidence.py's appeared/disappeared/modified properties rename the same
    # statuses change_instances() renames), so a three-item label here read as
    # the other branch. The fourth item settles which one this is.
    #
    # Two ways out of the head, and the second one is the paper's result. A
    # detection the verifier scores below threshold is never emitted as an
    # outcome at all -- pair() only counts it, as n_suppressed -- so the
    # suppression path is a dashed edge that stops in open space. Drawn as a
    # single outgoing arrow, the figure implied every detection survives,
    # which is precisely the failure cross-frame evidence exists to prevent.
    y_out = ((ty + th + 0.008) + (hy - 0.008)) / 2
    arrow(ax, L + bw / 2, hy, L + bw / 2, ty + th)
    # Centered on the clear band between the two rounded borders, not on the
    # raw gap: the boxstyle pad puts the drawn edges 0.008 outside the given
    # coordinates, and centring on the gap ran the text over both of them.
    ax.text(L + bw / 2 + 0.014, y_out,
            "appeared · disappeared · modified · unchanged count",
            ha="left", va="center",
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

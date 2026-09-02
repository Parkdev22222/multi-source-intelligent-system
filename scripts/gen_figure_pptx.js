// Editable PowerPoint version of the paper's Fig. 1, for slides and reuse.
//
//     npm install pptxgenjs      # once
//     node scripts/gen_figure_pptx.js
//
// Everything is a native shape: rounded rectangles, lines and text boxes, so a
// reader can move a box or retype a label in PowerPoint. It is not an image of
// the figure.
//
// The geometry is transcribed from scripts/gen_paper_figure.py in that script's
// own coordinate system and mapped once, at fx/fy, so the two cannot drift
// silently -- a change there is a one-line change here rather than a redraw.
//
// LibreOffice cannot open any .pptx in the sandbox this was built in (a trivial
// control deck fails identically), so it was checked by reading the shape tree
// back out of the file: every arrow's endpoints and direction, every string,
// and that nothing lands off-slide. That check found the vertical arrows all
// pointing backwards, which no amount of looking at this source would have.
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";               // 13.333 x 7.5 in
const slide = pres.addSlide();

// ---- palette, verbatim from the generator -------------------------------
const INK = "1a1d23", MUTED = "6b7280", LINE = "9aa2af";
const FILL_STAGE = "f4f6f9", EDGE_STAGE = "b8c0cc";
const FILL_HEAD = "e8eefb", EDGE_HEAD = "3b6cc4";
const FILL_EVID = "fdf1e3", EDGE_EVID = "c8802a";
const EDGE_EVAL = "7a8492";

// ---- figure space -> slide inches ---------------------------------------
// The figure spans x in [0,1] and y in [0.210, 0.975], drawn 3.5in wide.
const X0 = 1.6165, Y0 = 1.05, DW = 10.10, DH = 6.00;
const YTOP = 0.975, YSPAN = 0.765;
const fx = x => X0 + x * DW;
const fy = y => Y0 + ((YTOP - y) / YSPAN) * DH;
const sx = w => w * DW;
const sy = h => (h / YSPAN) * DH;
const FS = 2.886;                          // 10.10in / 3.5in: point sizes scale with it
const pt = p => Math.round(p * FS * 10) / 10;

// ---- helpers ------------------------------------------------------------
function box(x, y, w, h, label, sub, fill, edge, fs, lw) {
  const runs = [{ text: label, options: { fontSize: pt(fs), color: INK, breakLine: !!sub } }];
  if (sub) {
    sub.split("\n").forEach((line, i, all) =>
      runs.push({ text: line, options: { fontSize: pt(fs - 1.4), color: MUTED, breakLine: i < all.length - 1 } }));
  }
  slide.addText(runs, {
    shape: pres.ShapeType.roundRect, rectRadius: 0.06,
    x: fx(x), y: fy(y + h), w: sx(w), h: sy(h),
    fill: { color: fill }, line: { color: edge, width: lw },
    align: "center", valign: "middle", margin: 0, fontFace: "Calibri", lineSpacingMultiple: 1.05,
  });
}

function seg(x1, y1, x2, y2, color, width, opts) {
  const o = Object.assign({ color, width }, opts || {});
  // A line with no flipV runs from the box's top-left to its bottom-right, so
  // it already goes from the LARGER figure y to the smaller: slide y grows
  // downward where figure y grows up. The flip is needed for upward arrows,
  // not downward ones -- inverting this pointed every vertical arrow backwards.
  const flipH = x2 < x1, flipV = y2 > y1;
  slide.addShape(pres.ShapeType.line, {
    x: fx(Math.min(x1, x2)), y: fy(Math.max(y1, y2)),
    w: sx(Math.abs(x2 - x1)), h: sy(Math.abs(y2 - y1)),
    line: o, flipH, flipV,
  });
}
const arrow = (x1, y1, x2, y2, color, width, opts) =>
  seg(x1, y1, x2, y2, color, width, Object.assign({ endArrowType: "triangle" }, opts || {}));

function label(x, y, text, fs, color, opts) {
  // Width tracks the text: a fixed wide box makes neighbouring labels overlap
  // as click targets in PowerPoint, which is invisible until you try to edit.
  const w = Math.max(0.8, text.length * pt(fs) * 0.5 / 72 + 0.25);
  slide.addText(text, Object.assign({
    x: fx(x) - w / 2, y: fy(y) - 0.16, w, h: 0.32,
    fontSize: pt(fs), color, italic: true, align: "center", valign: "middle",
    margin: 0, isTextBox: true, fontFace: "Calibri",
  }, opts || {}));
}

// ---- title --------------------------------------------------------------
slide.addText("Vision-grounded change reports: pipeline", {
  x: 0.6, y: 0.34, w: 12.1, h: 0.5, fontSize: 26, bold: true,
  color: INK, align: "left", margin: 0, isTextBox: true, fontFace: "Calibri",
});

// ---- geometry, transcribed ----------------------------------------------
const L = 0.03, R = 0.97, cx = 0.5;
const ix = 0.056, inner_w = 0.813, chan_x = 0.9065;
const y_in = 0.900, h_in = 0.062, iw_in = 0.205;
const x_det_c = 0.2525, x_before = 0.0365, x_after = 0.2635;
const y_f = 0.757, h_f = 0.095, lw_ = 0.445;
const y_bus = 0.876;
const hy = 0.392, hh = 0.332;
const BGAP = 0.030, bw3 = 0.251;
const cols = [0.056, 0.337, 0.618];
const ctr = cols.map(c => c + bw3 / 2);
const by = 0.452, bh = 0.132;
const y_hb = 0.672, y_tip = 0.624, y_merge = 0.652;
const y_sel = 0.404, y_disc = by + bh / 2;
const ty = 0.220, th = 0.124, gap = 0.028, bw = 0.2947;
const y_out = ((ty + th + 0.008) + (hy - 0.008)) / 2;

// ---- inputs -------------------------------------------------------------
box(x_before, y_in, iw_in, h_in, "before image", null, "FFFFFF", EDGE_STAGE, 6.9, 0.9);
box(x_after, y_in, iw_in, h_in, "after image", null, "FFFFFF", EDGE_STAGE, 6.9, 0.9);

// ---- detections and cross-frame evidence --------------------------------
box(L, y_f, lw_, h_f, "SAM3 detections", "one set per image · geo-referenced",
    FILL_STAGE, EDGE_STAGE, 7.2, 0.9);
box(R - lw_, y_f, lw_, h_f, "cross-frame evidence", "re-checked in the other image",
    FILL_EVID, EDGE_EVID, 7.2, 0.9);

seg(x_before + iw_in / 2, y_bus, x_after + iw_in / 2, y_bus, LINE, 1.2);
seg(x_before + iw_in / 2, y_in, x_before + iw_in / 2, y_bus, LINE, 1.2);
seg(x_after + iw_in / 2, y_in, x_after + iw_in / 2, y_bus, LINE, 1.2);
arrow(x_det_c, y_bus, x_det_c, y_f + h_f, LINE, 1.2);
arrow(L + lw_, y_f + h_f / 2, R - lw_, y_f + h_f / 2, LINE, 1.2);
arrow(x_det_c, y_f, x_det_c, hy + hh, LINE, 1.2);

// ---- the head panel, then its contents ----------------------------------
slide.addShape(pres.ShapeType.roundRect, {
  x: fx(L), y: fy(hy + hh), w: sx(R - L), h: sy(hh), rectRadius: 0.05,
  fill: { color: FILL_HEAD }, line: { color: EDGE_HEAD, width: 2.0 },
});
slide.addText("learned pairing head", {
  x: fx(cx) - 2.5, y: fy(hy + hh - 0.024) - 0.18, w: 5.0, h: 0.36,
  fontSize: pt(7.8), bold: true, color: EDGE_HEAD, align: "center", valign: "middle",
  margin: 0, isTextBox: true, fontFace: "Calibri",
});

box(cols[0], by, bw3, bh, "match", "candidate pairs\nsame object?", "FFFFFF", EDGE_HEAD, 7.2, 1.0);
box(cols[1], by, bw3, bh, "state", "matched pairs\nmodified = change", "FFFFFF", EDGE_HEAD, 7.2, 1.0);
box(cols[2], by, bw3, bh, "verify", "unmatched ones\nkeep or discard", "FFFFFF", EDGE_HEAD, 7.2, 1.0);

// one detection bus, fanned to all three branches
seg(ctr[0], y_hb, ctr[2], y_hb, EDGE_HEAD, 1.2);
seg(x_det_c, hy + hh, x_det_c, y_hb, EDGE_HEAD, 1.2);
arrow(ctr[0], y_hb, ctr[0], y_tip, EDGE_HEAD, 1.2);
arrow(ctr[1], y_hb, ctr[1], y_tip, EDGE_HEAD, 1.2);

// cross-frame joins the unary vector at a junction, not at verify's input
seg(ctr[2], y_hb, ctr[2], y_merge, EDGE_HEAD, 1.2);
seg(chan_x, y_f, chan_x, y_merge, EDGE_EVID, 1.5);
seg(chan_x, y_merge, ctr[2], y_merge, EDGE_EVID, 1.5);
arrow(ctr[2], y_merge, ctr[2], y_tip, EDGE_HEAD, 1.2);
slide.addShape(pres.ShapeType.ellipse, {
  x: fx(ctr[2]) - 0.055, y: fy(y_merge) - 0.055, w: 0.11, h: 0.11,
  fill: { color: EDGE_HEAD }, line: { color: EDGE_HEAD, width: 0.5 },
});

["pair features (16)", "pair features (16)", "unary features (14)"].forEach((tag, i) =>
  label(ctr[i], 0.610, tag, 5.4, MUTED));
label(0.825, y_merge + 0.030, "4 cross-frame", 5.6, EDGE_EVID);

// selection, not data flow: match's assignment picks each branch's subset
const dot = { dashType: "sysDot" };
seg(ctr[0], by - 0.008, ctr[0], y_sel, MUTED, 1.0, dot);
seg(ctr[0], y_sel, ctr[2], y_sel, MUTED, 1.0, { dashType: "sysDot" });
arrow(ctr[1], y_sel, ctr[1], by - 0.008, MUTED, 1.0, { dashType: "sysDot" });
arrow(ctr[2], y_sel, ctr[2], by - 0.008, MUTED, 1.0, { dashType: "sysDot" });
label((ctr[1] + ctr[2]) / 2, y_sel + 0.020, "matched vs unmatched", 5.4, MUTED);

// suppressed detections leave sideways, off the flow axis
arrow(cols[2] + bw3 + 0.008, y_disc, 0.918, y_disc, LINE, 1.1, { dashType: "dash" });
slide.addText("discarded", {
  x: fx(0.940) - 0.60, y: fy(y_disc) - 0.16, w: 1.2, h: 0.32,
  fontSize: pt(5.4), color: MUTED, italic: true, align: "center", valign: "middle",
  margin: 0, isTextBox: true, fontFace: "Calibri", rotate: 270,
});

// ---- tail ---------------------------------------------------------------
const tail = [
  ["change\ninventory", FILL_STAGE, EDGE_STAGE, false],
  ["LLM report", FILL_STAGE, EDGE_STAGE, false],
  ["Change-\nFact-Score", "FFFFFF", EDGE_EVAL, true],
];
tail.forEach(([text, fill, edge, dashed], i) => {
  const x = L + i * (bw + gap);
  slide.addText(text.split("\n").map((t, j, all) =>
      ({ text: t, options: { fontSize: pt(6.9), color: INK, breakLine: j < all.length - 1 } })), {
    shape: pres.ShapeType.roundRect, rectRadius: 0.06,
    x: fx(x), y: fy(ty + th), w: sx(bw), h: sy(th),
    fill: { color: fill },
    line: dashed ? { color: edge, width: 1.2, dashType: "dash" } : { color: edge, width: 1.2 },
    align: "center", valign: "middle", margin: 0, fontFace: "Calibri", lineSpacingMultiple: 1.05,
  });
  if (i) arrow(x - gap, ty + th / 2, x, ty + th / 2, LINE, 1.2,
               i === tail.length - 1 ? { dashType: "dash" } : {});
});

arrow(L + bw / 2, hy, L + bw / 2, ty + th, LINE, 1.2);
slide.addText("appeared · disappeared · modified · unchanged count", {
  x: fx(L + bw / 2 + 0.014), y: fy(y_out) - 0.16, w: 5.2, h: 0.32,
  fontSize: pt(5.8), color: MUTED, italic: true, align: "left", valign: "middle",
  margin: 0, isTextBox: true, fontFace: "Calibri",
});

slide.addNotes(
  "Editable rebuild of Fig. 1. match and state share one 16-dimensional pair feature; " +
  "verify reads a 14-dimensional unary feature whose last four entries are cross-frame " +
  "evidence. Dotted is selection, not data: all three are scored in one pass, and match's " +
  "assignment then splits the detections into paired and leftover. The dashed edge leaving " +
  "verify is suppression, and most of the detection gain is there.");

pres.writeFile({ fileName: "docs/method_figure.pptx" }).then(f => console.log("wrote", f));

#!/usr/bin/env python3
"""
Build the paper tables the harness does not emit on its own.

`run_report_eval` writes one table per run directory. E5 compares two runs
(learned pairing against heuristic pairing) and so spans two directories, and
E7's scene results ship as JSON without a LaTeX writer. Both are assembled
here, from the JSON, so no number in the paper is typed by hand.

`build_detection` additionally folds what the harness emits as four tables
(pixel and instance, on each of two datasets) into the one table a six-page
paper has room for. It reads the same result JSON the four tables are written
from, so the numbers are identical; only the layout differs.

    python paper/make_tables.py

Writes paper/tables/{detection,e5_pairing,e7_scene}.tex. Missing inputs are
reported and skipped rather than faked, so running this before the full-scale
runs land leaves the earlier table in place and says which one it could not
rebuild.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "tables"

# E5: same grounding, pairing swapped. Order puts ours last so the table's
# emphasis falls where the argument does.
E5_RUNS = [
    ("production heuristic", "results/levir_cc_caption_heuristic_pairing",
     "results/e5_heuristic_pairing"),
    ("learned head (ours)", "results/levir_cc_caption",
     "results/e5_learned_pairing"),
]
E5_MODE = "llm_graphrag"

E7_RUN = ["results/levir_cc_scene", "results/pilot_cc_scenes8/scene_level"]
E7_ORDER = ["template", "llm_struct", "llm_flat_rag", "llm_graphrag"]
E7_LABELS = {"template": "template (no LLM)", "llm_struct": "full concatenation",
             "llm_flat_rag": "top-$k$ retrieval", "llm_graphrag": "graph aggregate"}


def _first_existing(*rels: str) -> Path | None:
    for rel in rels:
        if rel and (ROOT / rel).is_dir():
            return ROOT / rel
    return None


def _rows(path: Path, name: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {r["name"]: r for r in payload.get("results", [])}, payload


def _pct(v) -> str:
    return "--" if v is None else f"{v * 100:.2f}"


def _abs(v) -> str:
    """A count, not a rate: CountMAE is an instance count and is not scaled."""
    return "--" if v is None else f"{v:.1f}"


# Detection: LEVIR-CD and WHU-CD, pixel and instance, in one table. The
# harness writes these as four separate tables; at six pages they have to share
# one. Row order matches the argument: non-learned strategies first so their
# convergence is visible, ablations next, ours last.
DET_RUNS = [
    ("LEVIR-CD (128 tiles)", "results/levir_cd_test/cd_results.json", "levir_cd"),
    ("WHU-CD (690 pairs, no retraining)", "results/whu_cd_test/cd_results.json", None),
]
DET_ORDER = [
    ("geo-only", "geometry only"),
    ("heuristic (production)", "CLIP + geometry heuristic"),
    ("learned head, no verifier", "\\quad ablation: no verifier"),
    ("learned head, no cross-frame", "\\quad ablation: no cross-frame"),
    ("learned head (ours)", "\\textbf{learned head (ours)}"),
]


def build_detection() -> str | None:
    """One detection table spanning both datasets and both granularities."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from icce.eval.tables import load_baselines

    blocks, provenance = [], []
    for title, rel, baseline_group in DET_RUNS:
        path = ROOT / rel
        if not path.is_file():
            print(f"detection: missing {rel}; table not rebuilt")
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_name = {r["name"]: r for r in payload["results"]}
        provenance.append(f"%%   {title}: {rel} "
                          f"(n={payload.get('n_pairs')}, "
                          f"integrity clean={payload.get('integrity', {}).get('clean')})")

        lines = [f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\"]

        # Published zero-shot peers, verified rows only, for the dataset that has them.
        if baseline_group:
            for m in load_baselines().get(baseline_group, {}).get("methods", []):
                if m.get("tier") != "zero-shot" or not m.get("verified"):
                    continue
                lines.append(
                    f"{m['name']}~\\cite{{{m['cite']}}} & {_pct(m.get('f1'))} & "
                    f"{_pct(m.get('iou'))} & -- & -- \\\\")
                provenance.append(f"%%   baseline {m['name']}: {m.get('source', 'baselines.json')}")

        for key, label in DET_ORDER:
            r = by_name.get(key)
            if r is None:
                continue
            cells = [_pct(r.get("f1")), _pct(r.get("iou")), _pct(r.get("inst_f1")),
                     _abs(r.get("instance", {}).get("count_mae"))]
            if key == "learned head (ours)":
                cells = [f"\\textbf{{{c}}}" for c in cells]
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        blocks.append("\n".join(lines))

    body = "\n\\midrule\n".join(blocks)
    return "\n".join([
        "%% Generated by paper/make_tables.py -- do not edit.",
        *provenance,
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Change detection. Pixel F1 and IoU are the change class over "
        "the whole test set; instance F1 matches predicted change instances to "
        "ground-truth connected components at IoU~$\\geq$~0.5; Count MAE is the "
        "mean absolute error in the number of change instances per tile, in "
        "instances rather than a percentage. AnyChange "
        "provides published zero-shot context, not a supervision-matched "
        "comparison: our head uses LEVIR-CD training masks but no correspondence "
        "labels. The same head is applied to WHU-CD unchanged.}",
        "\\label{tab:detection}",
        "\\setlength{\\tabcolsep}{3pt}",
        # Five columns plus a cited baseline name overruns the IEEEtran column
        # at \small; measured 34pt too wide before this line.
        "\\scriptsize",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Pixel F1 & IoU & Inst.\\ F1 & Count MAE \\\\",
        "\\midrule",
        body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def build_e5() -> str | None:
    rows, provenance = [], []
    for label, preferred, fallback in E5_RUNS:
        run = _first_existing(preferred, fallback)
        if run is None:
            print(f"  E5: no run directory for {label!r} "
                  f"(looked for {preferred}, {fallback})")
            return None
        by_mode, payload = _rows(run / "report_results_caption.json", label)
        if E5_MODE not in by_mode:
            print(f"  E5: {run} has no {E5_MODE} row")
            return None
        rows.append((label, by_mode[E5_MODE]))
        provenance.append(f"{label}: {run.relative_to(ROOT)} "
                          f"(n={payload.get('n_pairs')}, "
                          f"checkpoint={payload.get('checkpoint')})")

    n = {r.get("n_generated") for _, r in rows}
    lines = [
        "% generated by paper/make_tables.py -- do not edit",
        *[f"% {p}" for p in provenance],
        "\\begin{table}[t]", "\\centering",
        "\\caption{Report factuality with grounding held fixed at "
        "\\texttt{llm\\_graphrag} and only the pairing swapped (E5). "
        "Identical crops, style examples and language model.}",
        "\\label{tab:e5_pairing}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccccc}", "\\toprule",
        "Pairing & CFS-P & CFS-R & CFS-F1 & Hal & ChgAcc \\\\", "\\midrule",
    ]
    for i, (label, r) in enumerate(rows):
        cells = [_pct(r.get(k)) for k in ("cfs_precision", "cfs_recall", "cfs_f1",
                                          "hallucination_rate", "change_accuracy")]
        if i == len(rows) - 1:
            label = f"\\textbf{{{label}}}"
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    if len(n) > 1:
        print(f"  E5: WARNING the two arms generated different counts: {n}")
    return "\n".join(lines)


def build_e7() -> str | None:
    run = _first_existing(*E7_RUN)
    if run is None:
        print(f"  E7: no run directory (looked for {', '.join(E7_RUN)})")
        return None
    payload = json.loads((run / "scene_results.json").read_text(encoding="utf-8"))
    by_name = {r["name"]: r for r in payload.get("results", [])}
    n_scenes = payload.get("n_scenes")
    gt = payload.get("gt_instances_per_scene") or {}
    mean_gt = sum(gt.values()) / len(gt) if gt else None

    # A tile is cut into 16 crops, but the test split does not carry all 16 for
    # every tile -- many scenes contribute one. Saying "all 16" would overstate
    # what the conditions were given, and on a single-crop scene the three
    # representations are trivially identical, so the spread is worth printing.
    cps = payload.get("crops_per_scene") or {}
    caption = ("Neighborhood-level grounding (E7): one report per "
               "neighborhood, over every crop of a tile the split provides. "
               "The conditions are three representations of the same "
               "observations, not an additive ladder.")
    if n_scenes is not None:
        caption += f" $n={n_scenes}$"
        if cps:
            vals = sorted(cps.values())
            multi = sum(1 for v in vals if v > 1)
            caption += (f", {min(vals)}--{max(vals)} crops each "
                        f"(mean {sum(vals) / len(vals):.1f}; {multi} scenes "
                        f"with more than one crop)")
        if mean_gt is not None:
            caption += f", mean {mean_gt:.1f} ground-truth instances per scene"
        caption += "."

    lines = [
        "% generated by paper/make_tables.py -- do not edit",
        f"% source: {run.relative_to(ROOT)} "
        f"(checkpoint={payload.get('checkpoint')})",
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:e7_scene}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "Representation & CFS-F1 & Hal & scene CountMAE \\\\", "\\midrule",
    ]
    for name in E7_ORDER:
        r = by_name.get(name)
        if r is None:
            continue
        mae = r.get("scene_count_mae")
        lines.append(f"{E7_LABELS.get(name, name)} & {_pct(r.get('cfs_f1'))} & "
                     f"{_pct(r.get('hallucination_rate'))} & "
                     f"{'--' if mae is None else f'{mae:.2f}'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    if payload.get("checkpoint") in (None, "None"):
        print("  E7: WARNING this run records no checkpoint -- it predates "
              "pairing_head.pt and its rows describe the pilot head")
    return "\n".join(lines)


# Crop-level factuality. The harness writes this table too, but without the
# CountMAE column: `report_results_caption.json` carries a crop-level
# `count_mae` per condition that no table printed, while the abstract and
# Section III-C both promise counting is scored separately. It is rebuilt here
# so the promised column exists, and `Hal` is dropped rather than widened into
# a sixth metric -- the paper defines it as 1 - CFS-P, so the table loses no
# measurement, and CFS-P is already in the first column.
E4_RUN = "results/levir_cc_caption"
E4_ORDER = ["template", "vlm_direct", "llm_raw", "llm_struct",
            "llm_flat_rag", "llm_graphrag"]


def _mono(name: str) -> str:
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def build_factuality() -> str | None:
    run = _first_existing(E4_RUN)
    if run is None:
        print(f"  E4: no run directory (looked for {E4_RUN})")
        return None
    payload = json.loads((run / "report_results_caption.json").read_text(encoding="utf-8"))
    by_name = {r["name"]: r for r in payload.get("results", [])}
    missing = [n for n in E4_ORDER if n not in by_name]
    if missing:
        print(f"  E4: run is missing conditions {missing}; table not rebuilt")
        return None

    lines = [
        "% generated by paper/make_tables.py -- do not edit",
        f"% source: {run.relative_to(ROOT)} (n={payload.get('n_pairs')}, "
        f"checkpoint={payload.get('checkpoint')})",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Report factuality by grounding condition, over "
        f"{payload.get('n_pairs')} LEVIR-CC test crops. CFS-P/R/F1 are "
        "claim-level against the human references; ChgAcc is agreement on "
        "whether anything changed; CountMAE is the crop-level absolute error "
        "in the asserted instance count, in instances. Hallucination rate is "
        "$1-\\text{CFS-P}$ and is not printed as a separate column.}",
        "\\label{tab:factuality_caption}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\small",
        "\\begin{tabular}{lccccc}", "\\toprule",
        "Condition & CFS-P & CFS-R & CFS-F1 & ChgAcc & CountMAE \\\\",
        "\\midrule",
    ]
    for i, name in enumerate(E4_ORDER):
        r = by_name[name]
        cells = [_pct(r.get("cfs_precision")), _pct(r.get("cfs_recall")),
                 _pct(r.get("cfs_f1")), _pct(r.get("change_accuracy")),
                 _abs(r.get("count_mae"))]
        label = _mono(name)
        if i == len(E4_ORDER) - 1:
            label = f"\\textbf{{{label}}}"
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# One worked example of the failure the pipeline exists to avoid. The crop is
# named rather than chosen at build time so the table is reproducible: it is
# one of the crops whose five references all say nothing changed, where
# vlm_direct asserts a change and the grounded pipeline does not.
EXAMPLE_KEY = "test_002021"
EXAMPLE_CAPTIONS = "data/benchmarks/LEVIR-CC/LevirCCcaptions.json"
EXAMPLE_ROWS = [
    ("human reference", None),
    ("vlm\\_direct", "results/levir_cc_caption_vlm/gen_vlm_direct_caption.jsonl"),
    ("llm\\_graphrag (ours)", "results/levir_cc_caption/gen_llm_graphrag_caption.jsonl"),
]


def _generation(rel: str, key: str) -> str | None:
    path = ROOT / rel
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("key") == key:
            return rec.get("text", "").strip()
    return None


def build_example() -> str | None:
    caps_path = ROOT / EXAMPLE_CAPTIONS
    if not caps_path.is_file():
        print(f"  example: missing {EXAMPLE_CAPTIONS}")
        return None
    refs = None
    for im in json.loads(caps_path.read_text(encoding="utf-8"))["images"]:
        if im["filename"].replace(".png", "") == EXAMPLE_KEY:
            refs = [s["raw"].strip() for s in im["sentences"]]
            break
    if not refs:
        print(f"  example: {EXAMPLE_KEY} not in the caption file")
        return None

    body = []
    for label, rel in EXAMPLE_ROWS:
        text = refs[0] if rel is None else _generation(rel, EXAMPLE_KEY)
        if text is None:
            print(f"  example: no generation for {label} in {rel}")
            return None
        text = " ".join(text.split())
        for punct in (".", ",", ";"):
            text = text.replace(" " + punct, punct)
        text = text.replace("&", "\\&").replace("%", "\\%")
        body.append((label, text))

    key = EXAMPLE_KEY.replace("_", "\\_")
    return "\n".join([
        "% generated by paper/make_tables.py -- do not edit",
        f"% crop {EXAMPLE_KEY}; reference from {EXAMPLE_CAPTIONS}",
        "\\begin{table}[t]", "\\centering",
        "\\caption{The invention failure on one test crop "
        f"(\\texttt{{{key}}}), whose five human references all state that "
        "nothing changed. The vision--language baseline names a specific "
        "object and a specific event; the grounded pipeline, having no "
        "surviving change instance to report, does not.}",
        "\\label{tab:example}",
        "\\small",
        "\\begin{tabular}{@{}p{0.30\\columnwidth}p{0.64\\columnwidth}@{}}",
        "\\toprule",
        "Source & Report \\\\",
        "\\midrule",
        *[f"{label} & {text} \\\\" for label, text in body],
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    built, skipped = [], []
    for name, fn in (("detection", build_detection),
                     ("factuality", build_factuality),
                     ("e5_pairing", build_e5), ("e7_scene", build_e7),
                     ("example", build_example)):
        print(f"{name}:")
        text = fn()
        if text is None:
            skipped.append(name)
            continue
        (OUT / f"{name}.tex").write_text(text + "\n", encoding="utf-8")
        print(f"  wrote {(OUT / f'{name}.tex').relative_to(ROOT)}")
        built.append(name)
    if skipped:
        print(f"\nnot rebuilt: {', '.join(skipped)} -- the previous .tex, if "
              f"any, is unchanged")
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())

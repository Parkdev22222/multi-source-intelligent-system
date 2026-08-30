"""
Result serialisation: JSON for the record, LaTeX for the paper.

Two guarantees this module enforces.

**Nothing unverified is typeset.** Published numbers live in `baselines.json`
with `verified: false` and no values. `latex_table` refuses to render an
unverified row and emits a loud TODO instead, so a number that was never
checked against its source paper cannot end up in the submission.

**Nothing is compared across leagues without saying so.** Methods carry a
`tier`. A model trained on LEVIR-CD's own training split and an open-vocabulary
model that has never seen it are not competing on equal terms, and printing
them in one undifferentiated block is misleading in whichever direction the
numbers happen to fall. Tables group by tier and label each group, so the
supervised block reads as a reference ceiling rather than as our opponent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

BASELINES_PATH = Path(__file__).with_name("baselines.json")


def load_baselines() -> Dict:
    return json.loads(BASELINES_PATH.read_text(encoding="utf-8"))


def _fmt(v: Optional[float], scale: float = 100.0, digits: int = 2) -> str:
    if v is None:
        return "--"
    return f"{v * scale:.{digits}f}"


# Characters LaTeX will not typeset as themselves. Backslash is absent on
# purpose: a label that already contains one is taken to be LaTeX the caller
# wrote deliberately, and is passed through untouched.
_TEX_ESCAPES = {"&": r"\&", "%": r"\%", "#": r"\#", "$": r"\$"}


def _tex_label(name: object, mono: bool = False) -> str:
    r"""Render a row or column label so pdflatex accepts it.

    Condition names reach this module as the identifiers the pipeline uses --
    ``vlm_direct``, ``llm_graphrag`` -- and an unescaped underscore is a fatal
    error, not a cosmetic one ("Missing $ inserted"). Identifiers are set in
    ``\texttt`` with the underscore escaped, which is also how ``method.tex``
    names these conditions in prose; published method names, which carry no
    underscore, are left as they read in their papers.
    """
    s = str(name)
    if "\\" in s:
        return s
    for ch, rep in _TEX_ESCAPES.items():
        s = s.replace(ch, rep)
    if mono or "_" in s:
        return "\\texttt{" + s.replace("_", r"\_") + "}"
    return s


def _to_dict(obj) -> Dict:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    return dict(obj)


def save_json(payload: Dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


TIER_HEADINGS = {
    "supervised": r"\multicolumn{%d}{l}{\emph{Trained on this dataset "
                  r"(reference ceiling)}} \\",
    "zero-shot": r"\multicolumn{%d}{l}{\emph{Zero-shot / open-vocabulary}} \\",
    "ours": r"\multicolumn{%d}{l}{\emph{This work}} \\",
}
TIER_ORDER = ("supervised", "zero-shot", "ours")


def latex_table(
    caption: str,
    label: str,
    columns: Sequence[str],
    rows: Sequence[Dict],
    baseline_group: Optional[str] = None,
    metric_map: Optional[Dict[str, str]] = None,
    scale: float = 100.0,
    highlight_last: bool = True,
    group_by_tier: bool = True,
    mono_rows: bool = False,
) -> str:
    """Build a booktabs table, grouped by tier.

    `rows` are dicts with a 'name' plus the metric keys named in `columns`.
    `metric_map` renames a column to the key used in baselines.json.
    `mono_rows` sets our own row names in \\texttt: the grounding conditions are
    identifiers the pipeline uses, and `method.tex` names them that way, so
    `template` should not be typeset differently from `llm_raw` just because it
    happens to contain no underscore. Baseline names are never affected --
    those are published method names.
    """
    metric_map = metric_map or {}
    lines: List[str] = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        # A five-metric table plus a cited baseline name overruns the IEEEtran
        # column even at 3pt with \footnotesize, so wide tables are set in
        # \scriptsize. Narrower tables are unaffected; this keys off the column
        # count rather than the caller, so a regenerated table cannot silently
        # start overflowing when a baseline row is added.
        f"\\setlength{{\\tabcolsep}}{{{3 if len(columns) >= 5 else 4}pt}}",
        *(["\\scriptsize"] if len(columns) >= 5 else []),
        "\\begin{tabular}{l" + "c" * len(columns) + "}",
        "\\toprule",
        "Method & " + " & ".join(_tex_label(c) for c in columns) + " \\\\",
        "\\midrule",
    ]

    n_cols = len(columns) + 1

    if baseline_group:
        group = load_baselines().get(baseline_group, {})
        methods = group.get("methods", [])
        unverified: List[str] = []

        by_tier: Dict[str, List[Dict]] = {}
        for m in methods:
            if not m.get("verified"):
                unverified.append(f"{m['name']} [{m.get('tier', '?')}]")
                continue
            by_tier.setdefault(m.get("tier", "supervised"), []).append(m)

        for tier in TIER_ORDER:
            entries = by_tier.get(tier)
            if not entries:
                continue
            if group_by_tier:
                lines.append(TIER_HEADINGS[tier] % n_cols)
            for m in entries:
                cells = [_fmt(m.get(metric_map.get(c, c)), scale) for c in columns]
                lines.append(f"{_tex_label(m['name'])}~\\cite{{{m['cite']}}} & "
                             + " & ".join(cells) + " \\\\")
            lines.append("\\midrule")

        if unverified:
            msg = ", ".join(unverified)
            logger.warning("[%s] unverified baselines omitted: %s", baseline_group, msg)
            lines.append(f"%% TODO fill and verify in baselines.json: {msg}")
            lines.append("\\midrule")

    if group_by_tier and rows:
        lines.append(TIER_HEADINGS["ours"] % (len(columns) + 1))

    for i, r in enumerate(rows):
        cells = [_fmt(r.get(metric_map.get(c, c)), scale) for c in columns]
        name = _tex_label(r.get("name", "?"), mono=mono_rows)
        if highlight_last and i == len(rows) - 1:
            name = f"\\textbf{{{name}}}"
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def save_latex(text: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def print_console_table(title: str, columns: Sequence[str], rows: Sequence[Dict],
                        scale: float = 100.0) -> str:
    """Human-readable table for the terminal / run logs."""
    width = max([len(str(r.get("name", ""))) for r in rows] + [6]) + 2
    out = [f"\n{title}", "-" * (width + 10 * len(columns))]
    out.append("method".ljust(width) + "".join(c.rjust(10) for c in columns))
    for r in rows:
        cells = "".join(_fmt(r.get(c), scale).rjust(10) for c in columns)
        out.append(str(r.get("name", "?")).ljust(width) + cells)
    text = "\n".join(out)
    print(text)
    return text

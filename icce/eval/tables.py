"""
Result serialisation: JSON for the record, LaTeX for the paper.

Published baseline numbers live in `baselines.json` with `verified: false` and
no values. `latex_table` refuses to typeset an unverified row and emits a loud
TODO comment instead, so a number that was never checked against its source
paper cannot silently end up in the submission.
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


def latex_table(
    caption: str,
    label: str,
    columns: Sequence[str],
    rows: Sequence[Dict],
    baseline_group: Optional[str] = None,
    metric_map: Optional[Dict[str, str]] = None,
    scale: float = 100.0,
    highlight_last: bool = True,
) -> str:
    """Build a booktabs table: published baselines above, our rows below.

    `rows` are dicts with a 'name' plus the metric keys named in `columns`.
    `metric_map` renames a column to the key used in baselines.json.
    """
    metric_map = metric_map or {}
    lines: List[str] = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + "c" * len(columns) + "}",
        "\\toprule",
        "Method & " + " & ".join(columns) + " \\\\",
        "\\midrule",
    ]

    if baseline_group:
        group = load_baselines().get(baseline_group, {})
        unverified: List[str] = []
        for m in group.get("methods", []):
            if not m.get("verified"):
                unverified.append(m["name"])
                continue
            cells = [_fmt(m.get(metric_map.get(c, c)), scale) for c in columns]
            lines.append(f"{m['name']}~\\cite{{{m['cite']}}} & " + " & ".join(cells) + " \\\\")
        if unverified:
            msg = ", ".join(unverified)
            logger.warning("[%s] unverified baselines omitted: %s", baseline_group, msg)
            lines.append(f"%% TODO fill and verify in baselines.json: {msg}")
        lines.append("\\midrule")

    for i, r in enumerate(rows):
        cells = [_fmt(r.get(metric_map.get(c, c)), scale) for c in columns]
        name = r.get("name", "?")
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

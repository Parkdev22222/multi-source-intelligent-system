"""
Evaluation integrity checks.

A result that wins because the evaluation leaked is worth less than no result:
it survives review and then fails when someone reruns it. These checks run
automatically before every evaluation and abort on a violation, so the failure
mode is a stopped run rather than a number in a submitted PDF.

What is checked
  1. **Split overlap.** No pair id may appear in both the training cache and the
     evaluation cache. LEVIR-CC crops share parent scenes with LEVIR-CD tiles,
     so a careless split gives the head tiles it trained on.
  2. **Scene overlap.** Stricter: crops of one parent scene must not straddle
     train and test. Two 256px crops of the same 1024px tile are near-duplicates.
  3. **Threshold provenance.** Match and verify thresholds must have been chosen
     on a split whose name is not the evaluation split.
  4. **Style-example provenance.** Caption exemplars must come from train.

Violations raise `IntegrityError`. `--allow-leakage` exists only for debugging
on a single split and stamps the result JSON so a leaked run can never be
mistaken for a clean one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)


class IntegrityError(RuntimeError):
    pass


@dataclass
class IntegrityReport:
    checks: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    waived: bool = False

    @property
    def clean(self) -> bool:
        return not self.violations

    def as_dict(self) -> Dict:
        return {"checks_passed": self.checks, "violations": self.violations,
                "waived": self.waived, "clean": self.clean and not self.waived}

    def raise_if_dirty(self, allow: bool = False) -> "IntegrityReport":
        if not self.violations:
            logger.info("integrity: %d checks passed", len(self.checks))
            return self
        msg = "evaluation integrity violated:\n  - " + "\n  - ".join(self.violations)
        if allow:
            self.waived = True
            logger.error("%s\n(continuing: --allow-leakage; results are marked unclean)", msg)
            return self
        raise IntegrityError(
            msg + "\n\nFix the splits rather than passing --allow-leakage. "
                  "A leaked number cannot be defended in review."
        )


def _parent(pair_id: str) -> str:
    from icce.convert.georef import parse_crop_id
    return parse_crop_id(pair_id)[0]


def check_split_disjoint(
    train_ids: Sequence[str],
    eval_ids: Sequence[str],
    train_name: str = "train",
    eval_name: str = "eval",
    report: Optional[IntegrityReport] = None,
) -> IntegrityReport:
    report = report or IntegrityReport()

    shared: Set[str] = set(train_ids) & set(eval_ids)
    if shared:
        sample = sorted(shared)[:5]
        report.violations.append(
            f"{len(shared)} pair id(s) appear in both {train_name} and "
            f"{eval_name} (e.g. {sample})"
        )
    else:
        report.checks.append(f"{train_name}/{eval_name} pair ids disjoint")

    tp = {_parent(i) for i in train_ids}
    ep = {_parent(i) for i in eval_ids}
    shared_scenes = tp & ep
    if shared_scenes:
        sample = sorted(shared_scenes)[:5]
        report.violations.append(
            f"{len(shared_scenes)} parent scene(s) straddle {train_name} and "
            f"{eval_name} (e.g. {sample}); crops of one tile are near-duplicates"
        )
    else:
        report.checks.append(f"{train_name}/{eval_name} parent scenes disjoint")
    return report


def check_threshold_provenance(
    extra: Dict,
    eval_split: str,
    report: Optional[IntegrityReport] = None,
) -> IntegrityReport:
    report = report or IntegrityReport()
    source = str(extra.get("threshold_split") or extra.get("val_split") or "").strip()

    if not source:
        report.checks.append(
            "threshold provenance unrecorded (checkpoint predates the field)")
        return report
    if source == eval_split:
        report.violations.append(
            f"thresholds were selected on '{source}', which is the split being "
            f"evaluated -- select them on validation"
        )
    else:
        report.checks.append(f"thresholds selected on '{source}', not '{eval_split}'")
    return report


def check_style_examples(
    split_used: Optional[str],
    report: Optional[IntegrityReport] = None,
) -> IntegrityReport:
    report = report or IntegrityReport()
    if split_used is None:
        report.checks.append("no style examples used")
    elif split_used != "train":
        report.violations.append(
            f"caption style examples drawn from '{split_used}'; they must come "
            f"from the training split"
        )
    else:
        report.checks.append("style examples drawn from train")
    return report


def check_evaluation(
    eval_ids: Sequence[str],
    eval_split: str,
    train_ids: Optional[Sequence[str]] = None,
    checkpoint_extra: Optional[Dict] = None,
    style_example_split: Optional[str] = None,
) -> IntegrityReport:
    """Run every applicable check for one evaluation run."""
    report = IntegrityReport()
    if train_ids is not None:
        check_split_disjoint(train_ids, eval_ids, "train", eval_split, report)
    if checkpoint_extra is not None:
        check_threshold_provenance(checkpoint_extra, eval_split, report)
    check_style_examples(style_example_split, report)
    return report

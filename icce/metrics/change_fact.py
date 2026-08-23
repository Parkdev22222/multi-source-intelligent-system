"""
Change-Fact-Score (CFS): does the generated report state the *right things*?

BLEU/CIDEr reward surface overlap with reference sentences and are blind to the
failure mode that actually matters for a consumer change-monitoring service:
the report confidently asserting a change that never happened, or missing one
that did. CFS scores a report against ground-truth *facts* instead of strings.

A report is reduced to a set of atomic claims

    ChangeClaim(direction, object_class, count)
      direction    : appeared | disappeared | modified | none
      object_class : building | road | vegetation | water | parking | land | ...
      count        : integer if the text commits to one, else None

Ground truth comes from two independent sources that we can cross-check:
  * the reference captions (LEVIR-CC, 5 per pair) -> claim set by majority vote
  * the binary change mask (LEVIR-CD)             -> instance count / presence

Reported quantities
  CFS-P / CFS-R / CFS-F1  : claim-set precision / recall / F1
  HalRate                 : share of generated claims unsupported by GT (1 - CFS-P)
  ChgAcc                  : binary change / no-change decision accuracy
  CountMAE                : |claimed instance count - GT instance count|

Everything here is rule-based and deterministic: no LLM sits in the metric, so
the metric cannot be gamed by the same model that writes the reports.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# --------------------------------------------------------------------------
# lexicon (bilingual: LEVIR-CC references are English, MSIS reports Korean)
# --------------------------------------------------------------------------
CLASS_LEXICON: Dict[str, Tuple[str, ...]] = {
    "building": ("building", "buildings", "house", "houses", "villa", "villas",
                 "apartment", "apartments", "cottage", "cottages", "warehouse",
                 "warehouses", "factory", "structure", "structures",
                 "건물", "주택", "아파트", "빌라", "창고", "공장", "건축물"),
    "road": ("road", "roads", "street", "streets", "highway", "path", "driveway",
             "도로", "길", "고속도로"),
    "vegetation": ("tree", "trees", "forest", "woods", "grass", "vegetation",
                   "나무", "수목", "숲", "산림", "식생", "잔디"),
    "water": ("river", "lake", "pond", "water", "reservoir",
              "강", "호수", "저수지", "수역", "연못"),
    "parking": ("parking lot", "parking", "car park", "주차장"),
    "land": ("bare land", "bareland", "villa land", "empty land", "vacant land",
             "ground", "soil", "나대지", "공터", "빈터"),
    "vehicle": ("car", "cars", "vehicle", "vehicles", "truck", "차량", "자동차"),
    "facility": ("stadium", "playground", "sports field", "pool", "solar panel",
                 "운동장", "경기장", "체육시설", "태양광"),
}

APPEAR_CUES = ("appear", "appeared", "appears", "built", "build", "constructed",
               "construct", "added", "add", "new", "newly", "emerged", "erected",
               "sprang up", "sprung up", "installed",
               "신축", "새로", "신규", "건설", "출현", "들어섰", "생겼", "추가")

DISAPPEAR_CUES = ("disappear", "disappeared", "disappears", "removed", "remove",
                  "demolished", "demolish", "razed", "cleared", "gone",
                  "vanished", "destroyed", "torn down",
                  "철거", "소실", "사라", "제거", "멸실", "없어졌")

MODIFY_CUES = ("widened", "widen", "expanded", "expand", "extended", "extend",
               "replaced", "replace", "renovated", "rebuilt", "changed", "altered",
               "확장", "증축", "개축", "변경", "대체", "재건")

NOCHANGE_CUES = ("nothing has changed", "no change", "the scene is the same",
                 "almost the same", "remain the same", "remains the same",
                 "there is no difference", "unchanged", "identical",
                 "변화 없", "변화가 없", "동일", "차이 없", "변동 없")

_NUM_WORDS = {
    "no": 0, "zero": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "several": None, "some": None, "many": None, "few": None, "numerous": None,
    "하나": 1, "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
}

_SENT_SPLIT = re.compile(r"[.!?;\n·]|(?<=다)\s")


@dataclass(frozen=True)
class ChangeClaim:
    direction: str          # appeared | disappeared | modified | none
    object_class: str       # key of CLASS_LEXICON, or "scene" for no-change
    count: Optional[int] = None

    def key(self) -> Tuple[str, str]:
        """Identity used for set matching -- counts are scored separately so a
        right-kind/wrong-number claim is a *count* error, not a hallucination."""
        return (self.direction, self.object_class)

    def as_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def _find_classes(sentence: str) -> List[str]:
    hits: List[str] = []
    for cls, words in CLASS_LEXICON.items():
        for w in words:
            if w in sentence:
                hits.append(cls)
                break
    return hits


def _find_direction(sentence: str) -> Optional[str]:
    # order matters: an explicit no-change statement wins over cue words
    if any(c in sentence for c in NOCHANGE_CUES):
        return "none"
    has_app = any(c in sentence for c in APPEAR_CUES)
    has_dis = any(c in sentence for c in DISAPPEAR_CUES)
    has_mod = any(c in sentence for c in MODIFY_CUES)
    if has_app and not has_dis:
        return "appeared"
    if has_dis and not has_app:
        return "disappeared"
    if has_app and has_dis:
        return "modified"     # "X was removed and Y was built" -> replacement
    if has_mod:
        return "modified"
    return None


def _find_count(sentence: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,4})\s*(?:개|동|채|대|곳)?\b", sentence)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    for w, v in _NUM_WORDS.items():
        if v is not None and re.search(rf"\b{re.escape(w)}\b", sentence):
            return v
    return None


def extract_claims(text: str) -> Set[ChangeClaim]:
    """Rule-based claim extraction from a caption or a full report."""
    if not text or not text.strip():
        return set()

    low = text.lower()
    claims: Set[ChangeClaim] = set()

    for raw in _SENT_SPLIT.split(low):
        sent = raw.strip()
        if len(sent) < 3:
            continue
        direction = _find_direction(sent)
        if direction is None:
            continue
        if direction == "none":
            claims.add(ChangeClaim("none", "scene", None))
            continue
        count = _find_count(sent)
        classes = _find_classes(sent)
        if not classes:
            claims.add(ChangeClaim(direction, "unspecified", count))
        for cls in classes:
            claims.add(ChangeClaim(direction, cls, count))

    # An explicit change claim contradicts a no-change claim; keep the changes.
    if len(claims) > 1:
        claims = {c for c in claims if c.direction != "none"} or claims
    return claims


def gt_claims_from_captions(
    captions: Sequence[str],
    min_votes: int = 2,
) -> Set[ChangeClaim]:
    """Majority-vote claim set over the N human references.

    A claim must be made by at least `min_votes` annotators to count as ground
    truth; this filters out one annotator's idiosyncratic detail while keeping
    everything the crowd agrees on.
    """
    votes: Counter = Counter()
    counts: Dict[Tuple[str, str], List[int]] = {}
    for cap in captions:
        for c in extract_claims(cap):
            votes[c.key()] += 1
            if c.count is not None:
                counts.setdefault(c.key(), []).append(c.count)

    out: Set[ChangeClaim] = set()
    for key, v in votes.items():
        if v < min_votes:
            continue
        cs = counts.get(key)
        median = int(np.median(cs)) if cs else None
        out.add(ChangeClaim(key[0], key[1], median))
    if not out and votes:                       # fall back to any claim at all
        for key in votes:
            out.add(ChangeClaim(key[0], key[1], None))
    return out


def gt_facts_from_mask(mask: np.ndarray, min_area_px: int = 32) -> Dict[str, float]:
    """Presence / instance count / area fraction from a binary change mask."""
    from icce.convert.mask_to_instances import connected_components

    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    binm = m > 127 if m.max() > 1 else m.astype(bool)
    comps = connected_components(binm, min_area=min_area_px)
    return {
        "change_present": bool(len(comps) > 0),
        "n_instances": len(comps),
        "area_ratio": float(binm.mean()),
    }


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
@dataclass
class FactScores:
    cfs_precision: float
    cfs_recall: float
    cfs_f1: float
    hallucination_rate: float
    change_accuracy: Optional[float]
    count_mae: Optional[float]
    n_samples: int
    n_pred_claims: int
    n_gt_claims: int
    per_direction: Dict[str, Dict[str, float]]

    def as_dict(self) -> Dict:
        return asdict(self)

    def as_row(self, name: str) -> str:
        s = (f"{name}  CFS-P={self.cfs_precision*100:.2f}  "
             f"CFS-R={self.cfs_recall*100:.2f}  CFS-F1={self.cfs_f1*100:.2f}  "
             f"Hal={self.hallucination_rate*100:.2f}")
        if self.change_accuracy is not None:
            s += f"  ChgAcc={self.change_accuracy*100:.2f}"
        if self.count_mae is not None:
            s += f"  CountMAE={self.count_mae:.2f}"
        return s


class ChangeFactEvaluator:
    """Accumulates CFS over a test set."""

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = 0
        self.n_samples = 0
        self._chg_hits = 0
        self._chg_total = 0
        self._count_errs: List[float] = []
        self._dir: Dict[str, List[int]] = {}

    def update(
        self,
        report: str,
        gt_claims: Set[ChangeClaim],
        gt_change_present: Optional[bool] = None,
        gt_instance_count: Optional[int] = None,
    ) -> Dict:
        pred = extract_claims(report)
        pred_keys = {c.key() for c in pred}
        gt_keys = {c.key() for c in gt_claims}

        hit = pred_keys & gt_keys
        self.tp += len(hit)
        self.fp += len(pred_keys - gt_keys)
        self.fn += len(gt_keys - pred_keys)
        self.n_samples += 1

        for d, _ in gt_keys:
            st = self._dir.setdefault(d, [0, 0])
            st[1] += 1
        for d, cls in hit:
            self._dir[d][0] += 1

        if gt_change_present is not None:
            pred_change = any(c.direction != "none" for c in pred)
            self._chg_total += 1
            self._chg_hits += int(pred_change == gt_change_present)

        if gt_instance_count is not None:
            claimed = [c.count for c in pred if c.count is not None]
            if claimed:
                self._count_errs.append(abs(max(claimed) - gt_instance_count))

        return {
            "pred_claims": sorted(pred_keys),
            "gt_claims": sorted(gt_keys),
            "matched": sorted(hit),
        }

    def compute(self) -> FactScores:
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return FactScores(
            cfs_precision=p,
            cfs_recall=r,
            cfs_f1=f1,
            hallucination_rate=1.0 - p,
            change_accuracy=(self._chg_hits / self._chg_total) if self._chg_total else None,
            count_mae=float(np.mean(self._count_errs)) if self._count_errs else None,
            n_samples=self.n_samples,
            n_pred_claims=self.tp + self.fp,
            n_gt_claims=self.tp + self.fn,
            per_direction={
                k: {"recall": (v[0] / v[1]) if v[1] else 0.0, "support": v[1]}
                for k, v in sorted(self._dir.items())
            },
        )

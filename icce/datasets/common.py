"""
Common types and helpers for bi-temporal change-detection benchmarks.

Every loader yields `ChangePair` records so that downstream code (MSIS adapter,
metrics, report generation) is dataset-agnostic.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Root that holds every downloaded benchmark. Override with MSIS_DATA_ROOT.
DATA_ROOT = Path(os.getenv("MSIS_DATA_ROOT", "data/benchmarks")).expanduser()

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass
class ChangePair:
    """One bi-temporal sample from a public benchmark."""

    pair_id: str
    dataset: str
    split: str
    image_a: Path                      # t0 ("before")
    image_b: Path                      # t1 ("after")
    mask: Optional[Path] = None        # binary change GT (255 = changed)
    captions: List[str] = field(default_factory=list)   # RSICC references
    change_flag: Optional[bool] = None                  # LEVIR-CC nochange flag
    meta: Dict = field(default_factory=dict)

    def exists(self) -> bool:
        return self.image_a.is_file() and self.image_b.is_file()


class DatasetNotFound(FileNotFoundError):
    """Raised with actionable download instructions."""

    def __init__(self, name: str, root: Path, how: str):
        super().__init__(
            f"[{name}] not found under {root}\n"
            f"Expected layout / download:\n{how}\n"
            f"Set MSIS_DATA_ROOT to change the search root "
            f"(current: {DATA_ROOT.resolve()})"
        )


def _stem_index(directory: Path) -> Dict[str, Path]:
    """Map file stem -> path for every image in `directory` (non-recursive)."""
    out: Dict[str, Path] = {}
    if not directory.is_dir():
        return out
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out


def find_dir(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    """First existing directory among `candidates` (case-insensitive)."""
    if not root.is_dir():
        return None
    lower = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    for cand in candidates:
        hit = lower.get(cand.lower())
        if hit is not None:
            return hit
    return None


def pair_up(
    dir_a: Path,
    dir_b: Path,
    dir_label: Optional[Path],
    dataset: str,
    split: str,
) -> List[ChangePair]:
    """Match A/B/label by filename stem."""
    idx_a = _stem_index(dir_a)
    idx_b = _stem_index(dir_b)
    idx_l = _stem_index(dir_label) if dir_label else {}

    common = sorted(set(idx_a) & set(idx_b))
    missing_b = len(idx_a) - len(common)
    if missing_b:
        logger.warning("%s/%s: %d A-images had no B counterpart", dataset, split, missing_b)

    pairs = []
    for stem in common:
        pairs.append(
            ChangePair(
                pair_id=stem,
                dataset=dataset,
                split=split,
                image_a=idx_a[stem],
                image_b=idx_b[stem],
                mask=idx_l.get(stem),
            )
        )
    return pairs


def subsample(pairs: List[ChangePair], limit: Optional[int], seed: int = 0) -> List[ChangePair]:
    """Deterministic subsample used for budget-limited ablations.

    Uses a stride so the subset stays spread across the (sorted) test set
    instead of clustering in one geographic region.
    """
    if limit is None or limit <= 0 or limit >= len(pairs):
        return pairs
    stride = len(pairs) / float(limit)
    picked = [pairs[min(int(i * stride), len(pairs) - 1)] for i in range(limit)]
    # stride collisions are possible on tiny sets; de-duplicate while ordered
    seen, out = set(), []
    for p in picked:
        if p.pair_id not in seen:
            seen.add(p.pair_id)
            out.append(p)
    return out


def dump_manifest(pairs: Iterable[ChangePair], path: Path) -> Path:
    """Write a JSON manifest so RunPod runs are reproducible/inspectable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "pair_id": p.pair_id,
            "dataset": p.dataset,
            "split": p.split,
            "image_a": str(p.image_a),
            "image_b": str(p.image_b),
            "mask": str(p.mask) if p.mask else None,
            "n_captions": len(p.captions),
            "change_flag": p.change_flag,
        }
        for p in pairs
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

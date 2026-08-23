"""
LEVIR-CD building change detection benchmark.

Chen & Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for
Remote Sensing Image Change Detection", Remote Sensing 2020.

637 bi-temporal 1024x1024 RGB patches at 0.5 m GSD, split 445/64/128
(train/val/test). Labels are binary building-change masks.

Expected layout (also accepts LEVIR-CD256 pre-cropped variants):

    $MSIS_DATA_ROOT/LEVIR-CD/
        train/{A,B,label}/*.png
        val/{A,B,label}/*.png
        test/{A,B,label}/*.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .common import DATA_ROOT, ChangePair, DatasetNotFound, find_dir, pair_up, subsample

NAME = "LEVIR-CD"
GSD_M = 0.5
TILE_PX = 1024

_HOWTO = """\
  huggingface-cli download --repo-type dataset ericyu/LEVIR-CD \\
      --local-dir $MSIS_DATA_ROOT/LEVIR-CD
  (or the official release: https://chenhao.in/LEVIR/)
  Resulting tree must contain <split>/A, <split>/B, <split>/label
"""

_DIR_ALIASES = {
    "A": ("A", "im1", "image1", "before", "T1", "t1"),
    "B": ("B", "im2", "image2", "after", "T2", "t2"),
    "label": ("label", "labels", "OUT", "gt", "mask", "label1"),
}


def root(custom: Optional[Path] = None) -> Path:
    if custom:
        return Path(custom)
    for cand in ("LEVIR-CD", "LEVIR_CD", "levir-cd", "LEVIR-CD256", "LEVIR-CD-256"):
        p = DATA_ROOT / cand
        if p.is_dir():
            return p
    return DATA_ROOT / "LEVIR-CD"


def load(
    split: str = "test",
    limit: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> List[ChangePair]:
    base = root(data_root)
    split_dir = find_dir(base, (split, split.upper(), split.capitalize()))
    if split_dir is None:
        # Some mirrors flatten the split into the directory name (LEVIR-CD/test/...)
        split_dir = base / split
    if not split_dir.is_dir():
        raise DatasetNotFound(f"{NAME}:{split}", split_dir, _HOWTO)

    dir_a = find_dir(split_dir, _DIR_ALIASES["A"])
    dir_b = find_dir(split_dir, _DIR_ALIASES["B"])
    dir_l = find_dir(split_dir, _DIR_ALIASES["label"])
    if dir_a is None or dir_b is None:
        raise DatasetNotFound(f"{NAME}:{split}", split_dir, _HOWTO)

    pairs = pair_up(dir_a, dir_b, dir_l, NAME, split)
    for p in pairs:
        p.meta.update({"gsd_m": GSD_M, "tile_px": TILE_PX})
    if not pairs:
        raise DatasetNotFound(f"{NAME}:{split} (empty)", split_dir, _HOWTO)
    return subsample(pairs, limit)

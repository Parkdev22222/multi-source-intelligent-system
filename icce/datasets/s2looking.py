"""
S2Looking: side-looking satellite building change detection (Shen et al., 2021).

5,000 bi-temporal 1024x1024 pairs with large view-angle differences — the
hardest public building-CD benchmark and a good stress test for the geometric
robustness of instance pairing.

Expected layout:
    $MSIS_DATA_ROOT/S2Looking/{train,val,test}/{Image1,Image2,label}/*.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .common import DATA_ROOT, ChangePair, DatasetNotFound, find_dir, pair_up, subsample

NAME = "S2Looking"
GSD_M = 0.5

_HOWTO = """\
  https://github.com/S2Looking/Dataset
  Extract to $MSIS_DATA_ROOT/S2Looking/<split>/{Image1,Image2,label}/
"""


def root(custom: Optional[Path] = None) -> Path:
    if custom:
        return Path(custom)
    for cand in ("S2Looking", "s2looking", "S2LOOKING"):
        p = DATA_ROOT / cand
        if p.is_dir():
            return p
    return DATA_ROOT / "S2Looking"


def load(
    split: str = "test",
    limit: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> List[ChangePair]:
    base = root(data_root)
    split_dir = find_dir(base, (split,)) or (base / split)
    if not split_dir.is_dir():
        raise DatasetNotFound(f"{NAME}:{split}", base, _HOWTO)

    dir_a = find_dir(split_dir, ("Image1", "A", "im1"))
    dir_b = find_dir(split_dir, ("Image2", "B", "im2"))
    dir_l = find_dir(split_dir, ("label", "gt", "mask"))
    if dir_a is None or dir_b is None:
        raise DatasetNotFound(f"{NAME}:{split}", split_dir, _HOWTO)

    pairs = pair_up(dir_a, dir_b, dir_l, NAME, split)
    for p in pairs:
        p.meta["gsd_m"] = GSD_M
    if not pairs:
        raise DatasetNotFound(f"{NAME}:{split} (empty)", split_dir, _HOWTO)
    return subsample(pairs, limit)

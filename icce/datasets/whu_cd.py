"""
WHU Building Change Detection dataset (Ji et al., IEEE TGRS 2019).

Aerial imagery of Christchurch, NZ before/after the 2011 earthquake rebuild.
Used here as an *unseen* domain: the pairing head is trained only on LEVIR-CD,
so WHU-CD measures zero-shot transfer of the open-vocabulary pipeline.

Expected layout (either the pre-cropped 256px release or the raw two-tile one):

    $MSIS_DATA_ROOT/WHU-CD/{train,val,test}/{A,B,label}/*.png
    $MSIS_DATA_ROOT/WHU-CD/{before,after,change_label}.tif      # raw variant
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .common import DATA_ROOT, ChangePair, DatasetNotFound, find_dir, pair_up, subsample

NAME = "WHU-CD"
GSD_M = 0.075

_HOWTO = """\
  http://gpcv.whu.edu.cn/data/building_dataset.html  (Building change detection dataset)
  Either extract the cropped release to
    $MSIS_DATA_ROOT/WHU-CD/<split>/{A,B,label}/
  or place the raw tiles as
    $MSIS_DATA_ROOT/WHU-CD/{before,after,change_label}.tif
  and run: python -m icce.convert.crop_raw --dataset whu_cd --size 256
"""


def root(custom: Optional[Path] = None) -> Path:
    if custom:
        return Path(custom)
    for cand in ("WHU-CD", "WHU_CD", "whu-cd", "WHU-BCD"):
        p = DATA_ROOT / cand
        if p.is_dir():
            return p
    return DATA_ROOT / "WHU-CD"


def load(
    split: str = "test",
    limit: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> List[ChangePair]:
    base = root(data_root)
    split_dir = find_dir(base, (split,)) or (base / split)
    if not split_dir.is_dir():
        raise DatasetNotFound(f"{NAME}:{split}", base, _HOWTO)

    dir_a = find_dir(split_dir, ("A", "im1", "before", "T1"))
    dir_b = find_dir(split_dir, ("B", "im2", "after", "T2"))
    dir_l = find_dir(split_dir, ("label", "OUT", "gt", "change_label", "mask"))
    if dir_a is None or dir_b is None:
        raise DatasetNotFound(f"{NAME}:{split}", split_dir, _HOWTO)

    pairs = pair_up(dir_a, dir_b, dir_l, NAME, split)
    for p in pairs:
        p.meta["gsd_m"] = GSD_M
    if not pairs:
        raise DatasetNotFound(f"{NAME}:{split} (empty)", split_dir, _HOWTO)
    return subsample(pairs, limit)

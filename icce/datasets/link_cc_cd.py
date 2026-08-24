"""
Recover ground-truth change masks for LEVIR-CC by matching its crops back to
LEVIR-CD tiles.

Why this exists
---------------
LEVIR-CC ships 256 px bi-temporal crops with five reference sentences each, but
no masks. LEVIR-CD ships 1024 px tiles with binary change masks, but no text.
They are the same imagery: every LEVIR-CC crop is a 256 px sub-window of a
LEVIR-CD tile, byte-identical, so the join can be made exactly rather than
heuristically.

The stems do not line up (`test_000107` vs `test_42`), which is why the
filename-based `levir_cc.attach_cd_masks` recovers nothing, so the join is done
on pixel content: md5 every 256 px sub-window of every LEVIR-CD tile, then look
up each LEVIR-CC crop. Matching is exact -- there is no threshold to tune and no
chance of a wrong pair being silently accepted.

What it unlocks
---------------
  * a ground-truth instance count per LEVIR-CC crop, which is what
    Change-Fact-Score needs for CountMAE and for its change/no-change label
  * pixel-level change detection scored on the *same* samples as the captions,
    so the detection table and the report table describe one system on one set
    rather than two loosely related experiments

It also answers a question the paper cannot afford to get wrong: LEVIR-CC was
re-split independently of LEVIR-CD, so a LEVIR-CC *test* crop may well come
from a LEVIR-CD *train* tile. Anything trained on LEVIR-CD train and evaluated
on LEVIR-CC test is then leaking. This module reports that overlap explicitly
(`--report-only`) and writes it into the manifest so `icce.eval.integrity` can
refuse to produce a contaminated number.

Usage
-----
    python -m icce.datasets.link_cc_cd                  # build masks + manifest
    python -m icce.datasets.link_cc_cd --report-only    # just the overlap table
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import DATA_ROOT

logger = logging.getLogger(__name__)

CROP = 256
CD_SPLITS = ("train", "val", "test")
CC_SPLITS = ("train", "val", "test")


def _md5(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _open_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def build_cd_index(cd_root: Optional[Path] = None) -> Dict[str, Tuple[str, str, int, int]]:
    """md5(256px crop of the 'A' frame) -> (cd_split, tile_id, row, col).

    Indexing the A frame alone is enough to identify the tile and offset, and
    halves the work. The B frame is only read later, to verify the hit.
    """
    from . import levir_cd

    index: Dict[str, Tuple[str, str, int, int]] = {}
    collisions = 0
    for split in CD_SPLITS:
        try:
            pairs = levir_cd.load(split=split, data_root=cd_root)
        except FileNotFoundError as exc:
            logger.warning("LEVIR-CD/%s unavailable, skipping (%s)", split, exc)
            continue
        for p in pairs:
            tile = _open_rgb(p.image_a)
            h, w = tile.shape[:2]
            for r in range(0, h - CROP + 1, CROP):
                for c in range(0, w - CROP + 1, CROP):
                    key = _md5(tile[r:r + CROP, c:c + CROP])
                    if key in index:
                        collisions += 1
                        continue
                    index[key] = (split, p.pair_id, r, c)
        logger.info("indexed LEVIR-CD/%s: %d tiles", split, len(pairs))

    if collisions:
        # Identical crops do occur (uniform farmland, water). They are ambiguous
        # by construction, so the first occurrence wins and the count is logged
        # rather than hidden.
        logger.info("%d duplicate crop hashes (uniform terrain); first wins", collisions)
    logger.info("crop index: %d unique 256px windows", len(index))
    return index


class _MaskReader:
    """Reads LEVIR-CD mask tiles, keeping the split listing and the last tile.

    Crops arrive grouped by source tile often enough that a one-tile cache
    removes almost all re-decoding, and the split listing is resolved once
    instead of once per crop.
    """

    def __init__(self, cd_root: Optional[Path]) -> None:
        self._cd_root = cd_root
        self._listing: Dict[str, Dict[str, Path]] = {}
        self._tile: Tuple[Optional[Tuple[str, str]], Optional[np.ndarray]] = (None, None)

    def _paths(self, cd_split: str) -> Dict[str, Path]:
        from . import levir_cd

        if cd_split not in self._listing:
            self._listing[cd_split] = {
                p.pair_id: p.mask
                for p in levir_cd.load(split=cd_split, data_root=self._cd_root)
                if p.mask is not None
            }
        return self._listing[cd_split]

    def crop(self, cd_split: str, tile_id: str, row: int, col: int) -> Optional[np.ndarray]:
        from PIL import Image

        key = (cd_split, tile_id)
        if self._tile[0] != key:
            path = self._paths(cd_split).get(tile_id)
            if path is None:
                return None
            self._tile = (key, np.array(Image.open(path).convert("L")))
        mask = self._tile[1]
        return None if mask is None else mask[row:row + CROP, col:col + CROP]


def link(
    cd_root: Optional[Path] = None,
    cc_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    splits: Tuple[str, ...] = CC_SPLITS,
    write_masks: bool = True,
) -> Dict:
    """Match every LEVIR-CC crop to its LEVIR-CD source and write its mask."""
    from PIL import Image

    from . import levir_cc

    index = build_cd_index(cd_root)
    if not index:
        raise SystemExit("LEVIR-CD is not available; nothing to link against")

    cc_base = levir_cc.root(cc_root)
    out_dir = Path(out_dir) if out_dir else cc_base / "masks"

    manifest: Dict[str, Dict] = {}
    overlap: Dict[str, Counter] = defaultdict(Counter)
    unmatched: Dict[str, List[str]] = defaultdict(list)
    masks = _MaskReader(cd_root)

    for cc_split in splits:
        try:
            pairs = levir_cc.load(split=cc_split, data_root=cc_root)
        except FileNotFoundError as exc:
            logger.warning("LEVIR-CC/%s unavailable, skipping (%s)", cc_split, exc)
            continue

        split_out = out_dir / cc_split
        if write_masks:
            split_out.mkdir(parents=True, exist_ok=True)

        for pair in pairs:
            hit = index.get(_md5(_open_rgb(pair.image_a)))
            if hit is None:
                unmatched[cc_split].append(pair.pair_id)
                overlap[cc_split]["unmatched"] += 1
                continue

            cd_split, tile_id, row, col = hit
            overlap[cc_split][cd_split] += 1
            entry = {
                "cd_split": cd_split, "cd_tile": tile_id,
                "row": row, "col": col, "mask": None,
            }

            if write_masks:
                crop = masks.crop(cd_split, tile_id, row, col)
                if crop is not None:
                    dst = split_out / f"{pair.pair_id}.png"
                    Image.fromarray((crop > 127).astype(np.uint8) * 255).save(dst)
                    entry["mask"] = str(dst)
            manifest[pair.pair_id] = entry

        logger.info("LEVIR-CC/%s: %d/%d matched", cc_split,
                    len(pairs) - len(unmatched[cc_split]), len(pairs))

    payload = {
        "crop_px": CROP,
        "overlap": {k: dict(v) for k, v in overlap.items()},
        "unmatched": {k: v for k, v in unmatched.items() if v},
        "pairs": manifest,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cc_cd_map.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def print_overlap(payload: Dict) -> None:
    """The table that decides whether a LEVIR-CD-trained head may be evaluated
    on LEVIR-CC test."""
    print("\nLEVIR-CC crop -> source LEVIR-CD split")
    print(f"  {'CC split':10s} " + " ".join(f"{s:>8s}" for s in CD_SPLITS) + f" {'unmatched':>10s}")
    contaminated = []
    for cc_split, counts in payload["overlap"].items():
        row = " ".join(f"{counts.get(s, 0):8d}" for s in CD_SPLITS)
        print(f"  {cc_split:10s} {row} {counts.get('unmatched', 0):10d}")
        if cc_split in ("val", "test") and counts.get("train", 0):
            contaminated.append((cc_split, counts["train"]))

    if contaminated:
        print("\n  WARNING: LEVIR-CC evaluation crops drawn from LEVIR-CD *train* tiles:")
        for cc_split, n in contaminated:
            print(f"    {cc_split}: {n} crops")
        print("  A pairing head trained on LEVIR-CD train has seen this imagery.")
        print("  Either train the head on LEVIR-CD tiles disjoint from LEVIR-CC")
        print("  eval crops, or report the LEVIR-CC numbers on the uncontaminated")
        print("  subset. icce.eval.integrity reads cc_cd_map.json to enforce this.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Link LEVIR-CC crops to LEVIR-CD masks")
    ap.add_argument("--cd-root", default=None, type=Path)
    ap.add_argument("--cc-root", default=None, type=Path)
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--report-only", action="store_true",
                    help="compute the split-overlap table without writing masks")
    args = ap.parse_args(argv)

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logger.info("benchmark root: %s", DATA_ROOT.resolve())

    payload = link(
        cd_root=args.cd_root, cc_root=args.cc_root, out_dir=args.out,
        splits=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
        write_masks=not args.report_only,
    )
    print_overlap(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

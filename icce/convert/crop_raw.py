"""
Materialise the WHU-CD official release into the tree `icce.datasets.whu_cd`
reads, cutting the one thing it ships whole: the change mask.

The release (`Building change detection dataset_add.zip`) is *almost* ready to
use and misleadingly so:

    1. The two-period image data/
        2012/splited_images/{train,test}/image/{0_i}.tif   512 px crops
        2016/splited_images/{train,test}/image/{0_i}.tif   512 px crops
        2012/splited_images/{train,test}/label/            per-crop BUILDING mask
        change_label/{train,test}/change_label.tif         whole-scene CHANGE mask

The per-crop `label/` directories are semantic building masks for one epoch,
not change. The change mask exists only as one image per split, so the crops
have images but no labels until it is cut on the same grid.

**The grid is recovered by measurement, not by assumption.** Crop names are a
flat running index (`0_0 .. 0_689`), which says nothing about position, and
guessing wrong would silently pair every image with the wrong label -- the
failure mode is a plausible-looking table computed on shuffled ground truth.
Both conventions are therefore tested against the whole-scene image for exact
pixel equality before anything is written, and the run aborts if neither holds:

    row-major   r = i // ncols, c = i % ncols        <- what WHU-CD uses
    column-major r = i % nrows, c = i // nrows

Edge crops are anchored back inside the scene rather than padded
(`y0 = min(r*S, H-S)`), which is why they overlap their neighbours and why a
naive `[r*S:(r+1)*S]` slice does not reproduce them. Verified the same way.

    python -m icce.convert.crop_raw --dataset whu_cd

Images are symlinked, not copied: the release is 3.9 GB and duplicating it buys
nothing. Only the generated label crops are written.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PIL_LIMIT_LIFTED = False

CROP_RE = re.compile(r"^\d+_(\d+)$")

# Relative to "1. The two-period image data".
WHU_LAYOUT = {
    "a_crops": "2012/splited_images/{split}/image",
    "b_crops": "2016/splited_images/{split}/image",
    "a_whole": "2012/whole_image/{split}/image/2012_{split}.tif",
    "change": "change_label/{split}/change_label.tif",
}
WHU_SPLITS = ("train", "test")


def _lift_pil_limit() -> None:
    """Aerial orthophotos are far past PIL's decompression-bomb ceiling."""
    global _PIL_LIMIT_LIFTED
    if not _PIL_LIMIT_LIFTED:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        _PIL_LIMIT_LIFTED = True


def open_array(path: Path) -> np.ndarray:
    _lift_pil_limit()
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im)


def find_release(base: Path) -> Path:
    """The '1. The two-period image data' directory, wherever it was unzipped."""
    if (base / "change_label").is_dir():
        return base
    hits = sorted(base.rglob("change_label"))
    for h in hits:
        if h.is_dir() and (h.parent / "2012").is_dir():
            return h.parent
    raise FileNotFoundError(
        f"WHU-CD release not found under {base}. Expected a directory containing "
        f"'change_label/' and '2012/'; unzip 'Building change detection "
        f"dataset_add.zip' there first."
    )


def crop_window(index: int, grid: Tuple[int, int], scene: Tuple[int, int],
                size: int, row_major: bool = True) -> Tuple[int, int]:
    """Top-left (y, x) of crop `index`, anchored inside the scene."""
    nrows, ncols = grid
    height, width = scene
    r, c = (index // ncols, index % ncols) if row_major else (index % nrows, index // nrows)
    return min(r * size, height - size), min(c * size, width - size)


def detect_ordering(
    crop_dir: Path, whole: np.ndarray, size: int, n_crops: int, probes: int = 6,
) -> Tuple[bool, Tuple[int, int]]:
    """Return (row_major, grid), verified by exact pixel equality.

    Raises if neither convention reproduces the sampled crops, which is the
    only safe outcome: a wrong grid pairs every image with someone else's mask.
    """
    height, width = whole.shape[:2]
    nrows, ncols = -(-height // size), -(-width // size)
    if nrows * ncols != n_crops:
        logger.warning("grid %dx%d = %d does not match %d crops on disk",
                       nrows, ncols, nrows * ncols, n_crops)

    names = sorted(crop_dir.iterdir())
    idx_of = {}
    for p in names:
        m = CROP_RE.match(p.stem)
        if m:
            idx_of[int(m.group(1))] = p
    if not idx_of:
        raise ValueError(f"no crops named like '0_123' in {crop_dir}")

    # Probe indices that separate the two conventions and include both edges.
    candidates = [0, 1, ncols - 1, ncols, ncols + 1, max(idx_of)]
    probe_idx = [i for i in dict.fromkeys(candidates) if i in idx_of][:probes]

    for row_major in (True, False):
        ok = True
        for i in probe_idx:
            arr = open_array(idx_of[i])
            y, x = crop_window(i, (nrows, ncols), (height, width), size, row_major)
            ref = whole[y:y + size, x:x + size]
            if ref.shape != arr.shape or not np.array_equal(ref, arr):
                ok = False
                break
        if ok:
            logger.info("crop ordering: %s, grid %dx%d, verified on %d crops",
                        "row-major" if row_major else "column-major",
                        nrows, ncols, len(probe_idx))
            return row_major, (nrows, ncols)

    raise ValueError(
        f"neither row-major nor column-major ordering reproduces the crops in "
        f"{crop_dir} against the whole scene. Refusing to guess: a wrong grid "
        f"silently pairs each image with another crop's change mask."
    )


def build_split(
    release: Path, split: str, out_root: Path, size: int, link_images: bool = True,
) -> Dict:
    from PIL import Image

    a_dir = release / WHU_LAYOUT["a_crops"].format(split=split)
    b_dir = release / WHU_LAYOUT["b_crops"].format(split=split)
    whole_p = release / WHU_LAYOUT["a_whole"].format(split=split)
    change_p = release / WHU_LAYOUT["change"].format(split=split)
    for p in (a_dir, b_dir, whole_p, change_p):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    whole = open_array(whole_p)
    change = open_array(change_p)
    if change.ndim == 3:
        change = change[..., 0]
    if change.shape[:2] != whole.shape[:2]:
        raise ValueError(
            f"{split}: change mask {change.shape[:2]} does not cover the scene "
            f"{whole.shape[:2]}; they must be the same raster."
        )
    change = (change > 0).astype(np.uint8) * 255

    crops = {int(CROP_RE.match(p.stem).group(1)): p
             for p in a_dir.iterdir() if CROP_RE.match(p.stem)}
    row_major, grid = detect_ordering(a_dir, whole, size, len(crops))
    del whole  # the scene is ~0.5 GB; the mask is all that is still needed

    split_out = out_root / split
    lab_out = split_out / "label"
    lab_out.mkdir(parents=True, exist_ok=True)

    for role, src in (("A", a_dir), ("B", b_dir)):
        dst = split_out / role
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink():
                dst.unlink()
            else:
                raise FileExistsError(f"{dst} exists and is not a symlink; move it aside")
        if link_images:
            dst.symlink_to(src.resolve(), target_is_directory=True)
        else:
            import shutil
            shutil.copytree(src, dst)

    height, width = change.shape[:2]
    n_changed = 0
    for i, src in sorted(crops.items()):
        y, x = crop_window(i, grid, (height, width), size, row_major)
        tile = change[y:y + size, x:x + size]
        Image.fromarray(tile).save(lab_out / f"{src.stem}.png")
        n_changed += int(bool(tile.any()))

    return {
        "split": split,
        "n_crops": len(crops),
        "n_crops_with_change": n_changed,
        "scene_px": [int(width), int(height)],
        "grid_rows_cols": list(grid),
        "crop_px": size,
        "ordering": "row-major" if row_major else "column-major",
        "edge_policy": "anchored inside the scene (min(r*S, H-S)); edge crops overlap",
        "images": "symlinked" if link_images else "copied",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Cut WHU-CD's whole-scene change mask into per-crop labels")
    ap.add_argument("--dataset", default="whu_cd", choices=("whu_cd",))
    ap.add_argument("--raw-root", type=Path, default=None,
                    help="where the release was unzipped (default: the dataset root)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output root (default: the dataset root)")
    ap.add_argument("--size", type=int, default=512,
                    help="crop size of the shipped images; 512 for the official release")
    ap.add_argument("--splits", nargs="*", default=list(WHU_SPLITS))
    ap.add_argument("--copy-images", action="store_true",
                    help="copy instead of symlink (3.9 GB duplicated)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from icce.datasets.whu_cd import root as ds_root
    base = args.raw_root or ds_root()
    out_root = args.out or base
    release = find_release(base)
    logger.info("release: %s", release)

    manifest = {"release": str(release), "splits": []}
    for split in args.splits:
        info = build_split(release, split, out_root, args.size,
                           link_images=not args.copy_images)
        manifest["splits"].append(info)
        logger.info("%-5s %d crops, %d with change (%.1f%%), grid %dx%d",
                    split, info["n_crops"], info["n_crops_with_change"],
                    100.0 * info["n_crops_with_change"] / max(1, info["n_crops"]),
                    *info["grid_rows_cols"])

    (out_root / "crop_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_root / "crop_manifest.json")
    logger.info("E2 uses 'test' only: the pairing head trains on LEVIR-CD, "
                "so WHU-CD is never trained on here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

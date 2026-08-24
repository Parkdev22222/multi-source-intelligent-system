"""
Fetch the public benchmarks and materialise them in the layout the loaders in
`icce/datasets/` expect.

The loaders read directory trees of PNGs; the practical mirrors on the Hub ship
parquet shards or a single zip. This module is the bridge, so that going from a
bare pod to a runnable benchmark is one command and not an afternoon of manual
unzipping:

    python -m icce.datasets.fetch --dataset levir_cd
    python -m icce.datasets.fetch --dataset levir_cc
    python -m icce.datasets.fetch --dataset all

Everything is idempotent: a split that is already materialised is skipped, so a
pre-empted pod costs one split rather than the whole download. Nothing here is
imported by the production pipeline.

Sources
-------
levir_cd  EVER-Z/torchange_levircd   parquet, full-resolution 1024 px tiles,
                                     official 445/64/128 split. This is the
                                     split published numbers are reported on,
                                     so it is the one the paper must use.
levir_cd256
          ericyu/LEVIRCD_Cropped256  parquet, the 256 px crops. Only needed if
                                     you want CD masks aligned to LEVIR-CC
                                     crops; see --dataset levir_cd256.
levir_cc  lcybuaa/LEVIR-CC           one zip: 10,077 crops + 5 references each.

WHU-CD has no working Hub mirror at the time of writing (the one repo that
carries the name is empty), so it is fetched by hand from
http://gpcv.whu.edu.cn/data/building_dataset.html if it is needed at all.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .common import DATA_ROOT

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# repo coordinates
# --------------------------------------------------------------------------
LEVIR_CD_REPO = "EVER-Z/torchange_levircd"
LEVIR_CD256_REPO = "ericyu/LEVIRCD_Cropped256"
LEVIR_CC_REPO = "lcybuaa/LEVIR-CC"
LEVIR_CC_ZIP = "Levir-CC-dataset.zip"

# column name -> destination subdirectory, for the parquet mirrors
_CD_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("t1_image", "A"),
    ("t2_image", "B"),
    ("change_mask", "label"),
)

# The official LEVIR-CD split sizes. Materialisation is checked against these
# so a truncated download fails loudly here instead of silently shrinking a
# test set and producing a number that cannot be compared to anything.
LEVIR_CD_EXPECTED = {"train": 445, "val": 64, "test": 128}


def _require_hub():
    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment issue
        raise SystemExit(
            "huggingface_hub is required to fetch benchmarks:\n"
            "    pip install huggingface_hub pyarrow\n"
            f"(import failed: {exc})"
        ) from exc


def _require_parquet():
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pyarrow is required to read the parquet mirrors:\n"
            "    pip install pyarrow\n"
            f"(import failed: {exc})"
        ) from exc


# --------------------------------------------------------------------------
# LEVIR-CD / LEVIR-CD256  (parquet -> A/B/label PNG tree)
# --------------------------------------------------------------------------
def _split_is_complete(split_dir: Path, expected: Optional[int]) -> bool:
    """A split counts as done when A, B and label all hold `expected` files."""
    counts = []
    for sub in ("A", "B", "label"):
        d = split_dir / sub
        if not d.is_dir():
            return False
        counts.append(sum(1 for p in d.iterdir() if p.suffix.lower() == ".png"))
    if len(set(counts)) != 1 or counts[0] == 0:
        return False
    return expected is None or counts[0] == expected


def _parquet_files_for_split(repo: str, split: str) -> List[str]:
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo, repo_type="dataset")
    hits = sorted(
        f for f in files
        if f.endswith(".parquet") and Path(f).name.startswith(f"{split}-")
    )
    if not hits:
        raise SystemExit(
            f"no parquet shard for split '{split}' in {repo}; saw:\n  "
            + "\n  ".join(f for f in files if f.endswith(".parquet"))
        )
    return hits


def _write_cd_split(repo: str, split: str, out_dir: Path, expected: Optional[int]) -> int:
    """Materialise one split of a parquet CD mirror into A/B/label PNGs."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    if _split_is_complete(out_dir, expected):
        logger.info("%s/%s already materialised, skipping", out_dir.name, split)
        return expected or 0

    for _, sub in _CD_COLUMNS:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    written = 0
    for shard in _parquet_files_for_split(repo, split):
        logger.info("downloading %s :: %s", repo, shard)
        local = hf_hub_download(repo, shard, repo_type="dataset")
        pf = pq.ParquetFile(local)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            rows = table.to_pylist()
            for row in rows:
                name = row.get("image_name") or f"{split}_{written:05d}.png"
                name = Path(name).name
                if not name.lower().endswith(".png"):
                    name += ".png"
                for col, sub in _CD_COLUMNS:
                    cell = row.get(col)
                    if cell is None:
                        continue
                    payload = cell["bytes"] if isinstance(cell, dict) else cell
                    if not payload:
                        continue
                    (out_dir / sub / name).write_bytes(payload)
                written += 1
            del table, rows

    logger.info("%s/%s: wrote %d pairs", out_dir.name, split, written)
    if expected is not None and written != expected:
        raise SystemExit(
            f"{out_dir.name}/{split}: materialised {written} pairs but the "
            f"official split has {expected}. Refusing to continue -- a "
            f"truncated split silently invalidates every number computed on it."
        )
    return written


def fetch_levir_cd(
    root: Path,
    splits: Iterable[str] = ("train", "val", "test"),
    cropped_256: bool = False,
) -> Path:
    _require_hub()
    _require_parquet()

    repo = LEVIR_CD256_REPO if cropped_256 else LEVIR_CD_REPO
    out = root / ("LEVIR-CD256" if cropped_256 else "LEVIR-CD")
    expected = None if cropped_256 else LEVIR_CD_EXPECTED

    for split in splits:
        _write_cd_split(
            repo, split, out / split,
            None if expected is None else expected.get(split),
        )
    return out


# --------------------------------------------------------------------------
# LEVIR-CC  (zip -> images/<split>/{A,B} + LevirCCcaptions.json)
# --------------------------------------------------------------------------
def _find_captions_json(base: Path) -> Optional[Path]:
    hits = sorted(base.rglob("*aption*.json"))
    return hits[0] if hits else None


def _find_split_images(base: Path) -> Dict[str, Path]:
    """Locate the directory holding <split>/A and <split>/B, whatever the depth."""
    found: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for cand in base.rglob(split):
            if cand.is_dir() and (cand / "A").is_dir() and (cand / "B").is_dir():
                found[split] = cand
                break
    return found


def _link_or_move(src: Path, dst: Path) -> None:
    """Prefer a symlink (the zip is 2.7 GB; do not copy it twice)."""
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _normalise_levir_cc(base: Path) -> None:
    """Reshape whatever the zip produced into the layout levir_cc.load() reads.

    The mirror nests everything one level deeper than the loader expects, so
    without this step the loader finds the captions but no images and reports
    'no usable pairs' -- a failure that looks like a broken loader rather than
    a layout mismatch.
    """
    cap = _find_captions_json(base)
    if cap is None:
        raise SystemExit(f"no captions json found under {base}")
    canonical_cap = base / "LevirCCcaptions.json"
    if cap != canonical_cap:
        _link_or_move(cap, canonical_cap)

    splits = _find_split_images(base)
    missing = {"train", "val", "test"} - set(splits)
    if missing:
        raise SystemExit(
            f"LEVIR-CC: could not locate image dirs for split(s) {sorted(missing)} "
            f"under {base}"
        )
    for split, src in splits.items():
        dst = base / "images" / split
        if dst.resolve() == src.resolve():
            continue
        _link_or_move(src, dst)


def fetch_levir_cc(root: Path) -> Path:
    _require_hub()
    from huggingface_hub import hf_hub_download

    out = root / "LEVIR-CC"
    if (out / "LevirCCcaptions.json").exists() and (out / "images" / "test").exists():
        logger.info("LEVIR-CC already materialised, skipping")
        return out

    out.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s :: %s (~2.7 GB)", LEVIR_CC_REPO, LEVIR_CC_ZIP)
    local = hf_hub_download(LEVIR_CC_REPO, LEVIR_CC_ZIP, repo_type="dataset")

    marker = out / ".extracted"
    if not marker.exists():
        logger.info("extracting into %s", out)
        with zipfile.ZipFile(local) as zf:
            zf.extractall(out)
        marker.write_text("ok", encoding="utf-8")

    _normalise_levir_cc(out)
    return out


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def verify(root: Path) -> int:
    """Load every materialised dataset through its real loader and report."""
    from . import levir_cc, levir_cd

    problems = 0
    for split, expected in LEVIR_CD_EXPECTED.items():
        try:
            pairs = levir_cd.load(split=split, data_root=root / "LEVIR-CD")
            ok = len(pairs) == expected
            n_masks = sum(1 for p in pairs if p.mask is not None)
            print(f"  LEVIR-CD/{split:5s} {len(pairs):5d} pairs "
                  f"(expected {expected}), {n_masks} with masks "
                  f"{'OK' if ok else 'MISMATCH'}")
            problems += 0 if ok else 1
        except Exception as exc:
            print(f"  LEVIR-CD/{split:5s} FAILED: {exc}")
            problems += 1

    for split in ("train", "val", "test"):
        try:
            pairs = levir_cc.load(split=split, data_root=root / "LEVIR-CC")
            n_caps = sum(len(p.captions) for p in pairs)
            print(f"  LEVIR-CC/{split:5s} {len(pairs):5d} pairs, "
                  f"{n_caps / max(1, len(pairs)):.2f} captions/pair")
        except Exception as exc:
            print(f"  LEVIR-CC/{split:5s} FAILED: {exc}")
            problems += 1

    return problems


# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dataset", default="all",
                    choices=["all", "levir_cd", "levir_cd256", "levir_cc", "verify"])
    ap.add_argument("--root", default=None,
                    help=f"benchmark root (default: MSIS_DATA_ROOT, now {DATA_ROOT})")
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    root = Path(args.root).expanduser() if args.root else DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.dataset in ("all", "levir_cd"):
        fetch_levir_cd(root, splits)
    if args.dataset == "levir_cd256":
        fetch_levir_cd(root, splits, cropped_256=True)
    if args.dataset in ("all", "levir_cc"):
        fetch_levir_cc(root)

    print(f"\nverifying under {root.resolve()}")
    problems = verify(root)
    print("\nall good" if problems == 0 else f"\n{problems} problem(s) above")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

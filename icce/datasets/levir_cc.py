"""
LEVIR-CC remote-sensing image change captioning benchmark.

Liu et al., "Remote Sensing Image Change Captioning with Dual-Branch
Transformers: A New Method and a Large-Scale Dataset", IEEE TGRS 2022.

10,077 bi-temporal 256x256 pairs derived from LEVIR-CD, each annotated with
5 English reference sentences (including explicit "no change" descriptions).
This is the only public benchmark that lets us score generated change *text*
against human references, so it anchors contribution C3.

Expected layout:

    $MSIS_DATA_ROOT/LEVIR-CC/
        images/{train,val,test}/{A,B}/*.png
        LevirCCcaptions.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .common import DATA_ROOT, ChangePair, DatasetNotFound, find_dir, subsample

logger = logging.getLogger(__name__)

NAME = "LEVIR-CC"

_HOWTO = """\
  git clone https://github.com/Chen-Yang-Liu/RSICC   # dataset card + links
  Download 'Levir-CC-dataset' (images/) and LevirCCcaptions.json, then:
    $MSIS_DATA_ROOT/LEVIR-CC/images/{train,val,test}/{A,B}/*.png
    $MSIS_DATA_ROOT/LEVIR-CC/LevirCCcaptions.json
"""

_CAPTION_FILES = (
    "LevirCCcaptions.json",
    "LevirCC_captions.json",
    "captions.json",
)


def root(custom: Optional[Path] = None) -> Path:
    if custom:
        return Path(custom)
    for cand in ("LEVIR-CC", "LEVIR_CC", "Levir-CC-dataset", "levir-cc"):
        p = DATA_ROOT / cand
        if p.is_dir():
            return p
    return DATA_ROOT / "LEVIR-CC"


def _find_caption_json(base: Path) -> Optional[Path]:
    for name in _CAPTION_FILES:
        p = base / name
        if p.is_file():
            return p
    hits = sorted(base.rglob("*aption*.json"))
    return hits[0] if hits else None


def _sentences(entry: Dict) -> List[str]:
    """LEVIR-CC stores both `raw` strings and pre-tokenised sentences."""
    out = []
    for s in entry.get("sentences", []):
        raw = s.get("raw")
        if raw:
            out.append(raw.strip())
        elif s.get("tokens"):
            out.append(" ".join(s["tokens"]))
    return out


def load(
    split: str = "test",
    limit: Optional[int] = None,
    data_root: Optional[Path] = None,
    include_nochange: bool = True,
) -> List[ChangePair]:
    base = root(data_root)
    cap_json = _find_caption_json(base)
    if cap_json is None:
        raise DatasetNotFound(f"{NAME} (captions json)", base, _HOWTO)

    img_root = find_dir(base, ("images", "Levir-CC-dataset", "img")) or base
    payload = json.loads(cap_json.read_text(encoding="utf-8"))
    entries = payload["images"] if isinstance(payload, dict) else payload

    pairs: List[ChangePair] = []
    missing = 0
    for e in entries:
        e_split = (e.get("filepath") or e.get("split") or "").strip()
        if e_split != split:
            continue
        fname = e.get("filename")
        if not fname:
            continue

        split_dir = find_dir(img_root, (e_split,)) or (img_root / e_split)
        dir_a = find_dir(split_dir, ("A", "im1", "before")) or (split_dir / "A")
        dir_b = find_dir(split_dir, ("B", "im2", "after")) or (split_dir / "B")
        img_a, img_b = dir_a / fname, dir_b / fname
        if not (img_a.is_file() and img_b.is_file()):
            missing += 1
            continue

        flag = e.get("changeflag")
        change_flag = None if flag is None else bool(int(flag))
        if not include_nochange and change_flag is False:
            continue

        pairs.append(
            ChangePair(
                pair_id=Path(fname).stem,
                dataset=NAME,
                split=split,
                image_a=img_a,
                image_b=img_b,
                mask=None,
                captions=_sentences(e),
                change_flag=change_flag,
                meta={"imgid": e.get("imgid"), "filename": fname},
            )
        )

    if missing:
        logger.warning("%s/%s: %d caption entries had no image on disk", NAME, split, missing)
    if not pairs:
        raise DatasetNotFound(f"{NAME}:{split} (no usable pairs)", base, _HOWTO)
    return subsample(pairs, limit)


def _attach_from_manifest(
    pairs: List[ChangePair],
    data_root: Optional[Path] = None,
) -> int:
    """Attach masks recovered by `icce.datasets.link_cc_cd` (pixel-content join).

    The manifest records mask paths relative to the directory the linker ran
    in, so a recorded path that does not resolve falls back to the canonical
    location under the LEVIR-CC root.
    """
    base = root(data_root)
    manifest = base / "masks" / "cc_cd_map.json"
    if not manifest.is_file():
        return 0

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("%s: unreadable CC->CD manifest (%s)", NAME, exc)
        return 0
    entries = payload.get("pairs", {})
    # The manifest stores row/col as *pixel* offsets into the parent tile
    # (0, 256, 512, 768); downstream wants crop indices (0, 1, 2, 3).
    crop_px = int(payload.get("crop_px") or 1) or 1

    hits = 0
    for pair in pairs:
        entry = entries.get(pair.pair_id)
        if not entry:
            continue
        candidates = [Path(entry["mask"])] if entry.get("mask") else []
        candidates.append(base / "masks" / pair.split / f"{pair.pair_id}.png")
        mask = next((p for p in candidates if p.is_file()), None)
        if mask is None:
            continue
        pair.mask = mask
        pair.meta["cd_tile"] = entry.get("cd_tile")
        pair.meta["cd_split"] = entry.get("cd_split")
        # (row, col) of this crop inside its parent tile. Crop ids in the public
        # release carry no position, so without these every crop looks like its
        # own scene and the knowledge graph never accumulates a neighbourhood.
        row, col = entry.get("row"), entry.get("col")
        pair.meta["cd_row"] = None if row is None else int(row) // crop_px
        pair.meta["cd_col"] = None if col is None else int(col) // crop_px
        hits += 1

    if hits:
        logger.info("%s: %d/%d masks from the CC->CD manifest", NAME, hits, len(pairs))
    return hits


def attach_cd_masks(
    pairs: List[ChangePair],
    levir_cd_root: Optional[Path] = None,
) -> int:
    """Best-effort: attach LEVIR-CD binary masks to LEVIR-CC pairs.

    Two joins, in order of reliability:

    1. The manifest written by `icce.datasets.link_cc_cd`, which matches every
       crop to its parent tile on pixel content. This is the one that works.
    2. Filename stems, for mirrors that name crops after the LEVIR-CD tile.
       The public release does not (`test_000107` vs `test_42`), so this
       recovers nothing there and is kept only as a fallback.

    Pairs also receive `meta['cd_tile']`/`meta['cd_split']` from the manifest,
    which is what lets an integrity check spot a LEVIR-CC test crop cut from a
    LEVIR-CD *train* tile. Returns the number of pairs that received a mask.
    """
    from . import levir_cd as cd

    hits = _attach_from_manifest(pairs)
    if hits:
        return hits

    base = cd.root(levir_cd_root)
    index = {}
    for split in ("train", "val", "test"):
        d = find_dir(base / split, ("label", "labels", "gt", "mask")) if (base / split).is_dir() else None
        if d:
            for p in d.iterdir():
                if p.is_file():
                    index[p.stem] = p

    hits = 0
    for pair in pairs:
        if pair.pair_id in index:
            pair.mask = index[pair.pair_id]
            hits += 1
    return hits

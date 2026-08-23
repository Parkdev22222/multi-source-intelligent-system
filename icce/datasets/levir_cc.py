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


def attach_cd_masks(
    pairs: List[ChangePair],
    levir_cd_root: Optional[Path] = None,
) -> int:
    """Best-effort: attach LEVIR-CD binary masks to LEVIR-CC pairs.

    LEVIR-CC crops are named `<levir_cd_stem>_<row>_<col>.png` in most mirrors;
    when the stems line up we can score pixel-level CD *and* caption quality on
    the very same samples, which is what the factuality metric needs.
    Returns the number of pairs that received a mask.
    """
    from . import levir_cd as cd

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

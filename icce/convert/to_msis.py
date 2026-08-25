"""
Benchmark ChangePair -> MSIS pipeline input.

Emits a `metadata.json` in exactly the schema `src.detection.image_loader`
expects, so the unmodified production pipeline can ingest LEVIR-CD / LEVIR-CC /
WHU-CD / S2Looking tiles. The `t0` image of a pair becomes the *past*
observation and `t1` the *current* one, 30 days apart -- long enough that the
pipeline treats them as separate campaigns rather than one revisit.

The returned `BenchmarkScene` keeps the GeoGrid alongside the image ids, which
is what the eval runners use to project detections back into pixel space.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from icce.convert.georef import GeoGrid, assign_geo, parse_crop_id
from icce.datasets.common import ChangePair

logger = logging.getLogger(__name__)

# Anchor date for synthetic capture times. Fixed so runs are reproducible.
T0 = datetime(2026, 3, 1, 2, 30, 0, tzinfo=timezone.utc)
REVISIT_DAYS = 30


@dataclass
class BenchmarkScene:
    """One benchmark pair, materialised as two MSIS observations."""

    pair_id: str
    dataset: str
    grid: GeoGrid
    past_file: str          # path relative to the metadata.json directory
    current_file: str
    past_time: datetime
    current_time: datetime
    gsd_m: float = 0.5
    mask_path: Optional[Path] = None
    captions: List[str] = field(default_factory=list)
    change_flag: Optional[bool] = None
    parent_scene: str = ""

    def entries(self, platform: str) -> List[Dict]:
        common = {
            "source_type": "satellite",
            "resolution_m": self.gsd_m,
            "sensor_platform": platform,
            "region_name": self.parent_scene or self.pair_id,
            "benchmark_pair_id": self.pair_id,
            "benchmark_dataset": self.dataset,
            **self.grid.as_metadata(),
        }
        return [
            {**common, "image_file": self.past_file,
             "capture_time": self.past_time.isoformat(), "benchmark_phase": "past"},
            {**common, "image_file": self.current_file,
             "capture_time": self.current_time.isoformat(), "benchmark_phase": "current"},
        ]


def _image_size(path: Path) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size            # (width, height)


def build_scenes(
    pairs: Sequence[ChangePair],
    gsd_m: float,
    metadata_dir: Path,
    revisit_days: int = REVISIT_DAYS,
) -> List[BenchmarkScene]:
    """Assign geo-grids and synthetic capture times to every pair.

    Crops of one parent scene are ordered by (row, col) and given *increasing*
    capture times, so a sequential run over them accumulates a genuine
    observation history in the knowledge graph.
    """
    if not pairs:
        return []

    size = _image_size(pairs[0].image_a)
    crop_index = _crop_index(pairs)
    grids = assign_geo([p.pair_id for p in pairs], size, gsd_m, crop_index)

    def _pos(p: ChangePair) -> Tuple[str, int, int]:
        parent, row, col = crop_index.get(p.pair_id) or parse_crop_id(p.pair_id)
        return parent, row or 0, col or 0

    # order within a parent scene drives the synthetic acquisition sequence
    ordered = sorted(pairs, key=_pos)

    scenes: List[BenchmarkScene] = []
    for i, p in enumerate(ordered):
        parent = _pos(p)[0]
        past = T0 + timedelta(hours=i)
        current = past + timedelta(days=revisit_days)
        scenes.append(
            BenchmarkScene(
                pair_id=p.pair_id,
                dataset=p.dataset,
                grid=grids[p.pair_id],
                past_file=_relative(p.image_a, metadata_dir),
                current_file=_relative(p.image_b, metadata_dir),
                past_time=past,
                current_time=current,
                gsd_m=gsd_m,
                mask_path=p.mask,
                captions=list(p.captions),
                change_flag=p.change_flag,
                parent_scene=parent,
            )
        )
    return scenes


def _crop_index(
    pairs: Sequence[ChangePair],
) -> Dict[str, Tuple[str, Optional[int], Optional[int]]]:
    """`pair_id -> (parent_scene, row, col)` for pairs that carry a position.

    LEVIR-CC crop ids (`test_000042`) encode neither parent nor position, so
    `parse_crop_id` makes every crop its own scene and the per-scene knowledge
    graph is handed one observation and no history. `levir_cc.attach_cd_masks`
    recovers the real position from the CC->CD manifest and leaves it in
    `meta`; this reads it back out.
    """
    index: Dict[str, Tuple[str, Optional[int], Optional[int]]] = {}
    for p in pairs:
        parent = p.meta.get("cd_tile")
        if parent:
            index[p.pair_id] = (parent, p.meta.get("cd_row"), p.meta.get("cd_col"))
    return index


def _relative(path: Path, base: Path) -> str:
    """metadata.json stores paths relative to its own directory."""
    p, b = Path(path).resolve(), Path(base).resolve()
    try:
        return str(p.relative_to(b))
    except ValueError:
        return str(p)


def write_metadata(
    scenes: Sequence[BenchmarkScene],
    out_path: Path,
    platform: str = "BENCHMARK",
) -> Path:
    """Write the MSIS metadata index for these scenes."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entries: List[Dict] = []
    for s in scenes:
        entries.extend(s.entries(platform))
    out_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    logger.info("wrote %d metadata entries (%d scenes) -> %s",
                len(entries), len(scenes), out_path)
    return out_path


def write_per_scene_metadata(
    scenes: Sequence[BenchmarkScene],
    out_dir: Path,
    platform: str = "BENCHMARK",
) -> Dict[str, Path]:
    """One metadata.json per scene.

    The pipeline pairs a current frame against the most recent past frame in
    the DB; feeding it one pair at a time keeps each benchmark sample isolated,
    which is what the *detection* experiments need. The knowledge-graph
    experiments instead use `write_metadata` over a whole parent scene so that
    history can accumulate.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for s in scenes:
        path = out_dir / f"{s.pair_id}.json"
        path.write_text(json.dumps(s.entries(platform), indent=2), encoding="utf-8")
        out[s.pair_id] = path
    return out


def group_by_scene(scenes: Sequence[BenchmarkScene]) -> Dict[str, List[BenchmarkScene]]:
    """Bucket crops by parent scene, preserving acquisition order."""
    out: Dict[str, List[BenchmarkScene]] = {}
    for s in scenes:
        out.setdefault(s.parent_scene, []).append(s)
    for v in out.values():
        v.sort(key=lambda s: s.past_time)
    return out

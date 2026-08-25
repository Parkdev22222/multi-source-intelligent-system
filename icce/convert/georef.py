"""
Synthetic but *invertible* geo-referencing for benchmark tiles.

The MSIS pipeline is geographic end to end: detections carry lat/lon, temporal
pairing matches on geodesic distance, and the knowledge graph clusters assets
on a lat/lon grid. Public CD benchmarks ship plain PNGs with no georeference.

We therefore synthesise one, with two hard requirements:

  1. **Invertibility** -- a detection that comes back as lat/lon must map to the
     exact pixel it came from, or instance IoU against the GT mask is garbage.
  2. **Neighbourhood fidelity** -- LEVIR-CC crops are 256 px sub-tiles of a
     1024 px LEVIR-CD scene. Crops of the same parent must land next to each
     other so the spatio-temporal knowledge graph sees a genuine neighbourhood
     instead of unrelated tiles scattered over the globe. That neighbourhood is
     precisely what GraphRAG retrieval exploits in the report ablation.

Layout: each parent scene owns a disjoint cell of a global grid, spaced far
enough apart that `GRAPHRAG_CONTEXT_RADIUS` never bridges two scenes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Equator-ish reference used to convert metres to degrees.
_M_PER_DEG_LAT = 111_320.0

# Base of the synthetic grid: an empty stretch of ocean-free land is not
# required (the pipeline's land check is bypassed for benchmark runs), but a
# mid-latitude origin keeps the lon scaling well conditioned.
BASE_LAT = 36.0
BASE_LON = 127.0

# Distance between two parent scenes, in degrees. Must exceed
# GRAPHRAG_CONTEXT_RADIUS_DEG (default 0.05) by a wide margin.
SCENE_SPACING_DEG = 0.5

# LEVIR-CC crop names look like "test_000042_0_1"; group 1 is the parent scene.
_CROP_RE = re.compile(r"^(?P<parent>.+?)_(?P<row>\d+)_(?P<col>\d+)$")


def m_per_deg_lon(lat: float) -> float:
    import math
    return _M_PER_DEG_LAT * max(0.1, math.cos(math.radians(lat)))


@dataclass(frozen=True)
class GeoGrid:
    """Axis-aligned linear mapping between pixel and geographic coordinates."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    width_px: int
    height_px: int

    # -- pixel -> geo ------------------------------------------------------
    def pixel_to_lonlat(self, x: float, y: float) -> Tuple[float, float]:
        """Image row 0 is the *north* edge, hence the inverted latitude term."""
        lon = self.lon_min + (x / self.width_px) * (self.lon_max - self.lon_min)
        lat = self.lat_max - (y / self.height_px) * (self.lat_max - self.lat_min)
        return lat, lon

    def bbox_to_geo(self, bbox: Sequence[float]) -> Tuple[float, float, float, float]:
        """(x1,y1,x2,y2) px -> (lat_min, lon_min, lat_max, lon_max)."""
        x1, y1, x2, y2 = bbox
        lat_a, lon_a = self.pixel_to_lonlat(x1, y1)
        lat_b, lon_b = self.pixel_to_lonlat(x2, y2)
        return (min(lat_a, lat_b), min(lon_a, lon_b), max(lat_a, lat_b), max(lon_a, lon_b))

    def centre_of(self, bbox: Sequence[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return self.pixel_to_lonlat((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    # -- geo -> pixel ------------------------------------------------------
    def lonlat_to_pixel(self, lat: float, lon: float) -> Tuple[float, float]:
        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * self.width_px
        y = (self.lat_max - lat) / (self.lat_max - self.lat_min) * self.height_px
        return x, y

    def geo_bbox_to_pixel(
        self, lat_min: float, lon_min: float, lat_max: float, lon_max: float
    ) -> Tuple[float, float, float, float]:
        x1, y1 = self.lonlat_to_pixel(lat_max, lon_min)   # NW corner
        x2, y2 = self.lonlat_to_pixel(lat_min, lon_max)   # SE corner
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def as_metadata(self) -> Dict[str, float]:
        return {
            "lat_center": (self.lat_min + self.lat_max) / 2.0,
            "lon_center": (self.lon_min + self.lon_max) / 2.0,
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }


def parse_crop_id(pair_id: str) -> Tuple[str, Optional[int], Optional[int]]:
    """`test_000042_0_1` -> ('test_000042', 0, 1); otherwise (pair_id, None, None)."""
    m = _CROP_RE.match(pair_id)
    if not m:
        return pair_id, None, None
    return m.group("parent"), int(m.group("row")), int(m.group("col"))


def _scene_origin(index: int) -> Tuple[float, float]:
    """Place scene `index` on a square spiral-free grid (row-major, 64 wide)."""
    row, col = divmod(index, 64)
    return BASE_LAT + row * SCENE_SPACING_DEG, BASE_LON + col * SCENE_SPACING_DEG


def assign_geo(
    pair_ids: Sequence[str],
    image_size: Tuple[int, int],
    gsd_m: float,
    crop_index: Optional[Dict[str, Tuple[str, Optional[int], Optional[int]]]] = None,
) -> Dict[str, GeoGrid]:
    """Build a GeoGrid per pair, preserving parent-scene adjacency.

    `image_size` is (width_px, height_px) of the tile.

    `crop_index` overrides the id-parsed `(parent, row, col)` for datasets whose
    crop ids do not encode their position -- LEVIR-CC being the one that
    matters, where the position comes from the CC->CD manifest instead.
    """
    crop_index = crop_index or {}
    width_px, height_px = image_size
    span_lat = (height_px * gsd_m) / _M_PER_DEG_LAT

    scene_index: Dict[str, int] = {}
    grids: Dict[str, GeoGrid] = {}

    for pid in pair_ids:
        parent, row, col = crop_index.get(pid) or parse_crop_id(pid)
        if parent not in scene_index:
            scene_index[parent] = len(scene_index)
        base_lat, base_lon = _scene_origin(scene_index[parent])
        span_lon = (width_px * gsd_m) / m_per_deg_lon(base_lat)

        # Crops without an (row, col) suffix each get their own slot in the
        # scene, laid out left to right so they still form a neighbourhood.
        if row is None or col is None:
            slot = sum(1 for k in grids
                       if (crop_index.get(k) or parse_crop_id(k))[0] == parent)
            row, col = divmod(slot, 8)

        lat_max = base_lat - row * span_lat
        lat_min = lat_max - span_lat
        lon_min = base_lon + col * span_lon
        lon_max = lon_min + span_lon

        grids[pid] = GeoGrid(
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
            width_px=width_px, height_px=height_px,
        )
    return grids

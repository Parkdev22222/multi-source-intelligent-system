"""
Change evidence: the neutral intermediate between pairing output and prompts.

Every grounding condition in the ablation renders *the same* evidence object.
That is deliberate -- if the conditions built their facts differently, the
comparison would measure prompt plumbing rather than grounding, and any gain
attributed to the knowledge graph would be unfalsifiable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ObservedChange:
    status: str                 # new | disappeared | changed | matched | moved
    object_class: str
    confidence: float
    lat: float
    lon: float
    bbox_px: Tuple[float, float, float, float]
    score: float = 1.0

    def bearing(self, width: int, height: int) -> str:
        """Coarse cardinal position inside the tile, for readable prose."""
        cx = (self.bbox_px[0] + self.bbox_px[2]) / 2.0 / max(1, width)
        cy = (self.bbox_px[1] + self.bbox_px[3]) / 2.0 / max(1, height)
        ns = "north" if cy < 0.33 else ("south" if cy > 0.67 else "")
        ew = "west" if cx < 0.33 else ("east" if cx > 0.67 else "")
        if ns and ew:
            return f"{ns}-{ew}ern"
        if ns:
            return f"{ns}ern"
        if ew:
            return f"{ew}ern"
        return "central"


@dataclass
class ChangeEvidence:
    pair_id: str
    scene: str
    lat: float
    lon: float
    past_time: datetime
    current_time: datetime
    image_size: Tuple[int, int]
    changes: List[ObservedChange] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    # -- views -------------------------------------------------------------
    def by_status(self, *statuses: str) -> List[ObservedChange]:
        return [c for c in self.changes if c.status in statuses]

    @property
    def appeared(self) -> List[ObservedChange]:
        return self.by_status("new")

    @property
    def disappeared(self) -> List[ObservedChange]:
        return self.by_status("disappeared")

    @property
    def modified(self) -> List[ObservedChange]:
        return self.by_status("changed")

    @property
    def stable(self) -> List[ObservedChange]:
        return self.by_status("matched", "moved")

    @property
    def has_change(self) -> bool:
        return bool(self.appeared or self.disappeared or self.modified)

    def class_counts(self, objs: Sequence[ObservedChange]) -> str:
        c = Counter(o.object_class for o in objs)
        return ", ".join(f"{k} x{v}" for k, v in c.most_common()) or "none"

    def interval_days(self) -> int:
        return max(0, (self.current_time - self.past_time).days)

    def summary_line(self) -> str:
        """One-line digest, also used as the retrieval key for flat RAG."""
        return (f"{self.scene} ({self.lat:.4f},{self.lon:.4f}) "
                f"appeared[{self.class_counts(self.appeared)}] "
                f"disappeared[{self.class_counts(self.disappeared)}] "
                f"modified[{self.class_counts(self.modified)}] "
                f"stable={len(self.stable)}")


def from_pairing_result(
    result,
    pair_id: str,
    scene: str,
    lat: float,
    lon: float,
    past_time: datetime,
    current_time: datetime,
    image_size: Tuple[int, int],
    meta: Optional[Dict] = None,
) -> ChangeEvidence:
    """Adapt `icce.pairing_head.infer.PairingResult` into evidence."""
    changes = [
        ObservedChange(
            status=o.status, object_class=o.object_class, confidence=o.confidence,
            lat=o.lat, lon=o.lon, bbox_px=tuple(o.bbox_px), score=o.score,
        )
        for o in result.outcomes
    ]
    return ChangeEvidence(
        pair_id=pair_id, scene=scene, lat=lat, lon=lon,
        past_time=past_time, current_time=current_time,
        image_size=image_size, changes=changes, meta=meta or {},
    )

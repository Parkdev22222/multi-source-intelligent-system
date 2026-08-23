"""
Drive the existing MSIS GraphRAG stack from benchmark pairing results.

The knowledge graph, entity extractor, Louvain community detection and the
local/global retriever already exist in `src/graph/`. This module only adapts
benchmark output into the `PairingRecord` shape they consume, so the ablation
exercises the *production* GraphRAG implementation rather than a reimplementation
tuned for the paper.

Ordering matters: crops of one parent scene are indexed in acquisition order and
each crop's context is retrieved *before* that crop is indexed. Otherwise the
graph would already contain the answer to the question being asked, and the
GraphRAG row would be measuring leakage rather than retrieval.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from icce.report.evidence import ChangeEvidence

logger = logging.getLogger(__name__)


def evidence_to_pairing_records(ev: ChangeEvidence) -> List:
    """In-memory PairingRecord rows (never committed to the pipeline DB)."""
    from src.database.models import PairingRecord

    records = []
    for i, c in enumerate(ev.changes):
        bbox = {"x1": c.bbox_px[0], "y1": c.bbox_px[1],
                "x2": c.bbox_px[2], "y2": c.bbox_px[3]}
        is_past_only = c.status == "disappeared"
        records.append(PairingRecord(
            id=f"{ev.pair_id}:{i}",
            lat_center=ev.lat,
            lon_center=ev.lon,
            current_detection_id=None if is_past_only else f"{ev.pair_id}:cur:{i}",
            current_object_class=None if is_past_only else c.object_class,
            current_confidence=None if is_past_only else c.confidence,
            current_lat=None if is_past_only else c.lat,
            current_lon=None if is_past_only else c.lon,
            current_capture_time=None if is_past_only else ev.current_time.replace(tzinfo=None),
            current_bbox=None if is_past_only else bbox,
            past_detection_id=None if c.status == "new" else f"{ev.pair_id}:past:{i}",
            past_object_class=None if c.status == "new" else c.object_class,
            past_confidence=None if c.status == "new" else c.confidence,
            past_lat=None if c.status == "new" else c.lat,
            past_lon=None if c.status == "new" else c.lon,
            past_capture_time=None if c.status == "new" else ev.past_time.replace(tzinfo=None),
            past_bbox=None if c.status == "new" else bbox,
            status=c.status,
            source_type="satellite",
        ))
    return records


class GraphContextBuilder:
    """Retrieve-then-index driver over a scene's crop sequence."""

    def __init__(self, db_path: Optional[Path] = None, radius_deg: float = 0.05,
                 community_interval: int = 1) -> None:
        if db_path is None:
            db_path = Path(tempfile.mkdtemp()) / "graph_eval.db"
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
        self.radius_deg = radius_deg

        from src.graph.graph_indexer import GraphIndexer
        self._indexer = GraphIndexer(db_path=str(db_path))
        try:
            self._indexer._auto_community_every = community_interval
        except AttributeError:
            pass
        self.n_indexed = 0

    def context_then_index(self, ev: ChangeEvidence) -> str:
        """Context for `ev` from everything seen *before* it, then index it."""
        records = evidence_to_pairing_records(ev)
        context = ""
        if self.n_indexed > 0:
            try:
                context = self._indexer.get_historical_context(records, self.radius_deg)
            except Exception as exc:
                logger.warning("graph context retrieval failed for %s: %s", ev.pair_id, exc)
        try:
            self._indexer.index_pairings(records, session_id=ev.pair_id)
            self.n_indexed += 1
        except Exception as exc:
            logger.warning("graph indexing failed for %s: %s", ev.pair_id, exc)
        return context

    def clear(self) -> None:
        try:
            self._indexer.clear()
        except Exception:
            pass
        self.n_indexed = 0

    def stats(self) -> Dict:
        try:
            return dict(self._indexer.stats())
        except Exception:
            return {}


def build_contexts(
    evidences: Sequence[ChangeEvidence],
    db_path: Optional[Path] = None,
    radius_deg: float = 0.05,
    per_scene: bool = True,
) -> Dict[str, str]:
    """Graph context per pair id, honouring acquisition order within a scene.

    With `per_scene=True` each parent scene gets a fresh graph, which is the
    honest setting for a consumer service that monitors one neighbourhood: the
    history available is that neighbourhood's own, not every other city's.
    """
    from collections import OrderedDict

    by_scene: "OrderedDict[str, List[ChangeEvidence]]" = OrderedDict()
    for ev in evidences:
        by_scene.setdefault(ev.scene if per_scene else "__all__", []).append(ev)

    out: Dict[str, str] = {}
    for scene, evs in by_scene.items():
        builder = GraphContextBuilder(
            db_path=(Path(db_path).with_name(f"{Path(db_path).stem}_{scene}.db")
                     if db_path else None),
            radius_deg=radius_deg,
        )
        for ev in sorted(evs, key=lambda e: e.current_time):
            out[ev.pair_id] = builder.context_then_index(ev)
        logger.info("scene %s: indexed %d crops, graph=%s", scene, builder.n_indexed, builder.stats())
    return out

"""
Temporal object pairing – two selectable strategies.

Strategy A – "sam3_tracker" (default):
  1. Sam3Tracker runs on the current frame with past-frame bboxes as prompts
     → returns TrackedObject list (past_detection_id + updated bbox + score)
  2. pair_by_tracking() assigns status for every object:
     ┌──────────────────────┬───────────────────────────────────────────────────┐
     │ status               │ condition                                         │
     ├──────────────────────┼───────────────────────────────────────────────────┤
     │ "matched" / "moved"  │ Sam3Tracker found the past object in current frame │
     │ "new"                │ current detection has no matching tracked past obj │
     │ "disappeared"        │ past object was NOT returned by Sam3Tracker        │
     └──────────────────────┴───────────────────────────────────────────────────┘

Strategy B – "similarity":
  No video tracker session needed.
  pair_by_similarity() directly matches current DetectionResult objects to
  past DetectionRecord objects using a weighted score:
    score = (1 - geo_dist / COORDINATE_MATCH_RADIUS_DEG)
            + SIMILARITY_CLASS_BONUS  (if classes match)
  Greedy assignment (best-score pair first).  Status rules:
     ┌──────────────────────┬───────────────────────────────────────────────────┐
     │ "matched" / "moved"  │ current ↔ past pair found within radius           │
     │ "new"                │ current detection – no past match within radius    │
     │ "disappeared"        │ past detection – no current match within radius    │
     └──────────────────────┴───────────────────────────────────────────────────┘

Select strategy via env var:  TRACKING_MODE=sam3_tracker | similarity
"""

import logging
from datetime import datetime
from typing import List, Optional

from src.config import (
    COORDINATE_MATCH_RADIUS_DEG,
    MOVE_DISTANCE_THRESHOLD_DEG,
    SIMILARITY_CLASS_BONUS,
)
from src.database.models import DetectionRecord, PairingRecord
from src.database.sensor_db import get_most_recent_past_detections
from src.detection.sam2_detector import DetectionResult, TrackedObject

logger = logging.getLogger(__name__)

# Minimum bbox IoU to accept a TrackedObject ↔ DetectionResult match
IOU_MATCH_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _geo_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Simple Euclidean distance in degrees between two lat/lon points."""
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5


def _bbox_iou(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> float:
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


# ---------------------------------------------------------------------------
# Main pairing function
# ---------------------------------------------------------------------------

def pair_by_tracking(
    tracked_objects: List[TrackedObject],
    current_detections: List[DetectionResult],
    past_detections: List[DetectionRecord],
    current_capture_time: datetime,
    region_lat: float,
    region_lon: float,
    session_id: str,
    source_type: str = "satellite",
) -> List[PairingRecord]:
    """
    Build PairingRecord list using Sam3Tracker object IDs.

    Args:
        tracked_objects:      Output of SAM3Detector.track_objects() –
                              past objects confirmed present in current frame.
        current_detections:   Sam3Model detections for the current frame
                              (from sensor DB, already stored).
        past_detections:      DetectionRecord list for the most-recent past frame.
        current_capture_time: Capture timestamp of the current image.
        region_lat/lon:       Geographic centre of the region.
        session_id:           Pipeline run UUID.
        source_type:          "satellite" | "drone"

    Returns:
        List of PairingRecord ORM objects ready for bulk insertion.
    """
    now = datetime.utcnow()

    past_by_id = {d.id: d for d in past_detections}

    # IDs that Sam3Tracker confirmed are present in the current frame
    tracked_past_ids = {t.past_detection_id for t in tracked_objects}

    matched_current_ids: set = set()
    pairing_records: List[PairingRecord] = []

    # ------------------------------------------------------------------
    # 1. "matched" – tracked by Sam3Tracker
    # ------------------------------------------------------------------
    for tracked in tracked_objects:
        past = past_by_id.get(tracked.past_detection_id)

        # Find the current Sam3Model detection with highest IoU overlap
        best_current: Optional[DetectionResult] = None
        best_iou = IOU_MATCH_THRESHOLD

        for cur in current_detections:
            if cur.detection_id in matched_current_ids:
                continue
            iou = _bbox_iou(
                tracked.bbox_x1, tracked.bbox_y1,
                tracked.bbox_x2, tracked.bbox_y2,
                cur.bbox_x1, cur.bbox_y1,
                cur.bbox_x2, cur.bbox_y2,
            )
            if iou > best_iou:
                best_iou = iou
                best_current = cur

        if best_current is not None:
            matched_current_ids.add(best_current.detection_id)

        # Determine if the object has moved significantly:
        # compare past lat/lon with current lat/lon
        cur_lat = best_current.lat if best_current else None
        cur_lon = best_current.lon if best_current else None
        past_lat = past.lat if past else None
        past_lon = past.lon if past else None

        if (cur_lat is not None and cur_lon is not None
                and past_lat is not None and past_lon is not None):
            dist = _geo_distance(cur_lat, cur_lon, past_lat, past_lon)
            status = "moved" if dist > MOVE_DISTANCE_THRESHOLD_DEG else "matched"
        else:
            status = "matched"

        pr = PairingRecord(
            pairing_time=now,
            lat_center=region_lat,
            lon_center=region_lon,
            # current side – may be None if Sam3Model missed this object
            current_detection_id=best_current.detection_id if best_current else None,
            current_object_class=best_current.object_class if best_current else tracked.past_object_class,
            current_confidence=best_current.confidence if best_current else tracked.score,
            current_lat=cur_lat,
            current_lon=cur_lon,
            current_capture_time=current_capture_time,
            current_bbox={
                "x1": best_current.bbox_x1, "y1": best_current.bbox_y1,
                "x2": best_current.bbox_x2, "y2": best_current.bbox_y2,
            } if best_current else {
                "x1": tracked.bbox_x1, "y1": tracked.bbox_y1,
                "x2": tracked.bbox_x2, "y2": tracked.bbox_y2,
            },
            # past side
            past_detection_id=tracked.past_detection_id,
            past_object_class=past.object_class if past else tracked.past_object_class,
            past_confidence=past.confidence if past else None,
            past_lat=past_lat,
            past_lon=past_lon,
            past_capture_time=past.detection_time if past else None,
            past_bbox={
                "x1": past.bbox_x1, "y1": past.bbox_y1,
                "x2": past.bbox_x2, "y2": past.bbox_y2,
            } if past else None,
            status=status,
            source_type=source_type,
            session_id=session_id,
        )
        pairing_records.append(pr)

    # ------------------------------------------------------------------
    # 2. "new" – current detections not matched to any tracked past object
    # ------------------------------------------------------------------
    for cur in current_detections:
        if cur.detection_id in matched_current_ids:
            continue
        pr = PairingRecord(
            pairing_time=now,
            lat_center=region_lat,
            lon_center=region_lon,
            current_detection_id=cur.detection_id,
            current_object_class=cur.object_class,
            current_confidence=cur.confidence,
            current_lat=cur.lat,
            current_lon=cur.lon,
            current_capture_time=current_capture_time,
            current_bbox={
                "x1": cur.bbox_x1, "y1": cur.bbox_y1,
                "x2": cur.bbox_x2, "y2": cur.bbox_y2,
            },
            status="new",
            source_type=source_type,
            session_id=session_id,
        )
        pairing_records.append(pr)

    # ------------------------------------------------------------------
    # 3. "disappeared" – past objects NOT returned by Sam3Tracker
    # ------------------------------------------------------------------
    for past in past_detections:
        if past.id in tracked_past_ids:
            continue
        pr = PairingRecord(
            pairing_time=now,
            lat_center=region_lat,
            lon_center=region_lon,
            past_detection_id=past.id,
            past_object_class=past.object_class,
            past_confidence=past.confidence,
            past_lat=past.lat,
            past_lon=past.lon,
            past_capture_time=past.detection_time,
            past_bbox={
                "x1": past.bbox_x1, "y1": past.bbox_y1,
                "x2": past.bbox_x2, "y2": past.bbox_y2,
            },
            status="disappeared",
            source_type=source_type,
            session_id=session_id,
        )
        pairing_records.append(pr)

    logger.info(
        f"[Pairing/tracker] ({region_lat:.4f}, {region_lon:.4f}): "
        f"{sum(1 for p in pairing_records if p.status == 'matched')} matched  "
        f"{sum(1 for p in pairing_records if p.status == 'moved')} moved  "
        f"{sum(1 for p in pairing_records if p.status == 'new')} new  "
        f"{sum(1 for p in pairing_records if p.status == 'disappeared')} disappeared"
    )
    return pairing_records


# ---------------------------------------------------------------------------
# Strategy B: similarity-based pairing (no SAM3 video tracker)
# ---------------------------------------------------------------------------

def _similarity_score(
    cur: DetectionResult,
    past: DetectionRecord,
) -> float:
    """
    Score in [0, 1 + SIMILARITY_CLASS_BONUS].
    Higher is better.  Returns -1 if outside the match radius.
    """
    geo_dist = _geo_distance(cur.lat, cur.lon, past.lat, past.lon)
    if geo_dist >= COORDINATE_MATCH_RADIUS_DEG:
        return -1.0
    geo_score = 1.0 - geo_dist / COORDINATE_MATCH_RADIUS_DEG
    class_bonus = SIMILARITY_CLASS_BONUS if cur.object_class == past.object_class else 0.0
    return geo_score + class_bonus


def pair_by_similarity(
    current_detections: List[DetectionResult],
    past_detections: List[DetectionRecord],
    current_capture_time: datetime,
    region_lat: float,
    region_lon: float,
    session_id: str,
    source_type: str = "satellite",
) -> List[PairingRecord]:
    """
    Build PairingRecord list by matching current detections to past detections
    using geo-distance + class similarity (no SAM3 video tracker required).

    Args:
        current_detections:   DetectionResult list for the current frame.
        past_detections:      DetectionRecord list for the most-recent past frame.
        current_capture_time: Capture timestamp of the current image.
        region_lat/lon:       Geographic centre of the region.
        session_id:           Pipeline run UUID.
        source_type:          "satellite" | "drone"

    Returns:
        List of PairingRecord ORM objects ready for bulk insertion.
    """
    now = datetime.utcnow()
    pairing_records: List[PairingRecord] = []

    # Build all (score, cur_idx, past_idx) candidate pairs within radius
    candidates = []
    for ci, cur in enumerate(current_detections):
        for pi, past in enumerate(past_detections):
            score = _similarity_score(cur, past)
            if score >= 0:
                candidates.append((score, ci, pi))

    # Greedy assignment: highest-score pair first, each object used once
    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_cur_ids: set = set()
    matched_past_ids: set = set()
    pairs: List[tuple] = []  # (DetectionResult, DetectionRecord)

    for _score, ci, pi in candidates:
        cur = current_detections[ci]
        past = past_detections[pi]
        if cur.detection_id in matched_cur_ids or past.id in matched_past_ids:
            continue
        matched_cur_ids.add(cur.detection_id)
        matched_past_ids.add(past.id)
        pairs.append((cur, past))

    # ------------------------------------------------------------------
    # 1. "matched" / "moved" – pairs within radius
    # ------------------------------------------------------------------
    for cur, past in pairs:
        dist = _geo_distance(cur.lat, cur.lon, past.lat, past.lon)
        status = "moved" if dist > MOVE_DISTANCE_THRESHOLD_DEG else "matched"
        pr = PairingRecord(
            pairing_time=now,
            lat_center=region_lat,
            lon_center=region_lon,
            current_detection_id=cur.detection_id,
            current_object_class=cur.object_class,
            current_confidence=cur.confidence,
            current_lat=cur.lat,
            current_lon=cur.lon,
            current_capture_time=current_capture_time,
            current_bbox={
                "x1": cur.bbox_x1, "y1": cur.bbox_y1,
                "x2": cur.bbox_x2, "y2": cur.bbox_y2,
            },
            past_detection_id=past.id,
            past_object_class=past.object_class,
            past_confidence=past.confidence,
            past_lat=past.lat,
            past_lon=past.lon,
            past_capture_time=past.detection_time,
            past_bbox={
                "x1": past.bbox_x1, "y1": past.bbox_y1,
                "x2": past.bbox_x2, "y2": past.bbox_y2,
            },
            status=status,
            source_type=source_type,
            session_id=session_id,
        )
        pairing_records.append(pr)

    # ------------------------------------------------------------------
    # 2. "new" – current detections with no past match
    # ------------------------------------------------------------------
    for cur in current_detections:
        if cur.detection_id in matched_cur_ids:
            continue
        pr = PairingRecord(
            pairing_time=now,
            lat_center=region_lat,
            lon_center=region_lon,
            current_detection_id=cur.detection_id,
            current_object_class=cur.object_class,
            current_confidence=cur.confidence,
            current_lat=cur.lat,
            current_lon=cur.lon,
            current_capture_time=current_capture_time,
            current_bbox={
                "x1": cur.bbox_x1, "y1": cur.bbox_y1,
                "x2": cur.bbox_x2, "y2": cur.bbox_y2,
            },
            status="new",
            source_type=source_type,
            session_id=session_id,
        )
        pairing_records.append(pr)

    # ------------------------------------------------------------------
    # 3. "disappeared" – past detections with no current match
    # ------------------------------------------------------------------
    for past in past_detections:
        if past.id in matched_past_ids:
            continue
        pr = PairingRecord(
            pairing_time=now,
            lat_center=region_lat,
            lon_center=region_lon,
            past_detection_id=past.id,
            past_object_class=past.object_class,
            past_confidence=past.confidence,
            past_lat=past.lat,
            past_lon=past.lon,
            past_capture_time=past.detection_time,
            past_bbox={
                "x1": past.bbox_x1, "y1": past.bbox_y1,
                "x2": past.bbox_x2, "y2": past.bbox_y2,
            },
            status="disappeared",
            source_type=source_type,
            session_id=session_id,
        )
        pairing_records.append(pr)

    logger.info(
        f"[Pairing/similarity] ({region_lat:.4f}, {region_lon:.4f}): "
        f"{sum(1 for p in pairing_records if p.status == 'matched')} matched  "
        f"{sum(1 for p in pairing_records if p.status == 'moved')} moved  "
        f"{sum(1 for p in pairing_records if p.status == 'new')} new  "
        f"{sum(1 for p in pairing_records if p.status == 'disappeared')} disappeared"
    )
    return pairing_records

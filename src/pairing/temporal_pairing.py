"""
Temporal object pairing.

For each detection in the *current* frame, find the closest matching detection
in the most recent *past* frame covering the same geographic region, then assign
one of three statuses:

  "new"          – present in current frame, no past match found
  "matched"      – present in both frames (optionally same class)
  "disappeared"  – present in past frame, no current match (generated from
                   the unmatched past detections)

Matching algorithm:
  1. Retrieve past detections via sensor_db.get_most_recent_past_detections().
  2. For each current detection, find the geographically nearest past detection
     of the same class within MAX_MATCH_DIST_DEG.
  3. Use greedy 1-to-1 assignment (best confidence wins).
  4. Unmatched past detections → "disappeared" pairing records.
"""

import logging
import uuid
from datetime import datetime
from math import sqrt
from typing import Dict, List, Optional, Tuple

from src.config import COORDINATE_MATCH_RADIUS_DEG
from src.database.models import DetectionRecord, PairingRecord
from src.database.sensor_db import get_most_recent_past_detections
from src.detection.sam2_detector import DetectionResult

logger = logging.getLogger(__name__)

# Maximum geographic distance (degrees) to consider two detections "the same object"
MAX_MATCH_DIST_DEG = COORDINATE_MATCH_RADIUS_DEG * 0.5


def _geo_dist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Euclidean distance in degrees (good approximation for small areas)."""
    return sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def _detection_result_to_pairing_fields(det: DetectionResult, capture_time: datetime) -> dict:
    return {
        "detection_id": det.detection_id,
        "object_class": det.object_class,
        "confidence": det.confidence,
        "lat": det.lat,
        "lon": det.lon,
        "capture_time": capture_time,
        "bbox": {
            "x1": det.bbox_x1,
            "y1": det.bbox_y1,
            "x2": det.bbox_x2,
            "y2": det.bbox_y2,
        },
    }


def _db_record_to_pairing_fields(rec: DetectionRecord) -> dict:
    return {
        "detection_id": rec.id,
        "object_class": rec.object_class,
        "confidence": rec.confidence,
        "lat": rec.lat,
        "lon": rec.lon,
        "capture_time": rec.detection_time,
        "bbox": {
            "x1": rec.bbox_x1,
            "y1": rec.bbox_y1,
            "x2": rec.bbox_x2,
            "y2": rec.bbox_y2,
        },
    }


def pair_detections(
    current_detections: List[DetectionResult],
    current_capture_time: datetime,
    region_lat: float,
    region_lon: float,
    session_id: str,
    source_type: str = "satellite",
) -> List[PairingRecord]:
    """
    Main pairing entry point for one region/image.

    Args:
        current_detections:  SAM2 detection results from the current frame.
        current_capture_time: Capture timestamp of the current image.
        region_lat/lon:      Geographic centre of the image region.
        session_id:          Pipeline run identifier.
        source_type:         "satellite" | "drone"

    Returns:
        List of PairingRecord ORM objects ready for bulk insertion.
    """
    now = datetime.utcnow()

    # 1. Retrieve past detections for this region
    past_records: List[DetectionRecord] = get_most_recent_past_detections(
        lat_center=region_lat,
        lon_center=region_lon,
        radius_deg=COORDINATE_MATCH_RADIUS_DEG,
        before_time=current_capture_time,
    )

    logger.info(
        f"[Pairing] {len(current_detections)} current detections, "
        f"{len(past_records)} past detections for region "
        f"({region_lat:.4f}, {region_lon:.4f})"
    )

    # 2. Build cost matrix (current × past) using geo distance, same-class bonus
    paired_past_ids: set = set()   # past_detection ids already matched
    pairing_records: List[PairingRecord] = []

    # Sort current detections by confidence (descending) to prioritise high-confidence
    current_sorted = sorted(current_detections, key=lambda d: d.confidence, reverse=True)

    for cur in current_sorted:
        best_past: Optional[DetectionRecord] = None
        best_dist = float("inf")

        for past in past_records:
            if past.id in paired_past_ids:
                continue
            dist = _geo_dist(cur.lat, cur.lon, past.lat, past.lon)
            if dist > MAX_MATCH_DIST_DEG:
                continue
            # Prefer same class; penalise class mismatch
            class_penalty = 0.0 if cur.object_class == past.object_class else MAX_MATCH_DIST_DEG * 0.3
            effective_dist = dist + class_penalty
            if effective_dist < best_dist:
                best_dist = effective_dist
                best_past = past

        if best_past is not None:
            # Matched pair
            paired_past_ids.add(best_past.id)
            status = "matched"
            pr = PairingRecord(
                pairing_time=now,
                lat_center=region_lat,
                lon_center=region_lon,
                # Current
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
                # Past
                past_detection_id=best_past.id,
                past_object_class=best_past.object_class,
                past_confidence=best_past.confidence,
                past_lat=best_past.lat,
                past_lon=best_past.lon,
                past_capture_time=best_past.detection_time,
                past_bbox={
                    "x1": best_past.bbox_x1, "y1": best_past.bbox_y1,
                    "x2": best_past.bbox_x2, "y2": best_past.bbox_y2,
                },
                status=status,
                source_type=source_type,
                session_id=session_id,
            )
        else:
            # New object – no past match
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

    # 3. Unmatched past detections → "disappeared"
    for past in past_records:
        if past.id in paired_past_ids:
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
        f"[Pairing] Generated {len(pairing_records)} pairing records "
        f"({sum(1 for p in pairing_records if p.status == 'matched')} matched, "
        f"{sum(1 for p in pairing_records if p.status == 'new')} new, "
        f"{sum(1 for p in pairing_records if p.status == 'disappeared')} disappeared)"
    )
    return pairing_records

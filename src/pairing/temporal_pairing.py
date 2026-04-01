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

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from src.config import (
    CLIP_MODEL_NAME,
    COORDINATE_MATCH_RADIUS_DEG,
    MOVE_DISTANCE_THRESHOLD_DEG,
    SIMILARITY_CLIP_WEIGHT,
    SIMILARITY_MATCH_THRESHOLD,
    SIMILARITY_SIZE_WEIGHT,
)
from src.database.models import DetectionRecord, PairingRecord
from src.database.sensor_db import (
    get_image_record_by_id,
    get_most_recent_past_detections,
    insert_detections_bulk,
)
from src.detection.sam2_detector import DetectionResult, TrackedObject

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Minimum bbox IoU to accept a TrackedObject ↔ DetectionResult match
IOU_MATCH_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Static (building) object classes – cannot physically move between frames.
# Pairing rules:
#   1. "moved" status is never assigned – always "matched".
#   2. Only pair two detections whose geo distance ≤ MOVE_DISTANCE_THRESHOLD_DEG
#      (same location constraint).
#   3. When one side is missing a detection, attempt cross-image similarity:
#      crop the same geo region from both images and compare with CLIP.
#      If similarity ≥ _STATIC_SIM_THRESHOLD → synthesise a DetectionRecord
#      for the missing frame, insert into sensor DB, and pair as "matched".
# ---------------------------------------------------------------------------
_STATIC_CLASSES: frozenset = frozenset({
    "civilian building",
    "command post",
    "fuel storage",
    "supply depot",
    "military building",
})
_STATIC_SIM_THRESHOLD: float = 0.5   # CLIP cosine similarity threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _naive(dt: Optional[datetime]) -> Optional[datetime]:
    """Strip timezone info so naive/aware datetimes can be compared safely."""
    if dt is None:
        return None
    return dt.replace(tzinfo=None)


def _geo_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Simple Euclidean distance in degrees between two lat/lon points."""
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5


def _size_similarity(a: "DetectionResult", b: "DetectionResult") -> float:
    """두 탐지 객체의 크기 유사도 (0~1).

    mask_area_px 가 유효하면 세그멘테이션 면적 사용, 없으면 bbox 면적 사용.
    min/max 비율이므로 크기가 같을수록 1.0, 차이가 클수록 0에 가까워진다.
    """
    def _area(det: "DetectionResult") -> float:
        if det.mask_area_px and det.mask_area_px > 0:
            return det.mask_area_px
        return max((det.bbox_x2 - det.bbox_x1) * (det.bbox_y2 - det.bbox_y1), 1e-6)

    area_a = _area(a)
    area_b = _area(b)
    return min(area_a, area_b) / max(area_a, area_b)


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
# Static-object cross-image similarity helpers
# ---------------------------------------------------------------------------

def _geo_bbox_on_image(
    lat_top: float, lon_left: float, lat_bottom: float, lon_right: float,
    img_w: int, img_h: int,
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    padding: int = 4,
) -> Optional[tuple]:
    """
    Convert geographic bbox (lat_top, lon_left, lat_bottom, lon_right) to
    pixel bbox (x1, y1, x2, y2) on an image defined by its geo bounds.

    pixel_to_geo formula (from image_loader.py):
        lat = lat_max - (cy / h) * (lat_max - lat_min)
        lon = lon_min + (cx / w) * (lon_max - lon_min)
    Inverse:
        cy = (lat_max - lat) / (lat_max - lat_min) * h
        cx = (lon - lon_min) / (lon_max - lon_min) * w

    Returns None if the bbox falls entirely outside the image bounds.
    """
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    if lat_span <= 0 or lon_span <= 0:
        return None

    x1 = int((lon_left  - lon_min) / lon_span * img_w) - padding
    x2 = int((lon_right - lon_min) / lon_span * img_w) + padding
    y1 = int((lat_max - lat_top)    / lat_span * img_h) - padding
    y2 = int((lat_max - lat_bottom) / lat_span * img_h) + padding

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _det_geo_bbox(det, img_record) -> Optional[tuple]:
    """
    Return the geographic bounding box of a detection as
    (lat_top, lon_left, lat_bottom, lon_right).

    Works with both DetectionResult (dataclass) and DetectionRecord (ORM)
    because both expose the same field names.
    """
    w = img_record.det_width
    h = img_record.det_height
    if not w or not h:
        return None
    lat_min = img_record.lat_min
    lat_max = img_record.lat_max
    lon_min = img_record.lon_min
    lon_max = img_record.lon_max
    if None in (lat_min, lat_max, lon_min, lon_max):
        return None

    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min

    lat_top    = lat_max - (det.bbox_y1 / h) * lat_span
    lat_bottom = lat_max - (det.bbox_y2 / h) * lat_span
    lon_left   = lon_min + (det.bbox_x1 / w) * lon_span
    lon_right  = lon_min + (det.bbox_x2 / w) * lon_span
    return lat_top, lon_left, lat_bottom, lon_right


def _load_pil_for_image_id(image_id: str) -> Optional["PILImage"]:
    """Load PIL image (with super-resolution) for a given image_id."""
    from PIL import Image as PILImage
    from src.detection.super_resolution import super_resolve

    record = get_image_record_by_id(image_id)
    if record is None:
        return None
    from pathlib import Path as _Path
    p = _Path(record.image_path)
    if not p.exists():
        return None
    img = PILImage.open(p).convert("RGB")
    arr = super_resolve(np.array(img, dtype=np.uint8))
    return PILImage.fromarray(arr)


def _clip_similarity_crops(crop_a: "PILImage", crop_b: "PILImage") -> float:
    """Return CLIP cosine similarity between two PIL crops. Falls back to 0.0 on error."""
    try:
        embeds = _clip_embedder.embed([crop_a, crop_b])   # (2, D)
        return float(embeds[0] @ embeds[1])
    except Exception as exc:
        logger.warning(f"[StaticCrossCheck] CLIP similarity failed: {exc}")
        return 0.0


def _static_cross_check(
    det,                    # DetectionResult or DetectionRecord – the side WITH a bbox
    img_rec_with: "ImageRecord",
    pil_with: "PILImage",
    img_rec_without: "ImageRecord",
    pil_without: "PILImage",
    capture_time_without,   # datetime for the "missing" frame
    session_id: str,
    source_type: str,
) -> Optional[DetectionRecord]:
    """
    For a static-class object detected in one frame but absent in the other:
    1. Crop the detected bbox from pil_with.
    2. Crop the same geographic region from pil_without.
    3. Compute CLIP cosine similarity.
    4. If similarity ≥ _STATIC_SIM_THRESHOLD, create and insert a synthetic
       DetectionRecord for the "missing" frame, then return it.
    5. Otherwise return None.
    """
    geo_box = _det_geo_bbox(det, img_rec_with)
    if geo_box is None:
        return None
    lat_top, lon_left, lat_bottom, lon_right = geo_box

    # Crop from the "with bbox" image
    w_with = img_rec_with.det_width or pil_with.width
    h_with = img_rec_with.det_height or pil_with.height
    coords_with = _geo_bbox_on_image(
        lat_top, lon_left, lat_bottom, lon_right,
        w_with, h_with,
        img_rec_with.lat_min, img_rec_with.lat_max,
        img_rec_with.lon_min, img_rec_with.lon_max,
    )
    if coords_with is None:
        return None
    cx1, cy1, cx2, cy2 = coords_with
    crop_with = pil_with.crop((cx1, cy1, cx2, cy2))

    # Crop the same geo region from the "without bbox" image
    w_wo = img_rec_without.det_width or pil_without.width
    h_wo = img_rec_without.det_height or pil_without.height
    coords_wo = _geo_bbox_on_image(
        lat_top, lon_left, lat_bottom, lon_right,
        w_wo, h_wo,
        img_rec_without.lat_min, img_rec_without.lat_max,
        img_rec_without.lon_min, img_rec_without.lon_max,
    )
    if coords_wo is None:
        return None
    wx1, wy1, wx2, wy2 = coords_wo
    crop_wo = pil_without.crop((wx1, wy1, wx2, wy2))

    sim = _clip_similarity_crops(crop_with, crop_wo)
    logger.info(
        f"[StaticCrossCheck] {det.object_class}  sim={sim:.3f}  "
        f"threshold={_STATIC_SIM_THRESHOLD}"
    )
    if sim < _STATIC_SIM_THRESHOLD:
        return None

    # Synthesise a DetectionRecord for the "without" frame
    from src.detection.image_loader import pixel_to_geo
    from src.database.models import ImageRecord as _IRModel

    # Pixel-space bbox on the "without" image
    cx_px = (wx1 + wx2) / 2.0
    cy_px = (wy1 + wy2) / 2.0
    # Geo centre from the known detection (more accurate than re-projecting)
    lat_c = det.lat
    lon_c = det.lon

    synthetic = DetectionRecord(
        image_id=img_rec_without.id,
        detection_time=capture_time_without,
        object_class=det.object_class,
        object_class_index=getattr(det, "object_class_index", None),
        confidence=det.confidence,
        bbox_x1=float(wx1),
        bbox_y1=float(wy1),
        bbox_x2=float(wx2),
        bbox_y2=float(wy2),
        lat=lat_c,
        lon=lon_c,
        mask_rle=None,
        mask_area_px=None,
        source_type=source_type,
        session_id=session_id,
    )
    ids = insert_detections_bulk([synthetic])
    synthetic.id = ids[0]
    logger.info(
        f"[StaticCrossCheck] Synthetic DetectionRecord inserted  "
        f"id={synthetic.id}  class={synthetic.object_class}  "
        f"sim={sim:.3f}  image_id={img_rec_without.id[:8]}"
    )
    return synthetic


# ---------------------------------------------------------------------------
# Main pairing function
# ---------------------------------------------------------------------------

def pair_by_tracking(
    tracked_objects: List[TrackedObject],
    current_detections: List[DetectionResult],
    past_detections: List[DetectionRecord],
    current_capture_time: datetime,
    past_capture_time: Optional[datetime],
    region_lat: float,
    region_lon: float,
    session_id: str,
    source_type: str = "satellite",
) -> List[PairingRecord]:
    """
    Build PairingRecord list using Sam3Tracker object IDs.

    Status assignment is explicitly based on metadata.json capture times:
      "new"         – object only in the LATER frame (current_capture_time)
      "matched"     – object in BOTH the earlier frame (past_capture_time) AND
                      the later frame (current_capture_time), same position
      "moved"       – same as matched but position changed significantly
      "disappeared" – object only in the EARLIER frame (past_capture_time)

    Args:
        tracked_objects:      Output of SAM3Detector.track_objects() –
                              past objects confirmed present in current frame.
        current_detections:   Sam3Model detections for the current frame
                              (from sensor DB, already stored).
        past_detections:      DetectionRecord list for the most-recent past frame.
        current_capture_time: Capture timestamp of the current (later) image.
        past_capture_time:    Capture timestamp of the past (earlier) image batch.
        region_lat/lon:       Geographic centre of the region.
        session_id:           Pipeline run UUID.
        source_type:          "satellite" | "drone"

    Returns:
        List of PairingRecord ORM objects ready for bulk insertion.
    """
    now = datetime.utcnow()

    # Guard: current frame must be strictly later than past frame.
    # Use naive datetimes to avoid offset-naive vs offset-aware comparison errors.
    if past_capture_time is not None and _naive(current_capture_time) <= _naive(past_capture_time):
        logger.warning(
            f"[Pairing/tracker] Skipping region ({region_lat:.4f}, {region_lon:.4f}): "
            f"current_capture_time ({current_capture_time}) is not later than "
            f"past_capture_time ({past_capture_time}). Check metadata.json ordering."
        )
        return []

    # Only pair against past detections that truly belong to the earlier frame.
    if past_capture_time is not None:
        past_detections = [
            d for d in past_detections
            if _naive(d.detection_time) <= _naive(past_capture_time)
        ]

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

        # 비교 기준 클래스: past DB 레코드가 있으면 그 클래스, 없으면 tracker가 기억한 클래스
        expected_cls = (
            past.object_class if past else tracked.past_object_class or ""
        ).lower()

        for cur in current_detections:
            if cur.detection_id in matched_current_ids:
                continue
            # 동일 클래스인 경우만 같은 객체 후보로 허용
            if cur.object_class.lower() != expected_cls:
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
            # 건물류는 물리적으로 이동 불가 → 항상 matched
            if expected_cls in _STATIC_CLASSES:
                status = "matched"
            else:
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
            past_capture_time=past_capture_time,
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
            past_capture_time=past_capture_time,
            past_bbox={
                "x1": past.bbox_x1, "y1": past.bbox_y1,
                "x2": past.bbox_x2, "y2": past.bbox_y2,
            },
            status="disappeared",
            source_type=source_type,
            session_id=session_id,
        )
        pairing_records.append(pr)

    # ------------------------------------------------------------------
    # 4. Static-class cross-image check for "new" and "disappeared"
    #    When a building is unmatched (new or disappeared), compare crops
    #    from both images.  If similarity ≥ threshold → synthetic detection
    #    inserted and pair recorded as "matched".
    # ------------------------------------------------------------------
    static_new        = [p for p in pairing_records
                         if p.status == "new"
                         and (p.current_object_class or "").lower() in _STATIC_CLASSES]
    static_disappeared = [p for p in pairing_records
                          if p.status == "disappeared"
                          and (p.past_object_class or "").lower() in _STATIC_CLASSES]

    if static_new or static_disappeared:
        # Collect distinct image_ids from current detections to load PILs
        cur_image_ids = list({d.image_id for d in current_detections if d.image_id})
        past_image_ids = list({d.image_id for d in past_detections if d.image_id})
        cur_img_cache: Dict[str, Optional] = {}
        past_img_cache: Dict[str, Optional] = {}

        def _get_img_rec_pil(iid: str, cache: dict):
            if iid not in cache:
                rec = get_image_record_by_id(iid)
                pil = _load_pil_for_image_id(iid) if rec else None
                cache[iid] = (rec, pil)
            return cache[iid]

        # For "new" static: past side is missing → load current PIL, check past image
        for pr in static_new:
            if pr.current_detection_id is None:
                continue
            # find the DetectionResult
            cur_det = next(
                (d for d in current_detections if d.detection_id == pr.current_detection_id),
                None,
            )
            if cur_det is None or not cur_det.image_id:
                continue
            cur_rec, cur_pil = _get_img_rec_pil(cur_det.image_id, cur_img_cache)
            if cur_rec is None or cur_pil is None:
                continue
            # Pick the past image that covers the same region
            if not past_image_ids:
                continue
            past_iid = past_image_ids[0]
            past_rec, past_pil = _get_img_rec_pil(past_iid, past_img_cache)
            if past_rec is None or past_pil is None:
                continue
            synth = _static_cross_check(
                det=cur_det,
                img_rec_with=cur_rec,
                pil_with=cur_pil,
                img_rec_without=past_rec,
                pil_without=past_pil,
                capture_time_without=past_capture_time or current_capture_time,
                session_id=session_id,
                source_type=source_type,
            )
            if synth is not None:
                pr.status = "matched"
                pr.past_detection_id = synth.id
                pr.past_object_class = synth.object_class
                pr.past_confidence = synth.confidence
                pr.past_lat = synth.lat
                pr.past_lon = synth.lon
                pr.past_capture_time = synth.detection_time
                pr.past_bbox = {
                    "x1": synth.bbox_x1, "y1": synth.bbox_y1,
                    "x2": synth.bbox_x2, "y2": synth.bbox_y2,
                }

        # For "disappeared" static: current side is missing → load past PIL, check current image
        for pr in static_disappeared:
            if pr.past_detection_id is None:
                continue
            past_det = next(
                (d for d in past_detections if d.id == pr.past_detection_id),
                None,
            )
            if past_det is None or not past_det.image_id:
                continue
            past_rec, past_pil = _get_img_rec_pil(past_det.image_id, past_img_cache)
            if past_rec is None or past_pil is None:
                continue
            if not cur_image_ids:
                continue
            cur_iid = cur_image_ids[0]
            cur_rec, cur_pil = _get_img_rec_pil(cur_iid, cur_img_cache)
            if cur_rec is None or cur_pil is None:
                continue
            synth = _static_cross_check(
                det=past_det,
                img_rec_with=past_rec,
                pil_with=past_pil,
                img_rec_without=cur_rec,
                pil_without=cur_pil,
                capture_time_without=current_capture_time,
                session_id=session_id,
                source_type=source_type,
            )
            if synth is not None:
                pr.status = "matched"
                pr.current_detection_id = synth.id
                pr.current_object_class = synth.object_class
                pr.current_confidence = synth.confidence
                pr.current_lat = synth.lat
                pr.current_lon = synth.lon
                pr.current_capture_time = synth.detection_time
                pr.current_bbox = {
                    "x1": synth.bbox_x1, "y1": synth.bbox_y1,
                    "x2": synth.bbox_x2, "y2": synth.bbox_y2,
                }

    logger.info(
        f"[Pairing/tracker] ({region_lat:.4f}, {region_lon:.4f}): "
        f"{sum(1 for p in pairing_records if p.status == 'matched')} matched  "
        f"{sum(1 for p in pairing_records if p.status == 'moved')} moved  "
        f"{sum(1 for p in pairing_records if p.status == 'new')} new  "
        f"{sum(1 for p in pairing_records if p.status == 'disappeared')} disappeared"
    )
    return pairing_records


# ---------------------------------------------------------------------------
# Strategy B: SAM mask crop + CLIP embedding similarity
# ---------------------------------------------------------------------------

class _CLIPEmbedder:
    """Lazy-loaded CLIP image encoder (module-level singleton)."""

    def __init__(self):
        self._model = None
        self._processor = None

    def _load(self):
        from transformers import AutoModel, AutoProcessor
        logger.info(f"[CLIPEmbedder] Loading {CLIP_MODEL_NAME} ...")
        self._processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
        self._model = AutoModel.from_pretrained(CLIP_MODEL_NAME)
        self._model.eval()
        logger.info("[CLIPEmbedder] Ready.")

    def embed(self, crops: list) -> np.ndarray:
        """
        Compute L2-normalised image embeddings for a list of PIL crops.
        Supports both CLIPModel (get_image_features) and plain ViTModel
        (pooler_output / CLS token from last_hidden_state).
        Returns float32 array of shape (N, D).
        """
        import torch

        if self._model is None:
            self._load()

        inputs = self._processor(images=crops, return_tensors="pt", padding=True)
        # CLIPProcessor may include text keys – keep only vision inputs for ViT
        vision_inputs = {k: v for k, v in inputs.items() if k == "pixel_values"}

        with torch.no_grad():
            if hasattr(self._model, "get_image_features"):
                # CLIPModel path
                feats = self._model.get_image_features(**inputs)   # (N, D)
            else:
                # ViTModel / CLIPVisionModel path
                out = self._model(**vision_inputs)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    feats = out.pooler_output                       # (N, D)
                else:
                    feats = out.last_hidden_state[:, 0, :]          # CLS token

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


_clip_embedder = _CLIPEmbedder()


def _decode_rle(mask_rle_json: str) -> Optional[np.ndarray]:
    """Decode a JSON RLE string produced by sam2_detector._encode_rle()."""
    try:
        data = json.loads(mask_rle_json)
        h, w = data["shape"]
        flat: list = []
        for val, count in data["rle"]:
            flat.extend([val] * count)
        return np.array(flat, dtype=bool).reshape(h, w)
    except Exception:
        return None


def _mask_crop(image: "PILImage", x1: float, y1: float, x2: float, y2: float,
               mask_rle: Optional[str], padding: int = 4) -> Optional["PILImage"]:
    """
    Crop the object region from *image* using the SAM segmentation mask.

    If mask_rle is available the background is zeroed out before cropping,
    so CLIP sees only the object pixels.  Falls back to a plain bbox crop
    when mask_rle is absent or cannot be decoded.
    Returns None when the bounding box is degenerate (zero or negative area).
    """
    from PIL import Image as PILImage

    iw, ih = image.size
    bx1 = max(0, int(x1) - padding)
    by1 = max(0, int(y1) - padding)
    bx2 = min(iw, int(x2) + padding)
    by2 = min(ih, int(y2) + padding)

    if bx2 <= bx1 or by2 <= by1:
        logger.warning(f"[MaskCrop] Degenerate bbox ({x1},{y1},{x2},{y2}); skipping crop.")
        return None

    if mask_rle:
        mask = _decode_rle(mask_rle)
        if mask is not None and mask.shape == (ih, iw):
            arr = np.array(image)
            arr[~mask] = 0          # zero out background
            crop = arr[by1:by2, bx1:bx2]
            return PILImage.fromarray(crop)

    # Fallback: plain bbox crop
    return image.crop((bx1, by1, bx2, by2))


def _load_pil_image(image_id: str) -> Optional["PILImage"]:
    """Load a PIL image from disk given a sensor-DB image_id.

    Applies super_resolve so that the returned image dimensions match the
    resolution at which bbox coordinates were originally computed.
    """
    from PIL import Image as PILImage
    from src.detection.super_resolution import super_resolve

    record = get_image_record_by_id(image_id)
    if record is None:
        logger.warning(f"[CLIPSim] ImageRecord not found for image_id={image_id}")
        return None
    path = Path(record.image_path)
    if not path.exists():
        logger.warning(f"[CLIPSim] Image file missing: {path}")
        return None
    img = PILImage.open(path).convert("RGB")
    arr = super_resolve(np.array(img, dtype=np.uint8))
    return PILImage.fromarray(arr)


def _compute_embeddings(
    detections: list,
    image: Optional["PILImage"],
    image_cache: Dict[str, Optional["PILImage"]],
    is_current: bool,
) -> np.ndarray:
    """
    Produce CLIP embeddings for *detections*.

    For current-frame detections the caller supplies *image* directly.
    For past-frame detections the image is looked up by image_id and cached
    in *image_cache*.  Objects whose source image cannot be loaded receive a
    zero vector (they will never win the greedy assignment).

    Returns float32 array of shape (N, D).
    """
    crops = []
    valid_indices = []

    for i, det in enumerate(detections):
        if is_current:
            src_image = image
        else:
            iid = det.image_id
            if iid not in image_cache:
                image_cache[iid] = _load_pil_image(iid)
            src_image = image_cache[iid]

        if src_image is None:
            continue

        crop = _mask_crop(
            src_image,
            det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2,
            det.mask_rle,
        )
        if crop is None:
            continue
        crops.append(crop)
        valid_indices.append(i)

    if not crops:
        raise ValueError(f"No valid crops for any of the {len(detections)} detections; cannot compute embeddings.")

    embeds_valid = _clip_embedder.embed(crops)          # (len(valid), D)
    D = embeds_valid.shape[1]
    embeds = np.zeros((len(detections), D), dtype=np.float32)
    for out_idx, orig_idx in enumerate(valid_indices):
        embeds[orig_idx] = embeds_valid[out_idx]
    return embeds


def pair_by_similarity(
    current_detections: List[DetectionResult],
    past_detections: List[DetectionRecord],
    current_image: "PILImage",
    current_capture_time: datetime,
    past_capture_time: Optional[datetime],
    region_lat: float,
    region_lon: float,
    session_id: str,
    source_type: str = "satellite",
) -> List[PairingRecord]:
    """
    Build PairingRecord list by matching current detections to past detections
    using SAM mask crop + CLIP embedding cosine similarity.

    Status assignment is explicitly based on metadata.json capture times:
      "new"         – object only in the LATER frame (current_capture_time):
                      unmatched current detections
      "matched"     – object in BOTH the earlier frame (past_capture_time) AND
                      the later frame (current_capture_time), same position
      "moved"       – same as matched but position changed significantly
      "disappeared" – object only in the EARLIER frame (past_capture_time):
                      unmatched past detections

    Scoring (CLIP cosine similarity; geo proximity as fallback):
        score = cosine_similarity(cur_embed, past_embed)   ∈ [-1, 1]
    Greedy assignment by descending score; pair accepted only when
    score > SIMILARITY_MATCH_THRESHOLD.

    Args:
        current_detections:  DetectionResult list for the current (later) frame.
        past_detections:     DetectionRecord list for the most-recent past frame.
        current_image:       PIL image of the current frame (for crop + embed).
        current_capture_time: Capture timestamp of the current (later) image.
        past_capture_time:   Capture timestamp of the past (earlier) image batch.
        region_lat/lon:      Geographic centre of the region.
        session_id:          Pipeline run UUID.
        source_type:         "satellite" | "drone"

    Returns:
        List of PairingRecord ORM objects ready for bulk insertion.
    """
    now = datetime.utcnow()
    pairing_records: List[PairingRecord] = []

    # Guard: current frame must be strictly later than past frame.
    # Use naive datetimes to avoid offset-naive vs offset-aware comparison errors.
    if past_capture_time is not None and _naive(current_capture_time) <= _naive(past_capture_time):
        logger.warning(
            f"[Pairing/similarity] Skipping region ({region_lat:.4f}, {region_lon:.4f}): "
            f"current_capture_time ({current_capture_time}) is not later than "
            f"past_capture_time ({past_capture_time}). Check metadata.json ordering."
        )
        return []

    # Only pair against past detections that truly belong to the earlier frame.
    # detection_time is set to meta.capture_time at ingestion time.
    if past_capture_time is not None:
        past_detections = [
            d for d in past_detections
            if _naive(d.detection_time) <= _naive(past_capture_time)
        ]

    if not current_detections or not past_detections:
        # Nothing to match – all current are "new", all past are "disappeared"
        for cur in current_detections:
            pairing_records.append(PairingRecord(
                pairing_time=now, lat_center=region_lat, lon_center=region_lon,
                current_detection_id=cur.detection_id,
                current_object_class=cur.object_class,
                current_confidence=cur.confidence,
                current_lat=cur.lat, current_lon=cur.lon,
                current_capture_time=current_capture_time,
                current_bbox={"x1": cur.bbox_x1, "y1": cur.bbox_y1,
                              "x2": cur.bbox_x2, "y2": cur.bbox_y2},
                status="new", source_type=source_type, session_id=session_id,
            ))
        for past in past_detections:
            pairing_records.append(PairingRecord(
                pairing_time=now, lat_center=region_lat, lon_center=region_lon,
                past_detection_id=past.id,
                past_object_class=past.object_class,
                past_confidence=past.confidence,
                past_lat=past.lat, past_lon=past.lon,
                past_capture_time=past_capture_time,
                past_bbox={"x1": past.bbox_x1, "y1": past.bbox_y1,
                           "x2": past.bbox_x2, "y2": past.bbox_y2},
                status="disappeared", source_type=source_type, session_id=session_id,
            ))
        return pairing_records

    # ------------------------------------------------------------------
    # Compute CLIP embeddings (batch per frame)
    # ------------------------------------------------------------------
    past_image_cache: Dict[str, Optional] = {}
    try:
        cur_embeds = _compute_embeddings(
            current_detections, current_image, {}, is_current=True,
        )                                                    # (N, D)
        past_embeds = _compute_embeddings(
            past_detections, None, past_image_cache, is_current=False,
        )                                                    # (M, D)
        # Cosine similarity matrix (N, M); embeddings are already L2-normalised
        clip_sim_matrix = cur_embeds @ past_embeds.T        # (N, M)
    except Exception as exc:
        logger.warning(f"[CLIPSim] Embedding failed ({exc}); falling back to geo-only scoring.")
        clip_sim_matrix = None

    # ------------------------------------------------------------------
    # Build scored candidate pairs (all combinations; no geo hard filter)
    # Score = CLIP cosine similarity when available, otherwise geo score.
    # ------------------------------------------------------------------
    candidates = []   # (score, cur_idx, past_idx)
    for ci, cur in enumerate(current_detections):
        for pi, past in enumerate(past_detections):
            # 동일 클래스인 경우만 같은 객체 후보로 허용
            if cur.object_class.lower() != past.object_class.lower():
                continue
            # 건물류: 위경도가 동일(MOVE_DISTANCE_THRESHOLD_DEG 이내)한 쌍만 허용
            if cur.object_class.lower() in _STATIC_CLASSES:
                if _geo_distance(cur.lat, cur.lon, past.lat, past.lon) > MOVE_DISTANCE_THRESHOLD_DEG:
                    continue
            if clip_sim_matrix is not None:
                base_score = float(clip_sim_matrix[ci, pi])
            else:
                # Fallback: geo proximity score when CLIP is unavailable
                geo_dist = _geo_distance(cur.lat, cur.lon, past.lat, past.lon)
                base_score = max(0.0, 1.0 - geo_dist / COORDINATE_MATCH_RADIUS_DEG)
            size_sim = _size_similarity(cur, past)
            score = (1.0 - SIMILARITY_SIZE_WEIGHT) * base_score + SIMILARITY_SIZE_WEIGHT * size_sim
            candidates.append((score, ci, pi))

    # Iterative per-ci best-pi assignment (Gale-Shapley / deferred-acceptance):
    #
    #  1. ci는 threshold 초과 pi 중 최고 점수 pi에 제안한다.
    #  2. pi가 비어 있으면 → 잠정 매칭.
    #  3. pi에 이미 다른 ci가 매칭돼 있으면:
    #       - 새 ci의 점수가 더 높으면  → 새 ci가 pi를 획득, 기존 ci는 차선 pi로 재시도.
    #       - 새 ci의 점수가 낮으면     → 새 ci는 차선 pi로 재시도.
    #  4. 시도할 pi가 더 없으면 → ci는 "new" 처리.
    #
    # ci_ptr 이 항상 단조 증가하므로 유한 횟수 안에 반드시 종료된다.
    # ------------------------------------------------------------------

    # 각 ci의 후보 목록 (점수 내림차순, threshold 초과만)
    ci_candidates: dict[int, list] = {}
    for score, ci, pi in candidates:
        if score > SIMILARITY_MATCH_THRESHOLD:
            ci_candidates.setdefault(ci, []).append((score, pi))
    for ci in ci_candidates:
        ci_candidates[ci].sort(key=lambda x: x[0], reverse=True)

    ci_ptr:  dict[int, int]             = {ci: 0 for ci in ci_candidates}
    pi_match: dict[int, tuple[float, int]] = {}   # pi -> (score, ci)

    unmatched: set = set(ci_candidates)
    while unmatched:
        ci    = unmatched.pop()
        ptr   = ci_ptr[ci]
        if ptr >= len(ci_candidates[ci]):
            continue                              # 후보 소진 → "new"
        score, pi  = ci_candidates[ci][ptr]
        ci_ptr[ci] = ptr + 1

        if pi not in pi_match:
            pi_match[pi] = (score, ci)            # pi 비어있음 → 즉시 매칭
        elif score > pi_match[pi][0]:
            _, displaced = pi_match[pi]
            pi_match[pi] = (score, ci)            # 더 높은 점수 → ci가 pi 획득
            unmatched.add(displaced)              # 밀려난 ci 재시도
        else:
            unmatched.add(ci)                     # 낮은 점수 → ci 차선으로 재시도

    matched_cur_ids: set = set()
    matched_past_ids: set = set()
    pairs: list = []   # [(DetectionResult, DetectionRecord)]
    for pi, (_, ci) in pi_match.items():
        cur  = current_detections[ci]
        past = past_detections[pi]
        matched_cur_ids.add(cur.detection_id)
        matched_past_ids.add(past.id)
        pairs.append((cur, past))

    # ------------------------------------------------------------------
    # 1. "matched" / "moved"
    # ------------------------------------------------------------------
    for cur, past in pairs:
        # 건물류는 이동 불가 → 항상 matched
        if cur.object_class.lower() in _STATIC_CLASSES:
            status = "matched"
        else:
            dist = _geo_distance(cur.lat, cur.lon, past.lat, past.lon)
            status = "moved" if dist > MOVE_DISTANCE_THRESHOLD_DEG else "matched"
        pairing_records.append(PairingRecord(
            pairing_time=now, lat_center=region_lat, lon_center=region_lon,
            current_detection_id=cur.detection_id,
            current_object_class=cur.object_class,
            current_confidence=cur.confidence,
            current_lat=cur.lat, current_lon=cur.lon,
            current_capture_time=current_capture_time,
            current_bbox={"x1": cur.bbox_x1, "y1": cur.bbox_y1,
                          "x2": cur.bbox_x2, "y2": cur.bbox_y2},
            past_detection_id=past.id,
            past_object_class=past.object_class,
            past_confidence=past.confidence,
            past_lat=past.lat, past_lon=past.lon,
            past_capture_time=past_capture_time,
            past_bbox={"x1": past.bbox_x1, "y1": past.bbox_y1,
                       "x2": past.bbox_x2, "y2": past.bbox_y2},
            status=status, source_type=source_type, session_id=session_id,
        ))

    # ------------------------------------------------------------------
    # 2. "new"
    # ------------------------------------------------------------------
    for cur in current_detections:
        if cur.detection_id in matched_cur_ids:
            continue
        pairing_records.append(PairingRecord(
            pairing_time=now, lat_center=region_lat, lon_center=region_lon,
            current_detection_id=cur.detection_id,
            current_object_class=cur.object_class,
            current_confidence=cur.confidence,
            current_lat=cur.lat, current_lon=cur.lon,
            current_capture_time=current_capture_time,
            current_bbox={"x1": cur.bbox_x1, "y1": cur.bbox_y1,
                          "x2": cur.bbox_x2, "y2": cur.bbox_y2},
            status="new", source_type=source_type, session_id=session_id,
        ))

    # ------------------------------------------------------------------
    # 3. "disappeared"
    # ------------------------------------------------------------------
    for past in past_detections:
        if past.id in matched_past_ids:
            continue
        pairing_records.append(PairingRecord(
            pairing_time=now, lat_center=region_lat, lon_center=region_lon,
            past_detection_id=past.id,
            past_object_class=past.object_class,
            past_confidence=past.confidence,
            past_lat=past.lat, past_lon=past.lon,
            past_capture_time=past_capture_time,
            past_bbox={"x1": past.bbox_x1, "y1": past.bbox_y1,
                       "x2": past.bbox_x2, "y2": past.bbox_y2},
            status="disappeared", source_type=source_type, session_id=session_id,
        ))

    # ------------------------------------------------------------------
    # 4. Static-class cross-image check for unmatched "new" / "disappeared"
    # ------------------------------------------------------------------
    static_new_sim = [p for p in pairing_records
                      if p.status == "new"
                      and (p.current_object_class or "").lower() in _STATIC_CLASSES]
    static_dis_sim = [p for p in pairing_records
                      if p.status == "disappeared"
                      and (p.past_object_class or "").lower() in _STATIC_CLASSES]

    if static_new_sim or static_dis_sim:
        cur_img_cache2: Dict[str, Optional] = {}
        past_img_cache2: Dict[str, Optional] = {}
        past_image_ids2 = list({d.image_id for d in past_detections if d.image_id})
        cur_image_ids2  = list({d.detection_id and d.image_id
                                for d in current_detections if d.image_id})

        def _rec_pil(iid, cache):
            if iid not in cache:
                rec = get_image_record_by_id(iid)
                pil = _load_pil_for_image_id(iid) if rec else None
                cache[iid] = (rec, pil)
            return cache[iid]

        # "new" static: past side missing → synthesise in past image
        for pr in static_new_sim:
            if pr.current_detection_id is None:
                continue
            cur_det = next(
                (d for d in current_detections if d.detection_id == pr.current_detection_id),
                None,
            )
            if cur_det is None or not cur_det.image_id:
                continue
            cur_rec, cur_pil = _rec_pil(cur_det.image_id, cur_img_cache2)
            if cur_rec is None or cur_pil is None or not past_image_ids2:
                continue
            past_iid = past_image_ids2[0]
            past_rec, past_pil = _rec_pil(past_iid, past_img_cache2)
            if past_rec is None or past_pil is None:
                continue
            synth = _static_cross_check(
                det=cur_det,
                img_rec_with=cur_rec,
                pil_with=cur_pil,
                img_rec_without=past_rec,
                pil_without=past_pil,
                capture_time_without=past_capture_time or current_capture_time,
                session_id=session_id,
                source_type=source_type,
            )
            if synth is not None:
                pr.status = "matched"
                pr.past_detection_id = synth.id
                pr.past_object_class = synth.object_class
                pr.past_confidence = synth.confidence
                pr.past_lat = synth.lat
                pr.past_lon = synth.lon
                pr.past_capture_time = synth.detection_time
                pr.past_bbox = {
                    "x1": synth.bbox_x1, "y1": synth.bbox_y1,
                    "x2": synth.bbox_x2, "y2": synth.bbox_y2,
                }

        # "disappeared" static: current side missing → synthesise in current image
        for pr in static_dis_sim:
            if pr.past_detection_id is None:
                continue
            past_det = next(
                (d for d in past_detections if d.id == pr.past_detection_id),
                None,
            )
            if past_det is None or not past_det.image_id:
                continue
            past_rec, past_pil = _rec_pil(past_det.image_id, past_img_cache2)
            if past_rec is None or past_pil is None:
                continue
            # current image: get image_id from any current detection
            cur_iid2 = next(
                (d.image_id for d in current_detections if d.image_id), None
            )
            if cur_iid2 is None:
                continue
            cur_rec, cur_pil = _rec_pil(cur_iid2, cur_img_cache2)
            if cur_rec is None or cur_pil is None:
                continue
            synth = _static_cross_check(
                det=past_det,
                img_rec_with=past_rec,
                pil_with=past_pil,
                img_rec_without=cur_rec,
                pil_without=cur_pil,
                capture_time_without=current_capture_time,
                session_id=session_id,
                source_type=source_type,
            )
            if synth is not None:
                pr.status = "matched"
                pr.current_detection_id = synth.id
                pr.current_object_class = synth.object_class
                pr.current_confidence = synth.confidence
                pr.current_lat = synth.lat
                pr.current_lon = synth.lon
                pr.current_capture_time = synth.detection_time
                pr.current_bbox = {
                    "x1": synth.bbox_x1, "y1": synth.bbox_y1,
                    "x2": synth.bbox_x2, "y2": synth.bbox_y2,
                }

    logger.info(
        f"[Pairing/similarity] ({region_lat:.4f}, {region_lon:.4f}): "
        f"{sum(1 for p in pairing_records if p.status == 'matched')} matched  "
        f"{sum(1 for p in pairing_records if p.status == 'moved')} moved  "
        f"{sum(1 for p in pairing_records if p.status == 'new')} new  "
        f"{sum(1 for p in pairing_records if p.status == 'disappeared')} disappeared"
    )
    return pairing_records

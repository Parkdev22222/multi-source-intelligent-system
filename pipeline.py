"""
Multi-Source Intelligent System (MSIS) – Main Pipeline

Implements the Project Maven-inspired processing pipeline:

  ┌─────────────────────────────────────────────────────────────────┐
  │  SENSORS (Satellite / Drone cameras)                            │
  │  → Images with GPS coordinates & timestamps                     │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  INGESTION LAYER                                                │
  │  ImageLoader: reads image files + metadata.json index          │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  DETECTION LAYER (AI/ML)                                        │
  │  SAM3 (facebook/sam3) text-prompted concept segmentation        │
  │  → masks + bounding boxes + confidence scores in one pass       │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  SENSOR DB (SQLite)                                             │
  │  Tables: image_records, detection_records                       │
  │  Stores: object_id, class, confidence, lat/lon, time            │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  TEMPORAL PAIRING LAYER                                         │
  │  Matches current detections ↔ most-recent-past detections       │
  │  at same coordinates → status: new / matched / disappeared      │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  PAIRING DB (SQLite)                                            │
  │  Table: pairing_records                                         │
  └───────────────────────────┬─────────────────────────────────────┘
                              │
  ┌───────────────────────────▼─────────────────────────────────────┐
  │  REPORTING LAYER (EXAONE4-32b LLM)                              │
  │  Input:  latest pairing records                                 │
  │  Output: Military change-detection intelligence report          │
  └─────────────────────────────────────────────────────────────────┘
"""

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.config import IMAGES_DIR, COORDINATE_MATCH_RADIUS_DEG, TRACKING_MODE
from src.database.models import DetectionRecord
from src.database.sensor_db import (
    insert_image_record,
    insert_detections_bulk,
)
from src.database.pairing_db import (
    insert_pairings_bulk,
    get_latest_pairings,
    get_pairings_by_session,
)
from src.detection.image_loader import load_metadata_index, iter_images
from src.detection.sam2_detector import SAM3Detector, DetectionResult
from src.pairing.temporal_pairing import pair_by_tracking, pair_by_similarity
from src.reporting.military_reporter import MilitaryReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class MavenPipeline:
    """
    End-to-end Project Maven pipeline orchestrator.

    Usage:
        pipeline = MavenPipeline()
        report = pipeline.run(
            metadata_json="data/images/metadata.json",
            report_output_path="data/reports/latest_report.txt",
        )
        print(report)
    """

    def __init__(self):
        self.detector = SAM3Detector()
        self.reporter = MilitaryReporter()

    # ------------------------------------------------------------------
    # Step 1: Ingest + Detect
    # ------------------------------------------------------------------

    def _detect_and_store(self, metadata_json: str, session_id: str) -> List[str]:
        """
        Load all images from metadata index, run SAM2 detection,
        save image records and detection records to Sensor DB.

        Returns list of image_ids processed.
        """
        metas = sorted(load_metadata_index(metadata_json), key=lambda m: m.capture_time)
        image_ids = []

        for loaded in iter_images(metas):
            meta = loaded.meta

            # Insert image record into Sensor DB
            img_record = insert_image_record(
                capture_time=meta.capture_time,
                source_type=meta.source_type,
                image_path=meta.image_path,
                lat_center=meta.lat_center,
                lon_center=meta.lon_center,
                lat_min=meta.lat_min,
                lat_max=meta.lat_max,
                lon_min=meta.lon_min,
                lon_max=meta.lon_max,
                resolution_m=meta.resolution_m,
                sensor_platform=meta.sensor_platform,
            )
            image_id = img_record.id
            image_ids.append(image_id)

            # Run SAM2 detection
            det_results: List[DetectionResult] = self.detector.detect(loaded, image_id)

            if not det_results:
                logger.info(f"  No detections for image {image_id}. Skipping.")
                continue

            # Convert to ORM records
            orm_detections = [
                DetectionRecord(
                    id=det.detection_id,
                    image_id=image_id,
                    detection_time=meta.capture_time,  # capture_time from metadata.json (not wall clock)
                    object_class=det.object_class,
                    object_class_index=det.object_class_index,
                    confidence=det.confidence,
                    bbox_x1=det.bbox_x1,
                    bbox_y1=det.bbox_y1,
                    bbox_x2=det.bbox_x2,
                    bbox_y2=det.bbox_y2,
                    lat=det.lat,
                    lon=det.lon,
                    mask_rle=det.mask_rle,
                    mask_area_px=det.mask_area_px,
                    source_type=det.source_type,
                )
                for det in det_results
            ]

            # Bulk insert to Sensor DB
            insert_detections_bulk(orm_detections)
            logger.info(
                f"  [{img_record.id[:8]}] Stored {len(orm_detections)} detections "
                f"for image at ({meta.lat_center:.4f}, {meta.lon_center:.4f})"
            )

        return image_ids

    # ------------------------------------------------------------------
    # Step 2: Temporal Pairing
    # ------------------------------------------------------------------

    def _pair_and_store(
        self,
        metadata_json: str,
        session_id: str,
    ) -> int:
        """
        For each current-session image:
          1. Load image from disk (needed for Sam3Tracker).
          2. Fetch current detections from Sensor DB (stored by _detect_and_store).
          3. Fetch most-recent past detections from Sensor DB.
          4. Run Sam3Tracker on current image with past bboxes → TrackedObject list.
          5. pair_by_tracking() assigns status by object ID (matched/new/disappeared).
          6. Bulk-insert PairingRecord list into Pairing DB.

        Returns total number of pairing records inserted.
        """
        from PIL import Image as PILImage
        from src.database.sensor_db import (
            get_engine,
            get_most_recent_past_detections,
        )
        from src.database.models import ImageRecord, DetectionRecord as DR
        from sqlalchemy.orm import Session

        # Sort by capture_time ascending so older frames are always stored before
        # newer ones — ensuring get_most_recent_past_detections finds them correctly.
        metas = sorted(load_metadata_index(metadata_json), key=lambda m: m.capture_time)
        total_pairings = 0

        for loaded in iter_images(metas):
            meta = loaded.meta
            engine = get_engine()

            # --- Fetch current detections from Sensor DB ---
            with Session(engine) as sess:
                img_rec = (
                    sess.query(ImageRecord)
                    .filter(
                        ImageRecord.image_path == meta.image_path,
                        ImageRecord.capture_time == meta.capture_time,
                    )
                    .order_by(ImageRecord.ingestion_time.desc())
                    .first()
                )
                if img_rec is None:
                    continue
                image_id = img_rec.id
                current_orm = (
                    sess.query(DR).filter(DR.image_id == image_id).all()
                )
                current_dets = [
                    DetectionResult(
                        detection_id=d.id,
                        detection_time=d.detection_time,
                        image_id=d.image_id,
                        object_class=d.object_class,
                        object_class_index=d.object_class_index or 0,
                        confidence=d.confidence,
                        bbox_x1=d.bbox_x1, bbox_y1=d.bbox_y1,
                        bbox_x2=d.bbox_x2, bbox_y2=d.bbox_y2,
                        lat=d.lat, lon=d.lon,
                        source_type=d.source_type or meta.source_type,
                    )
                    for d in current_orm
                ]

            # --- Fetch past detections (returns records + past batch capture_time) ---
            past_records, past_capture_time = get_most_recent_past_detections(
                lat_center=meta.lat_center,
                lon_center=meta.lon_center,
                radius_deg=COORDINATE_MATCH_RADIUS_DEG,
                before_time=meta.capture_time,
            )

            # --- Build pairing records ---
            orig_h, orig_w = loaded.array.shape[:2]

            if TRACKING_MODE == "similarity":
                # Strategy B: SAM mask crop + CLIP embedding similarity
                pil_image = PILImage.fromarray(loaded.array).convert("RGB")
                pairing_records = pair_by_similarity(
                    current_detections=current_dets,
                    past_detections=past_records,
                    current_image=pil_image,
                    current_capture_time=meta.capture_time,
                    past_capture_time=past_capture_time,
                    region_lat=meta.lat_center,
                    region_lon=meta.lon_center,
                    session_id=session_id,
                    source_type=meta.source_type,
                )
            else:
                # Strategy A (default): SAM3 video tracker
                pil_image = PILImage.fromarray(loaded.array).convert("RGB")
                tracked_objects = (
                    self.detector.track_objects(pil_image, past_records, orig_w, orig_h)
                    if past_records else []
                )
                pairing_records = pair_by_tracking(
                    tracked_objects=tracked_objects,
                    current_detections=current_dets,
                    past_detections=past_records,
                    current_capture_time=meta.capture_time,
                    past_capture_time=past_capture_time,
                    region_lat=meta.lat_center,
                    region_lon=meta.lon_center,
                    session_id=session_id,
                    source_type=meta.source_type,
                )

            if pairing_records:
                insert_pairings_bulk(pairing_records)
                total_pairings += len(pairing_records)

        return total_pairings

    # ------------------------------------------------------------------
    # Step 3: Report Generation
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        session_id: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Retrieve latest pairings and generate the military report."""
        pairings = get_pairings_by_session(session_id)
        if not pairings:
            # Fall back to global latest
            pairings = get_latest_pairings()

        report = self.reporter.generate_report(
            pairings, output_path=output_path, session_id=session_id
        )
        return report

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        metadata_json: str,
        report_output_path: Optional[str] = None,
    ) -> str:
        """
        Execute the full pipeline end-to-end.

        Args:
            metadata_json:      Path to the image metadata index JSON.
            report_output_path: Optional file path to save the generated report.

        Returns:
            Military intelligence report as a string.
        """
        session_id = str(uuid.uuid4())
        logger.info(f"{'='*60}")
        logger.info(f"MSIS Pipeline started  session={session_id}")
        logger.info(f"{'='*60}")

        # Auto-generate sample data if metadata.json is empty
        metas_check = load_metadata_index(metadata_json)
        if not metas_check:
            logger.info(
                "[Pipeline] metadata.json has no entries – "
                "auto-generating synthetic sample data for testing."
            )
            from scripts.generate_sample_data import generate_sample_data
            generate_sample_data(metadata_json)

        # --- Step 1: Detection ---
        logger.info("[Pipeline] Step 1/3 – Image ingestion & SAM3 detection")
        image_ids = self._detect_and_store(metadata_json, session_id)
        logger.info(f"[Pipeline] Processed {len(image_ids)} images.")

        # --- Step 2: Pairing ---
        logger.info("[Pipeline] Step 2/3 – Temporal object pairing")
        n_pairings = self._pair_and_store(metadata_json, session_id)
        logger.info(f"[Pipeline] Created {n_pairings} pairing records.")

        # --- Step 3: Reporting ---
        logger.info("[Pipeline] Step 3/3 – Military report generation (EXAONE4-32b)")
        report = self._generate_report(session_id, report_output_path)

        logger.info(f"[Pipeline] Pipeline complete.  session={session_id}")
        return report

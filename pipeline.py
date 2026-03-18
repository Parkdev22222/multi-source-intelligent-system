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
  │  SAM2AutomaticMaskGenerator → segment proposals                 │
  │  CLIP zero-shot classifier  → object class + confidence         │
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

from src.config import IMAGES_DIR
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
from src.detection.sam2_detector import SAM2Detector, DetectionResult
from src.pairing.temporal_pairing import pair_detections
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
        self.detector = SAM2Detector()
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
        metas = load_metadata_index(metadata_json)
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
                    detection_time=det.detection_time,
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
        For each processed image in the current session, compute temporal
        pairings against the most recent past frame of the same region,
        and store results in the Pairing DB.

        Returns total number of pairing records inserted.
        """
        metas = load_metadata_index(metadata_json)
        total_pairings = 0

        for meta in metas:
            # Use meta.capture_time as the boundary for "past" detections
            from src.database.sensor_db import get_most_recent_past_detections
            # We need current detections from Sensor DB for this image
            # To avoid re-running detection, query the DB for recently inserted records
            from src.database.sensor_db import get_engine
            from src.database.models import ImageRecord, DetectionRecord as DR
            from sqlalchemy.orm import Session

            engine = get_engine()
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
                    sess.query(DR)
                    .filter(DR.image_id == image_id)
                    .all()
                )
                # Convert back to DetectionResult for pairing
                current_dets = [
                    DetectionResult(
                        detection_id=d.id,
                        detection_time=d.detection_time,
                        image_id=d.image_id,
                        object_class=d.object_class,
                        object_class_index=d.object_class_index or 0,
                        confidence=d.confidence,
                        bbox_x1=d.bbox_x1,
                        bbox_y1=d.bbox_y1,
                        bbox_x2=d.bbox_x2,
                        bbox_y2=d.bbox_y2,
                        lat=d.lat,
                        lon=d.lon,
                        source_type=d.source_type or meta.source_type,
                    )
                    for d in current_orm
                ]

            if not current_dets:
                continue

            pairing_records = pair_detections(
                current_detections=current_dets,
                current_capture_time=meta.capture_time,
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

        report = self.reporter.generate_report(pairings, output_path=output_path)
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

        # --- Step 1: Detection ---
        logger.info("[Pipeline] Step 1/3 – Image ingestion & SAM2 detection")
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

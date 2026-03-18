"""
SQLAlchemy ORM models for Sensor Detections DB and Object Pairings DB.

Sensor DB: stores every detected object from current-time satellite/drone imagery.
Pairing DB: stores temporal pairs (current detection ↔ past detection at same coordinates).
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    DateTime,
    JSON,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()
PairingBase = declarative_base()


# ---------------------------------------------------------------------------
# Sensor DB Models
# ---------------------------------------------------------------------------

class ImageRecord(Base):
    """Metadata for each ingested satellite/drone image."""

    __tablename__ = "image_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ingestion_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    capture_time = Column(DateTime, nullable=False)
    source_type = Column(String(32), nullable=False)   # "satellite" | "drone"
    image_path = Column(Text, nullable=False)
    # Geographic bounding box of the image
    lat_center = Column(Float, nullable=False)
    lon_center = Column(Float, nullable=False)
    lat_min = Column(Float, nullable=True)
    lat_max = Column(Float, nullable=True)
    lon_min = Column(Float, nullable=True)
    lon_max = Column(Float, nullable=True)
    resolution_m = Column(Float, nullable=True)        # GSD in metres/pixel
    sensor_platform = Column(String(64), nullable=True)  # e.g. "WorldView-3", "MQ-9"

    detections = relationship("DetectionRecord", back_populates="image", cascade="all, delete-orphan")


class DetectionRecord(Base):
    """
    A single object detected by SAM2 + CLIP classifier in a satellite/drone image.
    Corresponds to the sensor-layer output in the Project Maven pipeline.
    """

    __tablename__ = "detection_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String(36), ForeignKey("image_records.id"), nullable=False)
    detection_time = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Object classification
    object_class = Column(String(128), nullable=False)   # e.g. "military tank"
    object_class_index = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=False)

    # Pixel-space bounding box within image
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)

    # Geographic coordinates of the detected object (projected from image metadata)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    # SAM2 mask summary (run-length encoded or polygon as JSON string)
    mask_rle = Column(Text, nullable=True)
    mask_area_px = Column(Float, nullable=True)

    # Extra attributes
    source_type = Column(String(32), nullable=True)  # "satellite" | "drone"
    extra = Column(JSON, nullable=True)

    image = relationship("ImageRecord", back_populates="detections")


# ---------------------------------------------------------------------------
# Pairing DB Models
# ---------------------------------------------------------------------------

class PairingRecord(PairingBase):
    """
    A temporal pairing between a current detection and the most recent past
    detection at the same geographic region.

    Status values:
        "new"        – object appears for the first time (no past match)
        "matched"    – object present in both current and past frame
        "disappeared"– object was in the past frame but absent in current
    """

    __tablename__ = "pairing_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    pairing_time = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Region identification
    lat_center = Column(Float, nullable=False)
    lon_center = Column(Float, nullable=False)

    # Current-frame detection (always populated for "new" and "matched")
    current_detection_id = Column(String(36), nullable=True)
    current_object_class = Column(String(128), nullable=True)
    current_confidence = Column(Float, nullable=True)
    current_lat = Column(Float, nullable=True)
    current_lon = Column(Float, nullable=True)
    current_capture_time = Column(DateTime, nullable=True)
    current_bbox = Column(JSON, nullable=True)   # {"x1":…, "y1":…, "x2":…, "y2":…}

    # Past-frame detection (None for "new")
    past_detection_id = Column(String(36), nullable=True)
    past_object_class = Column(String(128), nullable=True)
    past_confidence = Column(Float, nullable=True)
    past_lat = Column(Float, nullable=True)
    past_lon = Column(Float, nullable=True)
    past_capture_time = Column(DateTime, nullable=True)
    past_bbox = Column(JSON, nullable=True)

    status = Column(String(32), nullable=False)  # "new" | "matched" | "disappeared"
    source_type = Column(String(32), nullable=True)

    # Analysis session tag (groups all pairs from one pipeline run)
    session_id = Column(String(36), nullable=True)


# ---------------------------------------------------------------------------
# DB factory helpers
# ---------------------------------------------------------------------------

def create_sensor_engine(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return engine


def create_pairing_engine(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    PairingBase.metadata.create_all(engine)
    return engine

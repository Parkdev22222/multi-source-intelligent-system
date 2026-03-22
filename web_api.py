"""
MSIS Web API – FastAPI backend for the satellite dashboard.

Endpoints:
    GET /api/satellites          – live satellite positions (CelesTrak + SGP4)
    GET /api/space/intelligence  – space activity briefing
    GET /api/detections          – SAM3 detections near a lat/lon region
    GET /api/image/latest        – latest ImageRecord metadata near a lat/lon
    GET /api/report/latest       – latest military report near a lat/lon
    GET /api/report/{id}         – single report by ID
    GET /dashboard               – serve the HTML dashboard

Run:
    uvicorn web_api:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import IMAGES_DIR, SENSOR_DB_PATH, PAIRING_DB_PATH, REPORTS_DB_PATH
from src.database.sensor_db import (
    get_latest_detections_near,
    get_latest_image_near,
)
from src.database.pairing_db import get_session_ids_near
from src.database.reports_db import (
    get_all_reports,
    get_report_by_id,
    get_latest_report_for_sessions,
)
from src.satellite.tle_tracker import get_satellite_positions, get_space_intelligence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MSIS Satellite Dashboard API",
    description="Multi-Source Intelligent System – satellite tracking & object detection UI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve satellite imagery files under /static/images/...
_images_dir = Path(IMAGES_DIR)
if _images_dir.exists():
    app.mount("/static/images", StaticFiles(directory=str(_images_dir)), name="images")

# Dashboard HTML
_dashboard_path = Path(__file__).parent / "dashboard" / "index.html"


# ---------------------------------------------------------------------------
# Satellite tracking
# ---------------------------------------------------------------------------

@app.get("/api/satellites")
def api_satellites(
    categories: str = Query(
        default="military,stations",
        description="Comma-separated CelesTrak groups: military, stations, starlink, gps-ops",
    )
):
    """Return current positions of satellites from CelesTrak."""
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    try:
        sats = get_satellite_positions(cats)
    except Exception as exc:
        logger.error(f"[API] satellite fetch error: {exc}")
        raise HTTPException(status_code=502, detail=f"CelesTrak fetch failed: {exc}")
    return {"satellites": sats, "count": len(sats)}


@app.get("/api/space/intelligence")
def api_space_intelligence():
    """Return space activity briefing (recent launches, military counts, signals)."""
    try:
        data = get_space_intelligence()
    except Exception as exc:
        logger.error(f"[API] space intelligence error: {exc}")
        raise HTTPException(status_code=502, detail=str(exc))
    return data


# ---------------------------------------------------------------------------
# SAM3 detections
# ---------------------------------------------------------------------------

@app.get("/api/detections")
def api_detections(
    lat: float = Query(..., description="Latitude of target region"),
    lon: float = Query(..., description="Longitude of target region"),
    radius: float = Query(default=0.1, description="Search radius in degrees"),
):
    """
    Return the latest SAM3 object detections from the DB for the region
    nearest to (lat, lon).
    """
    records = get_latest_detections_near(lat, lon, radius_deg=radius)
    detections = [
        {
            "id": r.id,
            "object_class": r.object_class,
            "confidence": round(r.confidence, 3),
            "lat": r.lat,
            "lon": r.lon,
            "bbox": {
                "x1": r.bbox_x1, "y1": r.bbox_y1,
                "x2": r.bbox_x2, "y2": r.bbox_y2,
            },
            "source_type": r.source_type,
            "detection_time": r.detection_time.isoformat() if r.detection_time else None,
            "image_id": r.image_id,
        }
        for r in records
    ]
    return {"detections": detections, "count": len(detections)}


# ---------------------------------------------------------------------------
# Latest satellite image near region
# ---------------------------------------------------------------------------

@app.get("/api/image/latest")
def api_latest_image(
    lat: float = Query(...),
    lon: float = Query(...),
    radius: float = Query(default=0.1),
    embed: bool = Query(default=True, description="Include base64-encoded image data"),
):
    """
    Return metadata for the most recent satellite/drone image captured
    near (lat, lon), optionally embedding the image as base64.
    """
    record = get_latest_image_near(lat, lon, radius_deg=radius)
    if record is None:
        raise HTTPException(status_code=404, detail="No imagery found for this region.")

    image_path = Path(record.image_path)
    # Resolve relative paths against IMAGES_DIR
    if not image_path.is_absolute():
        image_path = _images_dir / record.image_path

    image_b64 = None
    image_url = None
    if image_path.exists():
        # Build static URL
        rel = image_path.relative_to(_images_dir) if image_path.is_relative_to(_images_dir) else image_path.name
        image_url = f"/static/images/{rel}"
        if embed:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()

    return {
        "id": record.id,
        "capture_time": record.capture_time.isoformat() if record.capture_time else None,
        "source_type": record.source_type,
        "sensor_platform": record.sensor_platform,
        "lat_center": record.lat_center,
        "lon_center": record.lon_center,
        "lat_min": record.lat_min,
        "lat_max": record.lat_max,
        "lon_min": record.lon_min,
        "lon_max": record.lon_max,
        "resolution_m": record.resolution_m,
        "image_url": image_url,
        "image_b64": image_b64,
        "image_path": str(image_path),
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@app.get("/api/report/latest")
def api_latest_report(
    lat: float = Query(...),
    lon: float = Query(...),
    radius: float = Query(default=0.5),
):
    """
    Return the most recent military intelligence report whose session
    contains pairings near (lat, lon).
    Falls back to the globally latest report if no region match is found.
    """
    session_ids = get_session_ids_near(lat, lon, radius_deg=radius)
    report = get_latest_report_for_sessions(session_ids)

    if report is None:
        # Fall back to latest overall report
        all_reports = get_all_reports(limit=1)
        report = all_reports[0] if all_reports else None

    if report is None:
        raise HTTPException(status_code=404, detail="No reports found.")

    return _report_to_dict(report)


@app.get("/api/report/{report_id}")
def api_report_by_id(report_id: str):
    """Return a specific report by its UUID."""
    report = get_report_by_id(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return _report_to_dict(report)


@app.get("/api/reports")
def api_all_reports(limit: int = Query(default=50, le=200)):
    """List all reports (without content, sorted newest-first)."""
    reports = get_all_reports(limit=limit)
    return {
        "reports": [
            {
                "id": r.id,
                "report_time": r.report_time.isoformat() if r.report_time else None,
                "saved_time": r.saved_time.isoformat() if r.saved_time else None,
                "llm_model": r.llm_model,
                "pairing_count": r.pairing_count,
                "session_id": r.session_id,
            }
            for r in reports
        ],
        "count": len(reports),
    }


def _report_to_dict(r) -> dict:
    return {
        "id": r.id,
        "report_time": r.report_time.isoformat() if r.report_time else None,
        "saved_time": r.saved_time.isoformat() if r.saved_time else None,
        "llm_model": r.llm_model,
        "llm_backend": r.llm_backend,
        "pairing_count": r.pairing_count,
        "session_id": r.session_id,
        "file_path": r.file_path,
        "report_content": r.report_content,
    }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serve the satellite map dashboard HTML."""
    if not _dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    return HTMLResponse(content=_dashboard_path.read_text(encoding="utf-8"))


@app.get("/")
def root():
    return {"message": "MSIS API running. Visit /dashboard for the map UI."}

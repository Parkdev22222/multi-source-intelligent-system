"""
MSIS Web API – FastAPI 백엔드 (위성 모의기 + 탐지 결과 대시보드)

엔드포인트:
  GET  /api/satellites          – 위성 모의기 현재 위치 (케플러 원형궤도 계산)
  POST /api/simulator/step      – 1스텝 실행 (metadata 갱신 → main.py 파이프라인)
  GET  /api/detections          – SAM3 탐지 결과 (DB 최신)
  GET  /api/image/latest        – 최신 위성영상 메타데이터 + base64
  GET  /api/report/latest       – 최신 판독 보고서 (DB)
  GET  /api/report/{id}         – 보고서 단건 조회
  GET  /api/reports             – 보고서 목록
  GET  /dashboard               – 대시보드 HTML

실행:
  uvicorn web_api:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import IMAGES_DIR
from src.database.pairing_db import get_session_ids_near
from src.database.reports_db import (
    get_all_reports,
    get_latest_report_for_sessions,
    get_report_by_id,
)
from src.database.sensor_db import get_latest_detections_near, get_latest_image_near
from src.satellite.simulator import get_positions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MSIS Satellite Dashboard API",
    description="위성 모의기 + SAM3 객체탐지 + 판독보고서 대시보드",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 위성영상 정적 파일
_images_dir = Path(IMAGES_DIR)
if _images_dir.exists():
    app.mount("/static/images", StaticFiles(directory=str(_images_dir)), name="images")

_dashboard_path = Path(__file__).parent / "dashboard" / "index.html"


# ══════════════════════════════════════════════════════════════════════════
# 위성 모의기
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/satellites")
def api_satellites():
    """위성 모의기에서 계산된 현재 위성 위치 목록을 반환."""
    sats = get_positions()
    return {"satellites": sats, "count": len(sats)}


@app.post("/api/simulator/step")
def api_simulator_step(background_tasks: BackgroundTasks):
    """
    위성 모의기 1스텝 실행:
      1. 위성 궤도 계산 → 현재 위경도
      2. sample/ 에서 이미지 2장 랜덤 선택
      3. metadata.json 갱신
      4. main.py 파이프라인 실행 (탐지 + 페어링 + 보고서 → DB 삽입)
    """
    try:
        from simulator_runner import run_step
        result = run_step()
    except Exception as exc:
        logger.error(f"[API] simulator step error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "success":    result["success"],
        "elapsed_s":  result["elapsed_s"],
        "active":     result["active"],
        "satellites": result["satellites"],
        "images":     result["images"],
        "pipeline_ok": result["success"],
        "stderr_tail": result["stderr_tail"] if not result["success"] else "",
    }


# ══════════════════════════════════════════════════════════════════════════
# SAM3 탐지 결과
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/detections")
def api_detections(
    lat:    float = Query(..., description="위도"),
    lon:    float = Query(..., description="경도"),
    radius: float = Query(default=0.1, description="검색 반경 (도)"),
):
    """DB에서 해당 지역 최신 SAM3 탐지 결과 반환."""
    records = get_latest_detections_near(lat, lon, radius_deg=radius)
    return {
        "detections": [
            {
                "id":            r.id,
                "object_class":  r.object_class,
                "confidence":    round(r.confidence, 3),
                "lat":           r.lat,
                "lon":           r.lon,
                "bbox":          {"x1": r.bbox_x1, "y1": r.bbox_y1,
                                  "x2": r.bbox_x2, "y2": r.bbox_y2},
                "source_type":   r.source_type,
                "detection_time": r.detection_time.isoformat() if r.detection_time else None,
                "image_id":      r.image_id,
            }
            for r in records
        ],
        "count": len(records),
    }


# ══════════════════════════════════════════════════════════════════════════
# 최신 위성영상
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/image/latest")
def api_latest_image(
    lat:    float = Query(...),
    lon:    float = Query(...),
    radius: float = Query(default=0.1),
    embed:  bool  = Query(default=True, description="base64 이미지 데이터 포함 여부"),
):
    """해당 지역 최근 촬영 영상 메타데이터 + 선택적 base64 반환."""
    record = get_latest_image_near(lat, lon, radius_deg=radius)
    if record is None:
        raise HTTPException(status_code=404, detail="해당 지역 영상 없음.")

    image_path = Path(record.image_path)
    if not image_path.is_absolute():
        image_path = _images_dir / record.image_path

    image_b64 = None
    image_url  = None
    if image_path.exists():
        rel = image_path.relative_to(_images_dir) if image_path.is_relative_to(_images_dir) else image_path.name
        image_url = f"/static/images/{rel}"
        if embed:
            image_b64 = base64.b64encode(image_path.read_bytes()).decode()

    return {
        "id":              record.id,
        "capture_time":    record.capture_time.isoformat() if record.capture_time else None,
        "source_type":     record.source_type,
        "sensor_platform": record.sensor_platform,
        "lat_center":      record.lat_center,
        "lon_center":      record.lon_center,
        "resolution_m":    record.resolution_m,
        "image_url":       image_url,
        "image_b64":       image_b64,
    }


# ══════════════════════════════════════════════════════════════════════════
# 판독 보고서
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/report/latest")
def api_latest_report(
    lat:    float = Query(...),
    lon:    float = Query(...),
    radius: float = Query(default=2.0),
):
    """해당 지역 최신 판독 보고서 반환 (위치 매칭 → 전체 최신 순 폴백)."""
    session_ids = get_session_ids_near(lat, lon, radius_deg=radius)
    report = get_latest_report_for_sessions(session_ids)

    if report is None:
        all_r = get_all_reports(limit=1)
        report = all_r[0] if all_r else None

    if report is None:
        raise HTTPException(status_code=404, detail="보고서 없음.")

    return _report_dict(report)


@app.get("/api/report/{report_id}")
def api_report_by_id(report_id: str):
    report = get_report_by_id(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="보고서 없음.")
    return _report_dict(report)


@app.get("/api/reports")
def api_all_reports(limit: int = Query(default=50, le=200)):
    reports = get_all_reports(limit=limit)
    return {
        "reports": [
            {
                "id":           r.id,
                "report_time":  r.report_time.isoformat() if r.report_time else None,
                "saved_time":   r.saved_time.isoformat()  if r.saved_time  else None,
                "llm_model":    r.llm_model,
                "pairing_count":r.pairing_count,
                "session_id":   r.session_id,
            }
            for r in reports
        ],
        "count": len(reports),
    }


def _report_dict(r) -> dict:
    return {
        "id":             r.id,
        "report_time":    r.report_time.isoformat() if r.report_time else None,
        "saved_time":     r.saved_time.isoformat()  if r.saved_time  else None,
        "llm_model":      r.llm_model,
        "llm_backend":    r.llm_backend,
        "pairing_count":  r.pairing_count,
        "session_id":     r.session_id,
        "file_path":      r.file_path,
        "report_content": r.report_content,
    }


# ══════════════════════════════════════════════════════════════════════════
# 대시보드 HTML
# ══════════════════════════════════════════════════════════════════════════

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    if not _dashboard_path.exists():
        raise HTTPException(status_code=404, detail="dashboard/index.html 없음.")
    return HTMLResponse(content=_dashboard_path.read_text(encoding="utf-8"))


@app.get("/")
def root():
    return {"message": "MSIS API 실행 중. /dashboard 에서 지도 UI 확인."}

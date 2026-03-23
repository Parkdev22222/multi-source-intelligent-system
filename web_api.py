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

import asyncio
import base64
import io
import json
import logging
import queue
import threading
import time as _time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import IMAGES_DIR
from src.database.pairing_db import get_session_ids_near, get_session_location, get_pairings_by_session
from src.database.reports_db import (
    get_all_reports,
    get_latest_report_for_sessions,
    get_report_by_id,
    update_report_content,
)
from src.database.sensor_db import (
    get_latest_detections_near,
    get_latest_image_near,
    get_image_record_by_id,
    get_detection_by_id,
    get_detections_by_image,
    get_images_by_capture_time,
    get_all_images_with_count,
    replace_detections_for_image,
)
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

# ══════════════════════════════════════════════════════════════════════════
# Server-Sent Events (SSE) – 실시간 DB 업데이트 브로드캐스트
# ══════════════════════════════════════════════════════════════════════════

_sse_clients: list = []   # 연결된 클라이언트 Queue 목록
_sse_lock = threading.Lock()


def _notify_db_updated(run_count: int = 0, success: bool = True, elapsed: float = 0.0):
    """모든 SSE 구독 클라이언트에게 DB 업데이트 이벤트를 브로드캐스트한다."""
    payload = json.dumps({
        "type":      "db_updated",
        "run_count": run_count,
        "success":   success,
        "elapsed":   elapsed,
        "ts":        datetime.utcnow().isoformat(),
    })
    with _sse_lock:
        dead = []
        for q in list(_sse_clients):
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass


# 위성영상 정적 파일
_images_dir = Path(IMAGES_DIR)
if _images_dir.exists():
    app.mount("/static/images", StaticFiles(directory=str(_images_dir)), name="images")

# 대시보드 정적 파일 (Leaflet CSS/JS, GeoJSON 등 – 폐쇄망 로컬 서빙)
# 경로를 /lib 으로 분리 – /static/images 와 prefix 충돌 방지
_dashboard_static = Path(__file__).parent / "dashboard" / "lib"
_dashboard_static.mkdir(parents=True, exist_ok=True)
app.mount("/lib", StaticFiles(directory=str(_dashboard_static)), name="dashboard_static")

_dashboard_path = Path(__file__).parent / "dashboard" / "index.html"


# ══════════════════════════════════════════════════════════════════════════
# 자동 시뮬레이션 백그라운드 스레드
# ══════════════════════════════════════════════════════════════════════════

_auto_state: dict = {
    "enabled":        True,   # 자동 실행 ON/OFF
    "running":        False,  # 파이프라인 실행 중 여부
    "last_run":       None,   # 마지막 완료 시각 (ISO str)
    "run_count":      0,      # 누적 실행 횟수
    "last_success":   None,   # 마지막 실행 성공 여부
    "last_elapsed":   None,   # 마지막 실행 소요 시간 (s)
    "last_images":    [],     # 마지막 선택 이미지
    "last_active":    None,   # 마지막 활성 위성
    "last_positions": {},     # sat_id → (lat, lon)  위치 변경 감지용
}
_auto_lock = threading.Lock()

POSITION_CHANGE_THRESHOLD_DEG = 0.003   # ~300 m – 이 이상 이동하면 실행
AUTO_SIM_POLL_SEC              = 10      # 위치 체크 간격 (초)


def _any_satellite_moved(new_sats: list) -> bool:
    """이전 위치 대비 임계값 이상 이동한 위성이 있으면 True."""
    prev = _auto_state["last_positions"]
    if not prev:
        return True
    for s in new_sats:
        p = prev.get(s["id"])
        if p is None:
            return True
        if (abs(s["lat"] - p[0]) > POSITION_CHANGE_THRESHOLD_DEG or
                abs(s["lon"] - p[1]) > POSITION_CHANGE_THRESHOLD_DEG):
            return True
    return False


def _auto_sim_worker():
    """위성 좌표 변경 감지 → 시뮬레이션 자동 실행 백그라운드 스레드."""
    logger.info("[AutoSim] 백그라운드 스레드 시작 (%.1fs 간격)", AUTO_SIM_POLL_SEC)
    _time.sleep(3)   # 서버 초기화 대기

    while True:
        try:
            if _auto_state["enabled"] and not _auto_state["running"]:
                sats = get_positions()

                if _any_satellite_moved(sats):
                    # 위치 스냅샷 갱신
                    _auto_state["last_positions"] = {
                        s["id"]: (s["lat"], s["lon"]) for s in sats
                    }

                    with _auto_lock:
                        _auto_state["running"] = True

                    logger.info("[AutoSim] 좌표 변경 감지 → 파이프라인 시작 (실행 #%d)",
                                _auto_state["run_count"] + 1)
                    try:
                        from simulator_runner import run_step
                        result = run_step()
                        _auto_state["run_count"]    += 1
                        _auto_state["last_run"]      = datetime.utcnow().isoformat()
                        _auto_state["last_success"]  = result["success"]
                        _auto_state["last_elapsed"]  = result["elapsed_s"]
                        _auto_state["last_images"]   = result.get("images", [])
                        _auto_state["last_active"]   = result.get("active")
                        logger.info("[AutoSim] 완료 (#%d, %.1fs, success=%s)",
                                    _auto_state["run_count"],
                                    result["elapsed_s"],
                                    result["success"])
                        # 파이프라인 완료 → SSE 실시간 알림
                        _notify_db_updated(
                            run_count=_auto_state["run_count"],
                            success=result["success"],
                            elapsed=result["elapsed_s"],
                        )
                    except Exception as exc:
                        logger.error("[AutoSim] 실행 오류: %s", exc)
                        _notify_db_updated(run_count=_auto_state["run_count"], success=False)
                    finally:
                        with _auto_lock:
                            _auto_state["running"] = False

        except Exception as exc:
            logger.error("[AutoSim] 루프 오류: %s", exc)

        _time.sleep(AUTO_SIM_POLL_SEC)


# 서버 시작 시 스레드 자동 실행
threading.Thread(target=_auto_sim_worker, daemon=True, name="AutoSimThread").start()


def _image_to_png_b64(image_path: Path, max_size: int = 512) -> str:
    """이미지 파일(TIFF 포함)을 PNG로 변환 후 base64 반환."""
    from PIL import Image
    with Image.open(image_path) as img:
        img.thumbnail((max_size, max_size))
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


def _image_with_detections_b64(image_path: Path, detections: list, max_size: int = 480) -> str:
    """이미지에 탐지 결과 바운딩박스를 그린 뒤 base64 PNG 반환."""
    from PIL import Image, ImageDraw, ImageFont
    with Image.open(image_path) as img:
        orig_w, orig_h = img.size
        img.thumbnail((max_size, max_size))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        scale_x = img.width  / orig_w
        scale_y = img.height / orig_h
        draw = ImageDraw.Draw(img, "RGBA")
        for det in detections:
            x1 = det.bbox_x1 * scale_x
            y1 = det.bbox_y1 * scale_y
            x2 = det.bbox_x2 * scale_x
            y2 = det.bbox_y2 * scale_y
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 136, 220), width=2)
            label = f"{det.object_class} {det.confidence:.2f}"
            draw.rectangle([x1, y1 - 12, x1 + len(label) * 5.5, y1], fill=(0, 255, 136, 160))
            try:
                draw.text((x1 + 2, y1 - 12), label, fill=(0, 0, 0, 255))
            except Exception:
                pass
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════════════════════════════════
# 위성 모의기
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/satellites")
def api_satellites():
    """위성 모의기에서 계산된 현재 위성 위치 목록을 반환."""
    sats = get_positions()
    return {"satellites": sats, "count": len(sats)}


@app.get("/api/simulator/status")
def api_simulator_status():
    """자동 시뮬레이션 현재 상태 조회."""
    return {
        "auto_enabled": _auto_state["enabled"],
        "running":      _auto_state["running"],
        "run_count":    _auto_state["run_count"],
        "last_run":     _auto_state["last_run"],
        "last_success": _auto_state["last_success"],
        "last_elapsed": _auto_state["last_elapsed"],
        "last_images":  _auto_state["last_images"],
        "last_active":  _auto_state["last_active"],
    }


@app.post("/api/simulator/auto/toggle")
def api_simulator_auto_toggle():
    """자동 시뮬레이션 ON/OFF 토글."""
    _auto_state["enabled"] = not _auto_state["enabled"]
    logger.info("[AutoSim] 자동 실행 %s", "ON" if _auto_state["enabled"] else "OFF")
    return {"auto_enabled": _auto_state["enabled"]}


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

    # 수동 실행 완료 → SSE 실시간 알림
    _notify_db_updated(
        run_count=_auto_state["run_count"],
        success=result["success"],
        elapsed=result["elapsed_s"],
    )

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
            try:
                image_b64 = _image_to_png_b64(image_path)
            except Exception:
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
# DB 전체 이미지/탐지 목록
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/images")
def api_all_images(limit: int = Query(default=30, le=100)):
    """DB에 저장된 모든 위성영상 메타데이터 + 탐지 건수 목록 (capture_time DESC)."""
    rows = get_all_images_with_count(limit=limit)
    result = []
    for r, cnt in rows:
        result.append({
            "id":              r.id,
            "capture_time":    r.capture_time.isoformat() if r.capture_time else None,
            "source_type":     r.source_type,
            "sensor_platform": r.sensor_platform,
            "lat_center":      r.lat_center,
            "lon_center":      r.lon_center,
            "resolution_m":    r.resolution_m,
            "detection_count": cnt,
        })
    return {"images": result, "count": len(result)}


@app.get("/api/image/{image_id}/thumb")
def api_image_thumb(image_id: str):
    """특정 이미지를 PNG로 변환하여 base64 반환 (TIFF 포함)."""
    record = get_image_record_by_id(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="이미지 없음.")

    image_path = Path(record.image_path)
    if not image_path.is_absolute():
        image_path = _images_dir / record.image_path

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="이미지 파일 없음.")

    try:
        b64 = _image_to_png_b64(image_path, max_size=600)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"이미지 변환 실패: {exc}")

    return {
        "id":           record.id,
        "capture_time": record.capture_time.isoformat() if record.capture_time else None,
        "image_b64":    b64,
        "image_mime":   "image/png",
    }


@app.get("/api/detections/by-image/{image_id}")
def api_detections_by_image(image_id: str):
    """특정 이미지에 속한 탐지 결과 전체 반환."""
    records = get_detections_by_image(image_id)
    return {
        "detections": [
            {
                "id":             r.id,
                "object_class":   r.object_class,
                "confidence":     round(r.confidence, 3),
                "lat":            r.lat,
                "lon":            r.lon,
                "bbox":           {"x1": r.bbox_x1, "y1": r.bbox_y1,
                                   "x2": r.bbox_x2, "y2": r.bbox_y2},
                "source_type":    r.source_type,
                "detection_time": r.detection_time.isoformat() if r.detection_time else None,
            }
            for r in records
        ],
        "count": len(records),
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


@app.get("/api/report/{report_id}/images")
def api_report_images(report_id: str):
    """보고서 생성에 활용된 이전/현재 위성사진 base64 반환."""
    report = get_report_by_id(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="보고서 없음.")

    if not report.session_id:
        return {"current_images": [], "past_images": []}

    pairings = get_pairings_by_session(report.session_id)

    # ── 이미지 수집 전략 ─────────────────────────────────────────────────────
    # 1순위: detection_id → DetectionRecord.image_id 경로 (가장 정확)
    #        current_detection_id / past_detection_id 는 서로 다른 탐지이므로
    #        반드시 서로 다른 이미지를 가리킨다.
    # 2순위: detection이 없는 경우(새로운 객체 등) capture_time 으로 폴백.
    #        이때 capture_time 차이가 충분히 있는 경우에만 허용.
    current_image_id: str | None = None
    current_capture_time = None
    past_image_id: str | None = None
    past_capture_time = None

    for p in pairings:
        # ── current 이미지 확보 ──────────────────────────────────────────────
        if current_image_id is None:
            if p.current_detection_id:
                det = get_detection_by_id(p.current_detection_id)
                if det and det.image_id:
                    current_image_id = det.image_id
                    current_capture_time = p.current_capture_time
            # detection 경로 실패 시 capture_time 폴백
            if current_image_id is None and p.current_capture_time:
                imgs = get_images_by_capture_time(p.current_capture_time)
                if imgs:
                    current_image_id = imgs[0].id
                    current_capture_time = p.current_capture_time

        # ── past 이미지 확보 ─────────────────────────────────────────────────
        if past_image_id is None:
            if p.past_detection_id:
                det = get_detection_by_id(p.past_detection_id)
                if det and det.image_id and det.image_id != current_image_id:
                    past_image_id = det.image_id
                    past_capture_time = p.past_capture_time
            # detection 경로 실패 시 capture_time 폴백 (current와 다른 이미지만)
            if past_image_id is None and p.past_capture_time:
                imgs = get_images_by_capture_time(p.past_capture_time)
                for img in imgs:
                    if img.id != current_image_id:
                        past_image_id = img.id
                        past_capture_time = p.past_capture_time
                        break

        if current_image_id and past_image_id:
            break

    def _build_info(image_id, capture_time_fallback, with_detections: bool = False):
        rec = get_image_record_by_id(image_id)
        if rec is None:
            return None
        img_path = Path(rec.image_path)
        if not img_path.is_absolute():
            img_path = _images_dir / rec.image_path
        b64 = None
        if img_path.exists():
            try:
                if with_detections:
                    dets = get_detections_by_image(image_id)
                    b64 = _image_with_detections_b64(img_path, dets, max_size=480)
                else:
                    b64 = _image_to_png_b64(img_path, max_size=480)
            except Exception:
                b64 = None
        ct = rec.capture_time or capture_time_fallback
        return {
            "id":           rec.id,
            "capture_time": ct.isoformat() if ct else None,
            "image_b64":    b64,
        }

    curr_info = _build_info(current_image_id, current_capture_time, with_detections=True) \
                if current_image_id else None
    past_info = _build_info(past_image_id,    past_capture_time,    with_detections=True) \
                if past_image_id else None

    return {
        "current_images": [curr_info] if curr_info else [],
        "past_images":    [past_info] if past_info else [],
    }


# ── 사용자 수정용 엔드포인트 ──────────────────────────────────────────────────

class BboxIn(BaseModel):
    x1: float; y1: float; x2: float; y2: float

class DetectionIn(BaseModel):
    object_class: str
    confidence: float
    bbox: BboxIn
    lat: float = 0.0
    lon: float = 0.0

class DetectionsUpdateBody(BaseModel):
    detections: list[DetectionIn]

class ReportContentBody(BaseModel):
    report_content: str


@app.get("/api/image/{image_id}/raw")
def api_image_raw(image_id: str, max_size: int = Query(default=1024, le=2048)):
    """원본 이미지(탐지 박스 없음) + 실제 출력 크기 반환."""
    from PIL import Image as PilImage
    rec = get_image_record_by_id(image_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="이미지 없음")
    img_path = Path(rec.image_path)
    if not img_path.is_absolute():
        img_path = _images_dir / rec.image_path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="이미지 파일 없음")
    with PilImage.open(img_path) as img:
        orig_w, orig_h = img.size
        img.thumbnail((max_size, max_size))
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        out_w, out_h = img.size
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "id":           image_id,
        "image_b64":    b64,
        "width":        out_w,
        "height":       out_h,
        "orig_width":   orig_w,
        "orig_height":  orig_h,
        "capture_time": rec.capture_time.isoformat() if rec.capture_time else None,
    }


@app.get("/api/image/{image_id}/rendered")
def api_image_rendered(image_id: str, t: int = Query(default=0)):
    """현재 DB에 저장된 탐지 결과를 이미지에 그려서 반환 (캐시버스팅용 t 파라미터 지원)."""
    rec = get_image_record_by_id(image_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="이미지 없음")
    img_path = Path(rec.image_path)
    if not img_path.is_absolute():
        img_path = _images_dir / rec.image_path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="이미지 파일 없음")
    dets = get_detections_by_image(image_id)
    b64 = _image_with_detections_b64(img_path, dets, max_size=480)
    return {
        "id":           image_id,
        "image_b64":    b64,
        "capture_time": rec.capture_time.isoformat() if rec.capture_time else None,
    }


@app.put("/api/image/{image_id}/detections")
def api_update_detections(image_id: str, body: DetectionsUpdateBody):
    """이미지의 탐지 결과를 사용자 수정본으로 교체."""
    from src.database.models import DetectionRecord as DetRec
    rec = get_image_record_by_id(image_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="이미지 없음")
    new_dets = []
    for d in body.detections:
        new_dets.append(DetRec(
            image_id=image_id,
            detection_time=rec.capture_time,
            object_class=d.object_class,
            confidence=max(0.0, min(1.0, d.confidence)),
            bbox_x1=d.bbox.x1,
            bbox_y1=d.bbox.y1,
            bbox_x2=d.bbox.x2,
            bbox_y2=d.bbox.y2,
            lat=d.lat if d.lat != 0.0 else rec.lat_center,
            lon=d.lon if d.lon != 0.0 else rec.lon_center,
            source_type="human_edit",
        ))
    count = replace_detections_for_image(image_id, new_dets)
    return {"updated": count, "image_id": image_id}


@app.patch("/api/report/{report_id}")
def api_update_report(report_id: str, body: ReportContentBody):
    """보고서 텍스트를 수정한다."""
    ok = update_report_content(report_id, body.report_content)
    if not ok:
        raise HTTPException(status_code=404, detail="보고서 없음")
    return {"updated": True, "report_id": report_id}


@app.get("/api/reports")
def api_all_reports(limit: int = Query(default=50, le=200)):
    reports = get_all_reports(limit=limit)
    items = []
    for r in reports:
        lat, lon = (None, None)
        if r.session_id:
            lat, lon = get_session_location(r.session_id)
        items.append({
            "id":           r.id,
            "report_time":  r.report_time.isoformat() if r.report_time else None,
            "saved_time":   r.saved_time.isoformat()  if r.saved_time  else None,
            "llm_model":    r.llm_model,
            "pairing_count":r.pairing_count,
            "session_id":   r.session_id,
            "lat_center":   lat,
            "lon_center":   lon,
        })
    return {"reports": items, "count": len(items)}


# ══════════════════════════════════════════════════════════════════════════
# SSE 엔드포인트
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/events")
async def api_events():
    """
    Server-Sent Events 스트림.
    DB(이미지/탐지/보고서)가 갱신될 때 'db_updated' 이벤트를 즉시 전송한다.
    25초마다 heartbeat 코멘트를 전송해 연결을 유지한다.
    """
    q: queue.Queue = queue.Queue()
    with _sse_lock:
        _sse_clients.append(q)

    async def event_gen():
        try:
            yield "data: {\"type\":\"connected\"}\n\n"
            last_hb = asyncio.get_event_loop().time()
            while True:
                try:
                    msg = q.get_nowait()
                    yield f"data: {msg}\n\n"
                    last_hb = asyncio.get_event_loop().time()
                except queue.Empty:
                    await asyncio.sleep(0.3)
                    now = asyncio.get_event_loop().time()
                    if now - last_hb > 25:
                        yield ": heartbeat\n\n"
                        last_hb = now
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            with _sse_lock:
                try:
                    _sse_clients.remove(q)
                except ValueError:
                    pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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

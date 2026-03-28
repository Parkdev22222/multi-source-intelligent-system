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

import pycountry
import reverse_geocoder as _rg


def _get_country_name(lat: float, lon: float) -> Optional[str]:
    """위경도로 국가명(영문) 반환. 좌표 없으면 None."""
    try:
        cc = _rg.search((lat, lon))[0]["cc"]
        country = pycountry.countries.get(alpha_2=cc)
        return country.name if country else cc
    except Exception:
        return None

from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import IMAGES_DIR, SR_TARGET_W, SR_TARGET_H
from src.database.pairing_db import get_session_ids_near, get_session_location, get_pairings_by_session, update_pairings_detection_refs, get_pairings_near_time
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
    get_images_by_session,
    get_all_images_with_count,
    replace_detections_for_image,
    delete_detection_by_id,
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


def _notify_db_updated(run_count: int = 0, success: bool = True, elapsed: float = 0.0,
                       changed: list | None = None):
    """모든 SSE 구독 클라이언트에게 DB 업데이트 이벤트를 브로드캐스트한다.

    changed: 갱신된 데이터 종류 목록. 예: ["detections", "reports"]
             None이면 전체("images", "detections", "reports")로 간주.
    """
    payload = json.dumps({
        "type":      "db_updated",
        "run_count": run_count,
        "success":   success,
        "elapsed":   elapsed,
        "changed":   changed or ["images", "detections", "reports"],
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
                        # 바다 위 → 파이프라인 건너뜀 (run_count 미증가, SSE 미전송)
                        if result.get("skipped"):
                            logger.info("[AutoSim] 건너뜀 — %s",
                                        result.get("skip_reason", ""))
                        else:
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


# ══════════════════════════════════════════════════════════════════════════
# DB 폴링 스레드 – 파이프라인 실행 중 탐지 결과 즉시 알림
# ══════════════════════════════════════════════════════════════════════════

_DB_POLL_INTERVAL = 2   # 폴링 간격 (초)

def _db_poll_worker():
    """탐지/이미지 DB 행 수를 주기적으로 확인해 변화 시 SSE 알림."""
    import sqlite3
    from src.config import SENSOR_DB_PATH, REPORTS_DB_PATH

    def _row_counts():
        counts = {"images": 0, "detections": 0, "reports": 0}
        for db_path, queries in [
            (SENSOR_DB_PATH,  {"images": "SELECT COUNT(*) FROM image_records",
                               "detections": "SELECT COUNT(*) FROM detection_records"}),
            (REPORTS_DB_PATH, {"reports": "SELECT COUNT(*) FROM reports"}),
        ]:
            try:
                con = sqlite3.connect(db_path, timeout=2)
                for key, sql in queries.items():
                    try:
                        counts[key] = con.execute(sql).fetchone()[0]
                    except Exception:
                        pass
                con.close()
            except Exception:
                pass
        return counts

    prev = {"images": -1, "detections": -1, "reports": -1}

    while True:
        _time.sleep(_DB_POLL_INTERVAL)
        try:
            cur = _row_counts()
            changed = [k for k in ("images", "detections", "reports") if cur[k] != prev[k]]
            if changed:
                # 파이프라인이 돌고 있을 때만 중간 알림 전송
                # (파이프라인 완료 후 _auto_sim_worker 가 이미 전체 알림을 보내므로
                #  running=False 일 때는 중복 전송하지 않음)
                if _auto_state.get("running"):
                    _notify_db_updated(
                        run_count=_auto_state.get("run_count", 0),
                        success=True,
                        elapsed=0.0,
                        changed=changed,
                    )
                prev = cur
        except Exception as exc:
            logger.debug("[DBPoll] 오류: %s", exc)


# 서버 시작 시 스레드 자동 실행
threading.Thread(target=_auto_sim_worker, daemon=True, name="AutoSimThread").start()
threading.Thread(target=_db_poll_worker,  daemon=True, name="DBPollThread").start()


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


def _det_space(file_w: int, file_h: int,
               det_width: int | None, det_height: int | None) -> tuple[int, int]:
    """bbox 좌표 기준 공간 크기 반환.
    ImageRecord.det_width/det_height 가 있으면 그대로 사용;
    없으면 (구버전 데이터) SR 설정값으로 역산.
    """
    if det_width and det_height:
        return det_width, det_height
    sr_scale = min(SR_TARGET_W / file_w, SR_TARGET_H / file_h)
    if sr_scale > 1.0:
        return int(file_w * sr_scale), int(file_h * sr_scale)
    return file_w, file_h


def _image_with_detections_b64(
    image_path: Path,
    detections: list,
    max_size: int = 480,
    det_width: int | None = None,
    det_height: int | None = None,
) -> str:
    """이미지에 탐지 결과 바운딩박스를 그린 뒤 base64 PNG 반환."""
    from PIL import Image, ImageDraw, ImageFont
    with Image.open(image_path) as img:
        file_w, file_h = img.size
        det_w, det_h = _det_space(file_w, file_h, det_width, det_height)
        img.thumbnail((max_size, max_size))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        scale_x = img.width  / det_w
        scale_y = img.height / det_h
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
def api_simulator_step():
    """
    위성 모의기 1스텝 실행:
      1. 위성 궤도 계산 → 현재 위경도
      2. config.py IMAGE_MODE 에 따라 이미지 선택
         separate: sample/ 에서 서로 다른 이미지 2장
         crop:     sample/ 에서 1장 선택 후 크롭해 2장 생성
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
        "success":     result["success"],
        "skipped":     result.get("skipped", False),
        "skip_reason": result.get("skip_reason", ""),
        "elapsed_s":   result["elapsed_s"],
        "image_mode":  result.get("image_mode", body.image_mode),
        "active":      result["active"],
        "satellites":  result["satellites"],
        "images":      result["images"],
        "pipeline_ok": result["success"] and not result.get("skipped", False),
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
def api_all_images(limit: int = Query(default=None)):
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
            "session_id":      r.session_id,
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
    """보고서 생성에 활용된 이전/현재 위성사진 base64 반환.

    매핑 전략:
      현재 이미지 – ImageRecord.session_id == report.session_id (직접 매핑)
      과거 이미지 – PairingRecord.past_detection_id → DetectionRecord.image_id
                   (다른 세션에서 촬영된 과거 프레임)
      폴백        – session_id 없는 구 데이터는 pairing capture_time 경로 사용
    """
    report = get_report_by_id(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="보고서 없음.")

    from datetime import datetime as _dt
    _MIN_DT = _dt.min

    # session_id 기반 pairing 조회 → 없으면 report_time 기반 폴백
    pairings = get_pairings_by_session(report.session_id) if report.session_id else []
    if not pairings and report.report_time:
        pairings = get_pairings_near_time(report.report_time)
    # 보고서 텍스트와 동일하게 가장 최근 pairing_time 배치만 사용
    if pairings:
        latest_pt = max(p.pairing_time for p in pairings)
        pairings = [p for p in pairings if p.pairing_time == latest_pt]

    # ── 현재 이미지: session_id 직접 매핑 ─────────────────────────────────────
    curr_imgs = get_images_by_session(report.session_id) if report.session_id else []
    if curr_imgs:
        # capture_time 기준 내림차순 → 가장 최신 프레임이 "현재"
        curr_imgs = sorted(curr_imgs, key=lambda x: x.capture_time or _MIN_DT)
        current_image_id      = curr_imgs[-1].id
        current_capture_time  = curr_imgs[-1].capture_time
    else:
        # ── 폴백: 구 데이터 – pairing current_detection_id 경로 ──────────────
        curr_seen: dict[str, object] = {}  # image_id → capture_time
        for p in pairings:
            if p.current_detection_id:
                det = get_detection_by_id(p.current_detection_id)
                if det and det.image_id and det.image_id not in curr_seen:
                    curr_seen[det.image_id] = p.current_capture_time
        if not curr_seen:
            for p in pairings:
                if p.current_capture_time:
                    for img in get_images_by_capture_time(p.current_capture_time):
                        if img.id not in curr_seen:
                            curr_seen[img.id] = p.current_capture_time
        if curr_seen:
            entries = sorted(curr_seen.items(), key=lambda x: x[1] or _MIN_DT)
            current_image_id, current_capture_time = entries[-1]
        else:
            current_image_id = current_capture_time = None

    # ── 과거 이미지: pairing past_detection_id → image_id (다른 세션) ─────────
    past_seen: dict[str, object] = {}  # image_id → capture_time
    for p in pairings:
        if p.past_detection_id:
            det = get_detection_by_id(p.past_detection_id)
            if det and det.image_id and det.image_id not in past_seen:
                past_seen[det.image_id] = p.past_capture_time
    # 폴백: capture_time 경로
    if not past_seen:
        for p in pairings:
            if p.past_capture_time:
                for img in get_images_by_capture_time(p.past_capture_time):
                    if img.id not in past_seen:
                        past_seen[img.id] = p.past_capture_time
    if past_seen:
        entries = sorted(past_seen.items(), key=lambda x: x[1] or _MIN_DT)
        past_image_id, past_capture_time = entries[-1]
    else:
        past_image_id = past_capture_time = None

    # ── _build_info: image_id → base64(탐지 박스 포함) ───────────────────────
    def _build_info(image_id: str, capture_time_fallback, with_detections: bool = False):
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
                    b64 = _image_with_detections_b64(
                        img_path, dets, max_size=480,
                        det_width=rec.det_width, det_height=rec.det_height,
                    )
                else:
                    b64 = _image_to_png_b64(img_path, max_size=480)
            except Exception:
                b64 = None
        ct = rec.capture_time or capture_time_fallback
        return {
            "id":           rec.id,
            "capture_time": ct.isoformat() if ct else None,
            "image_b64":    b64,
            "session_id":   rec.session_id,
        }

    curr_info = _build_info(current_image_id, current_capture_time, with_detections=True) \
                if current_image_id else None
    past_info = _build_info(past_image_id,    past_capture_time,    with_detections=True) \
                if past_image_id else None

    return {
        "current_images": [curr_info] if curr_info else [],
        "past_images":    [past_info] if past_info else [],
        "session_id":     report.session_id,
    }


@app.get("/api/report/{report_id}/pairings")
def api_report_pairings(report_id: str):
    """보고서 세션의 페어링 목록 (bbox·클래스 포함) 반환 – 매칭 도시용.

    Returns:
        current_image_id: 현재 이미지 ID (raw 엔드포인트에서 원본 이미지 취득용)
        past_image_id:    과거 이미지 ID
        pairs:            페어링 목록 (status·label·bbox·class·conf)
    """
    report = get_report_by_id(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="보고서 없음.")
    from datetime import datetime as _dt
    _MIN_DT = _dt.min

    pairings = get_pairings_by_session(report.session_id) if report.session_id else []
    if not pairings and report.report_time:
        pairings = get_pairings_near_time(report.report_time)
    if pairings:
        latest_pt = max(p.pairing_time for p in pairings)
        pairings = [p for p in pairings if p.pairing_time == latest_pt]

    if not pairings:
        return {"current_image_id": None, "past_image_id": None, "pairs": []}

    # ── 현재 이미지 ID (api_report_images 와 동일 로직) ─────────────────────
    curr_imgs = get_images_by_session(report.session_id) if report.session_id else []
    if curr_imgs:
        curr_imgs = sorted(curr_imgs, key=lambda x: x.capture_time or _MIN_DT)
        current_image_id = curr_imgs[-1].id
    else:
        current_image_id = None
        for p in pairings:
            if p.current_detection_id:
                det = get_detection_by_id(p.current_detection_id)
                if det and det.image_id:
                    current_image_id = det.image_id
                    break

    # ── 과거 이미지 ID ───────────────────────────────────────────────────────
    past_image_id = None
    for p in pairings:
        if p.past_detection_id:
            det = get_detection_by_id(p.past_detection_id)
            if det and det.image_id:
                past_image_id = det.image_id
                break

    # ── 페어링 목록 구성 ─────────────────────────────────────────────────────
    counters: dict[str, int] = {"matched": 0, "moved": 0, "new": 0, "disappeared": 0}
    pairs = []
    for p in pairings:
        st = p.status if p.status in counters else "new"
        counters[st] += 1
        cnt = counters[st]
        if st == "matched":
            label = str(cnt)
        elif st == "moved":
            label = f"M{cnt}"
        elif st == "new":
            label = f"N{cnt}"
        else:
            label = f"D{cnt}"

        pairs.append({
            "id":            p.id,
            "status":        st,
            "label":         label,
            "current_bbox":  p.current_bbox,
            "current_class": p.current_object_class,
            "current_conf":  round(p.current_confidence, 3) if p.current_confidence else None,
            "past_bbox":     p.past_bbox,
            "past_class":    p.past_object_class,
            "past_conf":     round(p.past_confidence, 3) if p.past_confidence else None,
        })

    return {
        "current_image_id": current_image_id,
        "past_image_id":    past_image_id,
        "pairs":            pairs,
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
        file_w, file_h = img.size
        # orig_width/orig_height = bbox 좌표 기준 공간.
        # ImageRecord.det_width/det_height 가 있으면 직접 사용 (정확),
        # 없으면 SR 설정값으로 역산 (구버전 데이터 폴백).
        orig_w, orig_h = _det_space(file_w, file_h, rec.det_width, rec.det_height)
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
    b64 = _image_with_detections_b64(
        img_path, dets, max_size=480,
        det_width=rec.det_width, det_height=rec.det_height,
    )
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
            session_id=rec.session_id,   # 원본 이미지의 세션 ID 유지
        ))
    # 교체 전 구 detection_id 수집 → 교체 후 pairing 참조를 새 ID로 업데이트
    old_det_ids = {d.id for d in get_detections_by_image(image_id)}
    count = replace_detections_for_image(image_id, new_dets)
    new_det_list = get_detections_by_image(image_id)
    first_new_id = new_det_list[0].id if new_det_list else None
    update_pairings_detection_refs(old_det_ids, first_new_id)
    _notify_db_updated(changed=["detections", "images"])
    return {"updated": count, "image_id": image_id}


@app.post("/api/image/{image_id}/regenerate-report")
def api_regenerate_report(image_id: str, background_tasks: BackgroundTasks):
    """
    수정된 탐지 결과를 바탕으로 Temporal Pairing + 보고서 재생성.

    - SensorDB의 현재 탐지 결과(사용자 수정 포함)를 읽어 pairing 재실행
    - 동일 session_id의 기존 pairing_records / report_records를 교체
    - 새 보고서 내용과 report_id 반환
    """
    rec = get_image_record_by_id(image_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="이미지 없음")
    if not rec.session_id:
        raise HTTPException(status_code=400, detail="이미지에 session_id 없음")

    try:
        from pipeline import MavenPipeline
        pipeline = MavenPipeline()
        report_text = pipeline.rerun_from_detections(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"[API] regenerate-report error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    # 새로 삽입된 보고서 레코드 조회
    from src.database.reports_db import get_reports_by_session
    reports = get_reports_by_session(rec.session_id)
    new_report = reports[0] if reports else None

    _notify_db_updated(run_count=_auto_state["run_count"], success=True, elapsed=0.0)

    return {
        "success":    True,
        "image_id":   image_id,
        "session_id": rec.session_id,
        "report_id":  new_report.id if new_report else None,
        "report_content": report_text,
    }


@app.delete("/api/detection/{detection_id}")
def api_delete_detection(detection_id: str):
    """탐지 결과 단건 삭제 → SensorDB에서 제거하고 pairing 참조도 정리."""
    det = get_detection_by_id(detection_id)
    if det is None:
        raise HTTPException(status_code=404, detail="탐지 결과 없음")
    delete_detection_by_id(detection_id)
    # pairing이 이 detection을 참조하고 있으면 null로 초기화
    update_pairings_detection_refs({detection_id}, None)
    _notify_db_updated(changed=["detections", "images"])
    return {"deleted": True, "detection_id": detection_id}


@app.patch("/api/report/{report_id}")
def api_update_report(report_id: str, body: ReportContentBody):
    """보고서 텍스트를 수정한다."""
    ok = update_report_content(report_id, body.report_content)
    if not ok:
        raise HTTPException(status_code=404, detail="보고서 없음")
    _notify_db_updated(changed=["reports"])
    return {"updated": True, "report_id": report_id}


@app.get("/api/reports")
def api_all_reports(limit: int = Query(default=None)):
    reports = get_all_reports(limit=limit)
    items = []
    for r in reports:
        lat, lon = (None, None)
        if r.session_id:
            lat, lon = get_session_location(r.session_id)
        # session_id 없거나 pairing에 session_id 미설정 구 데이터 → 시간 기반 폴백
        if (lat is None or lon is None) and r.report_time:
            fb = get_pairings_near_time(r.report_time)
            if fb:
                lat = fb[0].lat_center
                lon = fb[0].lon_center
        country = _get_country_name(lat, lon) if lat is not None and lon is not None else None
        items.append({
            "id":           r.id,
            "report_time":  r.report_time.isoformat() if r.report_time else None,
            "saved_time":   r.saved_time.isoformat()  if r.saved_time  else None,
            "llm_model":    r.llm_model,
            "pairing_count":r.pairing_count,
            "session_id":   r.session_id,
            "lat_center":   lat,
            "lon_center":   lon,
            "country_name": country,
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

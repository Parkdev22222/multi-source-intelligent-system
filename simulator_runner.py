"""
MSIS 위성 모의기 실행기 (Simulator Runner)

동작 단계:
  1. 위성 궤도 계산 → 현재 위경도 (한반도 최근접 위성 선택)
  2. data/images/sample/ 에서 PNG 2장 랜덤 선택
  3. data/images/metadata.json 업데이트
       - imgs[0] → 과거 프레임  (capture_time = now − 6h)
       - imgs[1] → 현재 프레임  (capture_time = now)
       - lat/lon  = 위성 현재 위치
  4. python main.py --metadata ... --report-output ... 실행
  5. 실행 결과 dict 반환

독립 실행:
  python simulator_runner.py

FastAPI 에서 호출:
  POST /api/simulator/step  →  run_step()
"""

import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.satellite.simulator import get_active_satellite, get_positions

logger = logging.getLogger(__name__)

# ── 경로 ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
SAMPLE_DIR    = BASE_DIR / "data" / "images" / "sample"
METADATA_PATH = BASE_DIR / "data" / "images" / "metadata.json"
REPORT_OUTPUT = BASE_DIR / "data" / "reports" / "report.txt"

PIPELINE_CMD = [
    sys.executable, str(BASE_DIR / "main.py"),
    "--metadata",      str(METADATA_PATH),
    "--report-output", str(REPORT_OUTPUT),
]


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

def _pick_images(n: int = 2) -> list:
    """
    SAMPLE_DIR 에서 PNG 파일 n 장을 중복 없이 랜덤 선택.
    반환값은 metadata.json 에 쓰이는 상대 경로 'sample/<파일명>' 형식.
    """
    pngs = sorted(SAMPLE_DIR.glob("*.png"))
    if len(pngs) < n:
        raise RuntimeError(
            f"data/images/sample/ 에 PNG 가 {len(pngs)}장뿐입니다 (최소 {n}장 필요)."
        )
    chosen = random.sample(pngs, n)
    return [f"sample/{p.name}" for p in chosen]


def _build_metadata(sat: dict, imgs: list) -> list:
    """
    위성 위치와 선택된 이미지 2장으로 metadata.json 항목 2개 생성.
    imgs[0] = 과거 프레임, imgs[1] = 현재 프레임.
    """
    now  = datetime.now(timezone.utc)
    past = now - timedelta(hours=6)
    lat, lon = sat["lat"], sat["lon"]
    r = 0.01   # 촬영 범위 반경 (° ≈ 1.1 km)

    base = dict(
        source_type    = "satellite",
        lat_center     = lat,
        lon_center     = lon,
        lat_min        = lat - r,
        lat_max        = lat + r,
        lon_min        = lon - r,
        lon_max        = lon + r,
        resolution_m   = 0.5,
        sensor_platform= sat["name"],
        region_name    = sat["id"],
    )

    return [
        {**base, "image_file": imgs[0], "capture_time": past.isoformat()},
        {**base, "image_file": imgs[1], "capture_time": now.isoformat()},
    ]


# ── 공개 API ───────────────────────────────────────────────────────────────

def run_step() -> dict:
    """
    위성 모의기 1 스텝 실행.

    Returns:
        {
          success:     bool,
          elapsed_s:   float,
          satellites:  list[dict],   # 전체 위성 현재 위치
          active:      dict,         # 선택된 위성
          images:      list[str],    # 선택된 이미지 경로 (상대)
          stdout_tail: str,
          stderr_tail: str,
        }
    """
    t0 = time.time()

    # ── 1. 위성 위치 계산 ──────────────────────────────────────────────────
    satellites = get_positions()
    active     = get_active_satellite()
    logger.info(
        f"[SimRunner] 활성 위성: {active['name']}  "
        f"lat={active['lat']:.4f}  lon={active['lon']:.4f}  alt={active['alt_km']}km"
    )

    # ── 2. 이미지 선택 ────────────────────────────────────────────────────
    imgs = _pick_images(2)
    logger.info(f"[SimRunner] 선택된 이미지: {imgs}")

    # ── 3. metadata.json 업데이트 ─────────────────────────────────────────
    meta = _build_metadata(active, imgs)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[SimRunner] metadata.json 업데이트 완료")

    # ── 4. main.py 파이프라인 실행 ────────────────────────────────────────
    logger.info(f"[SimRunner] 파이프라인 실행: {' '.join(PIPELINE_CMD)}")
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        PIPELINE_CMD,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(BASE_DIR),
    )
    elapsed = time.time() - t0
    success = proc.returncode == 0

    if success:
        logger.info(f"[SimRunner] 파이프라인 완료 ({elapsed:.1f}s)")
    else:
        logger.error(f"[SimRunner] 파이프라인 오류 (rc={proc.returncode}):\n{proc.stderr[-1500:]}")

    return {
        "success":     success,
        "elapsed_s":   round(elapsed, 2),
        "satellites":  satellites,
        "active":      active,
        "images":      imgs,
        "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-500:]  if proc.stderr else "",
    }


# ── 독립 실행 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    result = run_step()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)

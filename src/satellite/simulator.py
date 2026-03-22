"""
위성 궤도 모의기 (Satellite Orbital Simulator)

CelesTrak / TLE 없이 케플러 원형궤도 공식으로 위성의 현재 위경도를 계산.

궤도 방정식 (원형궤도, 구면 근사):
  M  = ω·Δt + φ₀                              # 평균근점이각
  φ  = arcsin(sin(i)·sin(M))                  # 위위도 (sub-satellite latitude)
  λ  = RAAN + arctan2(cos(i)·sin(M), cos(M)) − ω_e·Δt  # 경도 (지구 자전 보정)

환경 변수:
  SIM_TIME_WARP  모의 가속 배율 (기본: 20x → 현실 1초 = 궤도 20초)
"""

import math
import os
import time
from datetime import datetime, timezone
from typing import List, Optional

# ── 상수 ──────────────────────────────────────────────────────────────────
EARTH_ROT_DEG_PER_SEC = 360.0 / 86164.1   # 지구 자전 속도 (도/초, 항성일 기준)
TIME_WARP = float(os.getenv("SIM_TIME_WARP", "20"))  # 가속 배율

# 기준 에포크: 2026-03-22T00:00:00 UTC
_EPOCH = 1742601600.0

# ── 모의 위성 정의 ─────────────────────────────────────────────────────────
# 한국 군사위성 계열 파라미터 (실제 태양동기궤도 기반)
SATELLITES: List[dict] = [
    {
        "id": "KMS-001", "name": "한국군위성-1", "country": "KOR",
        "inclination": 97.4,   # SSO (태양동기궤도)
        "period_s":    5640,   # ~94분 (고도 480km LEO)
        "raan_deg":    0.0,    # 승교점 적경
        "phase_deg":   0.0,    # 초기 위상 (평균근점이각)
        "alt_km":      480,
        "category":    "military",
    },
    {
        "id": "KMS-002", "name": "한국군위성-2", "country": "KOR",
        "inclination": 97.4,
        "period_s":    5640,
        "raan_deg":    72.0,
        "phase_deg":   72.0,
        "alt_km":      480,
        "category":    "military",
    },
    {
        "id": "KMS-003", "name": "한국군위성-3", "country": "KOR",
        "inclination": 98.2,
        "period_s":    5760,
        "raan_deg":    144.0,
        "phase_deg":   144.0,
        "alt_km":      510,
        "category":    "military",
    },
    {
        "id": "KMS-004", "name": "한국군위성-4", "country": "KOR",
        "inclination": 97.0,
        "period_s":    5580,
        "raan_deg":    216.0,
        "phase_deg":   216.0,
        "alt_km":      460,
        "category":    "military",
    },
    {
        "id": "KMS-005", "name": "한국군위성-5", "country": "KOR",
        "inclination": 97.6,
        "period_s":    5700,
        "raan_deg":    288.0,
        "phase_deg":   288.0,
        "alt_km":      490,
        "category":    "military",
    },
]


# ── 핵심 계산 함수 ─────────────────────────────────────────────────────────

def _compute_position(cfg: dict, t_real: float) -> dict:
    """
    실제 unix 시각 t_real 에서 위성의 위경도를 계산.
    TIME_WARP 배율을 적용해 궤도 시간으로 변환.

    Returns dict: id, name, country, category, lat, lon, alt_km, period_min, inclination
    """
    i   = math.radians(cfg["inclination"])
    T   = cfg["period_s"]
    w   = 2 * math.pi / T          # 각속도 (rad/s)
    phi = math.radians(cfg["phase_deg"])
    raan = math.radians(cfg["raan_deg"])

    # 궤도 경과 시간 (TIME_WARP 적용)
    dt = (t_real - _EPOCH) * TIME_WARP

    # 평균근점이각
    M = w * dt + phi

    # 위위도 φ = arcsin(sin(i)·sin(M))
    lat_rad = math.asin(math.sin(i) * math.sin(M))

    # 승교점에서의 경도 진행 + 지구 자전 빼기
    u = math.atan2(math.cos(i) * math.sin(M), math.cos(M))
    lon_deg = math.degrees(raan + u) - EARTH_ROT_DEG_PER_SEC * dt
    lon_deg = ((lon_deg + 180) % 360) - 180  # −180 ‥ 180 정규화

    return {
        "id":            cfg["id"],
        "name":          cfg["name"],
        "country":       cfg["country"],
        "category":      cfg["category"],
        "category_label": "군사위성",
        "period_min":    round(cfg["period_s"] / 60, 2),
        "inclination":   cfg["inclination"],
        "alt_km":        cfg["alt_km"],
        "lat":           round(math.degrees(lat_rad), 4),
        "lon":           round(lon_deg, 4),
    }


def get_positions(t_real: Optional[float] = None) -> List[dict]:
    """현재 시각(또는 지정 unix 시각) 기준 모든 시뮬레이션 위성 위치 반환."""
    if t_real is None:
        t_real = time.time()
    return [_compute_position(cfg, t_real) for cfg in SATELLITES]


def get_active_satellite(t_real: Optional[float] = None) -> dict:
    """
    한반도 기준점(37.5°N, 127°E)에 가장 가까운 위성 반환.
    metadata.json 의 촬영 좌표(lat_center/lon_center)로 사용됨.
    """
    if t_real is None:
        t_real = time.time()
    positions = get_positions(t_real)
    TARGET = (37.5, 127.0)
    return min(
        positions,
        key=lambda s: (s["lat"] - TARGET[0]) ** 2 + (s["lon"] - TARGET[1]) ** 2,
    )

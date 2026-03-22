#!/usr/bin/env python3
"""
폐쇄망 대응 – 지도 관련 정적 파일 다운로드 스크립트

인터넷이 연결된 환경에서 1회 실행하면
dashboard/static/ 에 필요한 파일 3개를 저장합니다.

실행:
    python scripts/download_map_assets.py
"""

import json
import sys
import urllib.request
from pathlib import Path

# __file__ 이 없는 환경(Jupyter 등)에서는 현재 디렉터리 기준으로 탐색
if "__file__" in dir():
    _base = Path(__file__).resolve().parent.parent
else:
    # cwd 가 프로젝트 루트이거나, 그 안의 하위 디렉터리일 때 모두 대응
    _cwd = Path.cwd()
    if (_cwd / "dashboard").exists():
        _base = _cwd
    elif (_cwd.parent / "dashboard").exists():
        _base = _cwd.parent
    else:
        raise RuntimeError(
            "dashboard/ 폴더를 찾을 수 없습니다. "
            "프로젝트 루트 또는 scripts/ 디렉터리에서 실행하세요."
        )

STATIC_DIR = _base / "dashboard" / "static"

ASSETS = [
    {
        "filename": "leaflet.min.css",
        "url": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.min.css",
        "desc": "Leaflet CSS (지도 라이브러리 스타일)",
    },
    {
        "filename": "leaflet.min.js",
        "url": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.min.js",
        "desc": "Leaflet JS (지도 라이브러리)",
    },
    {
        "filename": "world.geojson",
        "url": (
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
            "/master/geojson/ne_110m_admin_0_countries.geojson"
        ),
        "desc": "세계 국경 GeoJSON (Natural Earth 110m – 타일 서버 대체)",
    },
]


def download(url: str, dest: Path, desc: str) -> None:
    print(f"  ↓ {desc}")
    print(f"    {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "MSIS/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"    ✓ 저장 완료 → {dest}  ({len(data):,} bytes)\n")


def main() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"저장 위치: {STATIC_DIR}\n")

    errors = []
    for asset in ASSETS:
        dest = STATIC_DIR / asset["filename"]
        if dest.exists():
            print(f"  ✓ 이미 존재: {asset['filename']} ({dest.stat().st_size:,} bytes) – 건너뜀\n")
            continue
        try:
            download(asset["url"], dest, asset["desc"])
        except Exception as exc:
            print(f"  ✗ 실패: {asset['filename']} – {exc}\n")
            errors.append(asset["filename"])

    if errors:
        print(f"[경고] 다음 파일 다운로드 실패: {errors}")
        print("       수동으로 다운로드해 dashboard/static/ 에 저장하세요.")
        sys.exit(1)
    else:
        print("모든 파일 준비 완료. 이제 폐쇄망에서도 대시보드가 동작합니다.")


if __name__ == "__main__":
    main()

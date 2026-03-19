# Multi-Source Intelligent System (MSIS)

Project Maven-inspired aerial imagery intelligence pipeline that performs:
- **SAM3** (`facebook/sam3`) text-prompted concept segmentation of satellite/drone images
- **Temporal change detection** via object pairing between time points
- **Military intelligence report** generation using **EXAONE4-32b**

---

## Architecture

```
SENSORS (Satellite / Drone)
         │  imagery + GPS metadata
         ▼
INGESTION LAYER  ──►  ImageLoader (metadata.json index)
         │
         ▼
DETECTION LAYER  ──►  SAM3 (facebook/sam3) via HuggingFace Transformers
                       Text-prompted concept segmentation (single forward pass)
                       → object_class, confidence, bbox, mask_rle, lat/lon
         │
         ▼
SENSOR DB (SQLite)
  ├─ image_records       (image metadata)
  └─ detection_records   (object_id, class, confidence, lat, lon, bbox, mask_rle, time)
         │
         ▼
TEMPORAL PAIRING  ──►  current frame ↔ most-recent past frame
  ┌──────────────────────────────────────────────────────────────┐
  │  Mode A  sam3_tracker (default)                              │
  │    SAM3 video predictor tracks past bbox prompts → IoU match │
  │                                                              │
  │  Mode B  similarity                                          │
  │    1. SAM mask_rle로 배경 마스킹 → bbox crop (object-only)   │
  │    2. CLIP (openai/clip-vit-base-patch16) image embedding    │
  │    3. cosine similarity + geo proximity → greedy assignment  │
  └──────────────────────────────────────────────────────────────┘
                        status: new / matched / moved / disappeared
         │
         ▼
PAIRING DB (SQLite)
  └─ pairing_records     (current_det ↔ past_det, status, session_id)
         │
         ▼
REPORTING LAYER  ──►  EXAONE4-32b LLM
                       → Military change-detection intelligence report
```

---

## Temporal Pairing 전략

### Mode A: `sam3_tracker` (기본값)

SAM3 비디오 예측기를 이용한 ID-기반 추적.

```
과거 프레임 bbox + class → SAM3 video predictor → 현재 프레임 TrackedObject
                                                         ↓ IoU matching
                                              current DetectionResult
```

- 장점: 카메라 각도 변화·조명 변화에 강건
- 단점: SAM3 비디오 세션 오버헤드, GPU 메모리 추가 소모

### Mode B: `similarity` (SAM mask crop + CLIP embedding)

```
현재/과거 프레임 각 detection
    ├─ SAM mask_rle로 배경 마스킹 (없으면 bbox crop fallback)
    └─ CLIP 임베딩 → cosine similarity matrix (N × M)

score = SIMILARITY_CLIP_WEIGHT × clip_sim
      + (1 − SIMILARITY_CLIP_WEIGHT) × geo_score

geo_dist ≥ COORD_MATCH_RADIUS → 후보 제외 (hard filter)
greedy assignment (점수 내림차순)
```

- 장점: 비디오 세션 불필요, 프레임 간격이 길어도 사용 가능
- 단점: CLIP 모델 로드 (~400 MB), 동일 외형 객체 혼동 가능
- fallback: CLIP 로드 실패 시 geo-only scoring으로 자동 전환

---

## SAM3 vs SAM2 – Key Difference

| | SAM2 (old) | SAM3 (current) |
|---|---|---|
| Segmentation | Automatic mask generator (class-agnostic) | Text-prompted concept segmentation |
| Classification | Separate CLIP model (2-stage) | Built-in (1-stage, text-conditioned) |
| Output | Segments → CLIP classify each | All instances of named concept |
| Accuracy | SAM2 + CLIP zero-shot | ~75–80% human performance on SA-Co benchmark |
| HF Model | `facebook/sam2-hiera-large` | `facebook/sam3` |

---

## Quick Start

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

SAM3는 HuggingFace Transformers(≥ 4.48.0)에 포함되어 있어 별도 설치 불필요.

### 2. HuggingFace 로그인 (모델 다운로드용)

```bash
pip install huggingface_hub
huggingface-cli login
# HuggingFace 토큰 입력 (https://huggingface.co/settings/tokens)
```

### 3. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일에서 LLM_BACKEND, SAM3_DEVICE 등 설정
```

주요 환경 변수:

**SAM3 / 탐지**

| 변수 | 기본값 | 설명 |
|---|---|---|
| `SAM3_MODEL_NAME` | `facebook/sam3` | SAM3 HuggingFace 모델 ID |
| `SAM3_DEVICE` | `cuda` | 추론 장치 (`cuda` / `cpu`) |
| `DETECTION_CONFIDENCE` | `0.3` | 탐지 신뢰도 임계값 |

**Temporal Pairing**

| 변수 | 기본값 | 설명 |
|---|---|---|
| `TRACKING_MODE` | `sam3_tracker` | `sam3_tracker` \| `similarity` |
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch16` | similarity 모드에서 사용할 CLIP 모델 |
| `SIMILARITY_CLIP_WEIGHT` | `0.7` | CLIP cosine sim 가중치 (나머지는 geo proximity) |
| `COORD_MATCH_RADIUS` | `0.01` | 매칭 탐색 반경 (도 단위, ≈ 1 km) |
| `MOVE_DISTANCE_THRESHOLD` | `0.001` | matched/moved 구분 임계값 (도 단위, ≈ 100 m) |

**LLM / 보고서**

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LLM_BACKEND` | `huggingface` | LLM 백엔드 (`huggingface` / `ollama`) |
| `LLM_MODEL_NAME` | `LGAI-EXAONE/EXAONE-4.0-32B-Instruct` | EXAONE 모델 ID |
| `OLLAMA_MODEL` | `exaone4:32b` | Ollama 사용 시 모델명 |

### 4. 실행

#### 옵션 A — HuggingFace 백엔드 (EXAONE4-32b 직접 로드)

```bash
# 보고서 출력 디렉터리 생성
mkdir -p data/reports

# 합성 테스트 이미지 생성 + 전체 파이프라인 실행 + 보고서 저장
python main.py --generate-samples \
               --report-output data/reports/report.txt

# 실제 이미지 사용 시 (data/images/metadata.json 작성 후)
python main.py --metadata data/images/metadata.json \
               --report-output data/reports/report.txt
```

#### 옵션 B — Ollama 백엔드 (경량, GPU 메모리 절약)

```bash
# Ollama 설치 후 EXAONE 모델 다운로드
ollama pull exaone4:32b

# Ollama 백엔드로 실행
python main.py --generate-samples \
               --llm-backend ollama \
               --report-output data/reports/report.txt
```

### 5. CLI 옵션 전체 목록

```
python main.py [옵션]

  --metadata PATH        이미지 메타데이터 JSON 경로 (기본: data/images/metadata.json)
  --report-output PATH   보고서 저장 파일 경로 (미지정 시 stdout만 출력)
  --generate-samples     합성 위성/드론 테스트 이미지 생성 후 파이프라인 실행
  --llm-backend BACKEND  LLM 백엔드 선택: huggingface | ollama
```

---

## Image Metadata Format

Place satellite/drone image files and a `metadata.json` in `data/images/`:

```json
[
  {
    "image_file": "sample/region_alpha_current_20260318T120000.png",
    "capture_time": "2026-03-18T12:00:00+00:00",
    "source_type": "satellite",
    "lat_center": 37.5765,
    "lon_center": 126.9680,
    "lat_min": 37.5665,
    "lat_max": 37.5865,
    "lon_min": 126.9580,
    "lon_max": 126.9780,
    "resolution_m": 0.5,
    "sensor_platform": "WorldView-3"
  }
]
```

---

## Database Schema

### Sensor DB (`data/db/sensor_detections.db`)

| Table | Key Columns |
|---|---|
| `image_records` | `id`, `capture_time`, `source_type`, `lat_center`, `lon_center`, `sensor_platform` |
| `detection_records` | `id`, `image_id`, `detection_time`, `object_class`, `confidence`, `lat`, `lon`, `bbox_*`, `mask_rle` |

### Pairing DB (`data/db/object_pairings.db`)

| Table | Key Columns |
|---|---|
| `pairing_records` | `id`, `pairing_time`, `status` (new/matched/disappeared), `current_detection_id`, `past_detection_id`, `lat_center`, `lon_center`, `session_id` |

### Reports DB (`data/db/reports.db`)

| Table | Key Columns |
|---|---|
| `report_records` | `id`, `saved_time`, `report_time`, `session_id`, `llm_model`, `llm_backend`, `pairing_count`, `file_path`, `report_content` |

---

## Military Object Classes

SAM3 detects these classes via text-prompted concept segmentation:

```
military tank · armored personnel carrier · military truck · military jeep
fighter aircraft · helicopter · military ship · missile launcher · artillery
military building · radar installation · military personnel
supply depot · fuel storage · command post
civilian vehicle · civilian building · road · runway · unknown object
```

---

## 탐지 결과 시각화

파이프라인 실행 후 두 가지 도구로 결과를 확인할 수 있습니다.

---

### 1. DB 조회 (`view_detections.py`)

탐지 결과를 터미널 텍스트로 조회합니다.

```bash
# 최근 탐지 20건 (기본)
python view_detections.py

# 최근 50건
python view_detections.py --limit 50

# 특정 클래스 필터 (부분 일치)
python view_detections.py --class "military tank"

# 특정 이미지의 탐지 결과
python view_detections.py --image-id <uuid>

# 특정 세션의 페어링 결과
python view_detections.py --session <uuid>

# 클래스별 통계 요약 + 이미지별 탐지 수
python view_detections.py --summary

# 페어링 결과 조회
python view_detections.py --pairings

# 페어링 결과 + status 필터 (matched / new / disappeared)
python view_detections.py --pairings --status matched
python view_detections.py --pairings --status new
python view_detections.py --pairings --status disappeared
```

출력 예시:
```
══════════════════════════════════════════════════════════════════════════════════════════
  SAM3 탐지 결과  (sensor_detections.db  →  detection_records)
══════════════════════════════════════════════════════════════════════════════════════════
  총 5건 표시 (최신순)

  ID        탐지시각              클래스               신뢰도    위도          경도         BBox W×H
  ──────────────────────────────────────────────────────────────────────────────────────────
  a1b2c3d4  2026-03-18 12:00:00  military tank        0.872   37.5765    126.9680   120×80
  ...
```

---

### 2. 이미지 시각화 (`visualize_detections.py`)

DB의 탐지 결과를 원본 이미지 위에 렌더링하여 PNG로 저장합니다.
바운딩박스, 반투명 마스크 오버레이, 클래스·신뢰도 레이블을 그립니다.

```bash
# 최근 이미지 1장 시각화 (기본, detection_output/ 에 저장)
python visualize_detections.py

# 최근 이미지 5장 시각화
python visualize_detections.py --limit 5

# 특정 이미지 UUID 지정
python visualize_detections.py --image-id <uuid>

# 특정 클래스가 탐지된 이미지만 시각화
python visualize_detections.py --class "military tank"

# 마스크 오버레이 없이 bbox + 레이블만 그리기
python visualize_detections.py --no-mask

# 저장 디렉터리 지정
python visualize_detections.py --out-dir ./my_output

# 저장 후 기본 이미지 뷰어로 바로 열기
python visualize_detections.py --show

# 옵션 조합 예시
python visualize_detections.py --limit 3 --class "helicopter" --no-mask --out-dir ./results --show
```

출력 파일명 형식: `YYYYMMDD_HHMMSS_<이미지ID8자>_<탐지수>dets.png`

#### 렌더링 요소

| 요소 | 설명 |
|---|---|
| 반투명 마스크 | SAM3가 출력한 `mask_rle`를 복원해 클래스별 색상으로 오버레이 (투명도 35%) |
| 바운딩박스 | 3px 두께, 클래스별 고유 색상 |
| 레이블 | `클래스명  신뢰도` 텍스트 (배경 박스 포함) |

> `--no-mask` 옵션을 사용하면 mask_rle 없이 bbox + 레이블만 렌더링하므로 더 빠릅니다.

---

### 클래스별 색상 팔레트

| 인덱스 | 색상 |
|---|---|
| 0 | 빨강 |
| 1 | 파랑 |
| 2 | 초록 |
| 3 | 주황 |
| 4 | 보라 |
| 5 | 청록 |
| 6 | 핑크 |
| 7 | 노랑 |
| 8 | 연주황 |
| 9 | 연파랑 |

---

## Notes

- **SAM3 resolution**: SAM3 requires a fixed inference resolution of 1008×1008.
  Images are automatically resized by `Sam3Processor` before inference.
- **EXAONE4-32b**: Uses LG AI Research's EXAONE 4.0 32B Instruct model.
  Falls back to Ollama if HuggingFace backend is unavailable.
- **Fallback mode**: If SAM3 weights are unavailable (no GPU / offline),
  the system uses a grid-based pseudo-detector for development/testing.
- **CLIP similarity mode**: `TRACKING_MODE=similarity` 설정 시
  `openai/clip-vit-base-patch16` (~400 MB)가 첫 실행에 HuggingFace Hub에서
  자동 다운로드됩니다. CLIP 로드에 실패하면 geo proximity 점수만으로 매칭하며
  파이프라인은 중단 없이 계속 실행됩니다.

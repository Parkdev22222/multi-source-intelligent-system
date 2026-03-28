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

## 보고서 생성 과정 상세 (현재 config.py 설정 기준)

현재 `config.py` 설정을 기준으로 `python main.py` 실행 시 보고서가 생성되기까지의 전 과정을 단계별로 설명합니다.

---

### 전체 흐름 요약

```
metadata.json 읽기
      │
      ▼
[Step 1] 이미지 수집 + 초해상도 (Real-ESRGAN / PIL LANCZOS) → 8000×6000px
      │
      ▼
[Step 1] SAM3 탐지 – 멀티스케일 + 슬라이딩 윈도우 타일
  ├─ 전체 이미지 탐지  (큰 객체용, TILE_MULTISCALE=true)
  └─ 타일 탐지         (작은 객체용, 1008px 타일, 200px 겹침)
      │  NMS (IoU=0.3) 중복 제거
      ▼
[Step 1] Sensor DB 저장 (image_records + detection_records)
      │
      ▼
[Step 2] Temporal Pairing – similarity 모드
  ├─ CLIP ViT-Large 임베딩 (현재/과거 각 객체 crop)
  ├─ score = 0.8 × CLIP cosine sim + 0.2 × 크기 유사도
  └─ Gale-Shapley 할당 → new / matched / moved / disappeared
      │
      ▼
[Step 2] Pairing DB 저장 (pairing_records)
      │
      ▼
[Step 3] LLM 보고서 생성 (EXAONE-3.5-7.8B-Instruct-AWQ, vLLM)
  ├─ 영어 보고서 생성  (new + disappeared 객체 중심)
  └─ 한국어 번역       (동일 모델 재사용)
      │
      ▼
[Step 3] Reports DB 저장 + (선택) 파일 저장
```

---

### Step 1-A: 이미지 수집 및 초해상도 (Super Resolution)

**설정값**: `SR_TARGET_W=8000`, `SR_TARGET_H=6000`

1. `metadata.json`을 읽어 `capture_time` 오름차순으로 정렬합니다.
2. 각 이미지 파일을 로드한 뒤 **초해상도(SR) 업스케일**을 적용합니다.
   - 목표 해상도: **8000×6000 px** (종횡비 유지, 이미 목표 크기 이상이면 건너뜀)
   - 필요 배율 > 2× → `Real-ESRGAN x4` 모델 사용
   - 필요 배율 ≤ 2× → `Real-ESRGAN x2` 모델 사용
   - `basicsr` / `realesrgan` 패키지 미설치 시 → **PIL LANCZOS 폴백**
3. SR 이후 실제 배열 크기가 Sensor DB의 `det_width` / `det_height`로 기록됩니다.
   이후 탐지 결과의 bbox 좌표는 이 SR 적용 후 해상도를 기준으로 합니다.

---

### Step 1-B: SAM3 탐지 (멀티스케일 슬라이딩 윈도우)

**설정값**: `SAM3_MODEL_NAME="/content/drive/MyDrive/sam3"`, `SAM3_DEVICE=cuda`
**타일 설정**: `TILE_ENABLED=true`, `TILE_SIZE=1008`, `TILE_OVERLAP=200`, `TILE_MULTISCALE=true`, `TILE_NMS_IOU=0.3`

SAM3는 텍스트 프롬프트를 받아 해당 개념의 모든 인스턴스를 한 번의 순전파(forward pass)로 세그멘테이션합니다. SAM3의 고정 입력 해상도는 **1008×1008 px**입니다.

#### 탐지 방식: 멀티스케일 + 슬라이딩 윈도우

```
SR 이후 이미지 (최대 8000×6000px)
      │
      ├─── [전체 이미지 탐지] ──────────────────────────────── TILE_MULTISCALE=true
      │    이미지 전체를 1008×1008로 리사이즈 → SAM3 추론
      │    → 큰 객체(탱크, 건물, 활주로 등) 탐지에 유리
      │
      └─── [타일 탐지] ─────────────────────────────────────── TILE_ENABLED=true
           stride = TILE_SIZE - TILE_OVERLAP = 808px
           ┌───────────────────────────────────────────────┐
           │  타일 분할 (1008×1008px, 200px 겹침)           │
           │  각 타일을 1008×1008로 리사이즈 → SAM3 추론    │
           │  bbox, lat/lon 좌표를 원본 이미지 기준으로 역변환│
           └───────────────────────────────────────────────┘
           → 작은 객체(차량, 미사일 발사대, 인원 등) 탐지에 유리
                    │
                    ▼
           전체 이미지 탐지 결과 + 타일 탐지 결과 병합
                    │
                    ▼
           NMS (IoU 임계값 = 0.3) 중복 제거
```

#### 탐지 대상 클래스 (20개, 텍스트 프롬프트 방식)

| 군사 장비 | 군사 시설 | 민간/기타 |
|---|---|---|
| military tank | military building | civilian vehicle |
| armored personnel carrier | radar installation | civilian building |
| military truck | supply depot | road |
| military jeep | fuel storage | runway |
| fighter aircraft | command post | unknown object |
| helicopter | military personnel | |
| military ship | | |
| missile launcher | | |
| artillery | | |

#### 탐지 후처리 필터

| 설정 | 값 | 설명 |
|---|---|---|
| `DETECTION_CONFIDENCE_THRESHOLD` | 0.3 | 신뢰도 0.3 미만 탐지 제거 |
| `NMS_IOU_THRESHOLD` | 0.3 | IoU ≥ 0.3이면 낮은 신뢰도 bbox 제거 |
| `MAX_BBOX_AREA_RATIO` | 0.15 | 이미지 전체 면적의 15% 초과 bbox 제거 |
| `SAM3_MASK_SCORE_THRESHOLD` | 0.5 | 마스크 신뢰도 0.5 미만 제거 |

#### 위경도 변환

각 탐지 객체의 bbox 중심 픽셀 좌표를 `metadata.json`의 `lat_min/max`, `lon_min/max` 값을 이용한 선형 보간으로 위경도로 변환합니다.

```
lat = lat_max - (bbox_cy / img_h) × (lat_max - lat_min)
lon = lon_min + (bbox_cx / img_w) × (lon_max - lon_min)
```

탐지 결과는 `detection_records` 테이블에 저장됩니다:
- `object_class`, `confidence`, `bbox_x1/y1/x2/y2`, `lat`, `lon`, `mask_rle`, `mask_area_px`

---

### Step 2: Temporal Pairing (similarity 모드)

**설정값**: `TRACKING_MODE=similarity`, `CLIP_MODEL_NAME="/content/drive/MyDrive/.../vit-large-patch16-224"`
`SIMILARITY_CLIP_WEIGHT=0.7`, `SIMILARITY_SIZE_WEIGHT=0.2`, `SIMILARITY_MATCH_THRESHOLD=0.5`
`COORDINATE_MATCH_RADIUS_DEG=0.01` (≈ 1.1 km), `MOVE_DISTANCE_THRESHOLD_DEG=0.001` (≈ 111 m)

현재 프레임의 탐지 결과와 직전 프레임(같은 지역, `capture_time` 기준 가장 최근 과거)의 탐지 결과를 매칭하여 변화를 분류합니다.

#### 2-1. CLIP 임베딩 생성

```
현재 탐지 객체 N개          과거 탐지 객체 M개
       │                           │
       ▼                           ▼
  SAM mask_rle 복원           과거 이미지 파일 로드 (+ SR 재적용)
  배경 픽셀 zeroing            SAM mask_rle 복원 → 배경 zeroing
  bbox crop (4px padding)      bbox crop
       │                           │
       └──────────┬────────────────┘
                  ▼
      CLIP ViT-Large-patch16-224 임베딩
      L2 정규화 → float32 (N×D), (M×D)
```

CLIP 로드 실패 시 geo proximity 점수만으로 매칭 (파이프라인 중단 없음).

#### 2-2. 유사도 점수 계산

클래스가 다른 쌍은 후보에서 제외되며, 같은 클래스 쌍에 대해:

```
CLIP cosine similarity  =  cur_embed · past_embed    (∈ [-1, 1])
size_similarity         =  min(area_cur, area_past) / max(area_cur, area_past)
                           (mask_area_px 우선, 없으면 bbox 면적)

최종 score = 0.8 × CLIP_cosine + 0.2 × size_similarity
```

> **geo distance는 CLIP 실패 시 대체 점수로만 사용됩니다.** CLIP이 정상 동작하면 공간 거리는 점수에 반영되지 않습니다.

#### 2-3. 최적 매칭 (Gale-Shapley 알고리즘)

- 점수 > `SIMILARITY_MATCH_THRESHOLD` (0.5) 인 쌍만 후보로 등록
- Gale-Shapley 지연 승인(deferred-acceptance) 방식으로 1:1 최적 매칭
- 매칭된 쌍 중 geo distance > `MOVE_DISTANCE_THRESHOLD_DEG` (0.001°, ≈ 111 m) 이면 `"moved"`, 아니면 `"matched"`

#### 2-4. 상태 분류

| 상태 | 조건 | 의미 |
|---|---|---|
| `new` | 현재에만 있고 매칭 안 됨 | 새로 출현한 객체 |
| `matched` | 현재 ↔ 과거 매칭, 이동 없음 | 정지 객체 |
| `moved` | 현재 ↔ 과거 매칭, 위치 변화 > 111 m | 이동한 객체 |
| `disappeared` | 과거에만 있고 매칭 안 됨 | 현재 영상에서 사라진 객체 |

결과는 `pairing_records` 테이블에 저장됩니다.

---

### Step 3-A: LLM 보고서 생성 (EXAONE via vLLM)

**설정값**: `LLM_BACKEND=vllm`, `LLM_MODEL_NAME=".../EXAONE-3.5-7.8B-Instruct-AWQ"`
`LLM_TEMPERATURE=0.2`, `LLM_MAX_NEW_TOKENS=2048`, `LLM_GPU_MEMORY_UTILIZATION=0.85`

#### 프롬프트 구성

같은 세션의 pairing_records 중 **가장 최근 `pairing_time` 배치**만 선택합니다 (동일 파이프라인에서 여러 이미지를 처리할 때 이전 프레임 결과가 섞이지 않도록).

LLM에게 전달되는 컨텍스트:

```
PAST_OBS: <과거 촬영 시각>   CURRENT_OBS: <현재 촬영 시각>   ROI: <위경도 중심>
CURRENT_FRAME_DETECTIONS: N  (NEW:n1  STATIONARY:n2  MOVED:n3)
PAST_ONLY (disappeared): n4
NOTE: Report covers only NEW and DISAPPEARED objects.

=== NEW OBJECTS ===
  military tank CONF=0.87 (37.576,126.968) DETECTED=2026-03-18T12:00:00Z
  ...

=== DISAPPEARED OBJECTS ===
  fighter aircraft CONF=0.79 (37.571,126.962) LAST_SEEN=2026-03-17T08:00:00Z
  ...

=== TASK ===
Write a military intelligence report ...
Sections: 1.CLASSIFICATION 2.EXECUTIVE SUMMARY 3.SITUATION 4.CHANGE ANALYSIS
          5.THREAT ASSESSMENT 6.INTELLIGENCE GAPS 7.RECOMMENDED ACTIONS 8.APPENDIX
```

> `matched` (정지) 및 `moved` (이동) 객체는 보고서에서 **제외**됩니다. 활동 지표인 신규 출현과 소실 객체만 분석합니다.

#### vLLM 추론 파라미터

| 파라미터 | 값 |
|---|---|
| quantization | AWQ |
| dtype | float16 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 4096 |
| temperature | 0.2 |
| max_new_tokens | 2048 |

---

### Step 3-B: 한국어 번역

**설정값**: `LLM_TRANSLATE_TO_KOREAN=true`, `LLM_TRANSLATE_MAX_TOKENS=4096`

영어 보고서가 생성된 직후, **동일한 EXAONE 모델**을 재사용하여 한국어로 번역합니다.

번역 규칙:
- 섹션 헤더(1. CLASSIFICATION, 2. EXECUTIVE SUMMARY 등)는 그대로 유지
- 좌표, 타임스탬프, 신뢰도 점수, 객체 클래스명(TANK, APC 등)은 번역하지 않음
- 서술·분석·설명 텍스트만 번역
- 번역 실패 시 영어 원문 반환 (파이프라인 중단 없음)

---

### Step 3-C: 보고서 헤더 및 저장

번역 완료 후 다음 메타데이터 헤더가 앞에 붙습니다:

```
════════════════════════════════════════════════════════════════════════
  MILITARY INTELLIGENCE REPORT
  Generated by: Multi-Source Intelligent System (MSIS)
  Model: .../EXAONE-3.5-7.8B-Instruct-AWQ
  Language: 한국어 (EXAONE 번역)
  Past observation:    2026-03-17T08:00:00Z
  Current observation: 2026-03-18T12:00:00Z
  Report generated:    2026-03-18T12:05:00Z
  Current frame detections: 12  (new=5 / stationary=4 / moved=3)
  Disappeared (past only):  7
  Total pairing records:    19
════════════════════════════════════════════════════════════════════════
```

보고서는 다음 두 위치에 저장됩니다:

| 저장 위치 | 내용 |
|---|---|
| `data/db/reports.db` → `report_records` 테이블 | 전체 보고서 텍스트 + 메타데이터 (session_id, 모델명, pairing_count 등) |
| `--report-output` 경로 (지정 시) | 동일 텍스트를 파일로 저장 |

---

### GPU 메모리 사용 패턴

현재 설정에서 CUDA 장치는 다음 순서로 모델을 로드합니다:

```
[Step 1] SAM3 (/content/drive/MyDrive/sam3) 로드 → CUDA
         ↓ 탐지 완료 후 메모리 유지 (동일 세션에서 재사용)

[Step 2] CLIP ViT-Large-patch16-224 로드 → (CPU 또는 CUDA)
         ↓ 임베딩 완료 후 캐시 유지

[Step 3] EXAONE-3.5-7.8B-Instruct-AWQ → vLLM (GPU 85% 사용)
         ↓ AWQ 4-bit 양자화로 메모리 절감
         ↓ 영어 보고서 생성 + 한국어 번역 (모델 재사용)
```

> SAM3와 EXAONE을 동시에 GPU에 올리기 어려운 경우, Step 1 완료 후 SAM3를 언로드하거나 `SAM3_DEVICE=cpu`로 설정하여 EXAONE에 GPU 메모리를 양보할 수 있습니다.

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

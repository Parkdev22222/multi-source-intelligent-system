# Multi-Source Intelligent System (MSIS)

Project Maven-inspired aerial imagery intelligence pipeline that performs:
- **SAM2** automatic segmentation of satellite/drone images
- **CLIP** zero-shot military object classification
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
DETECTION LAYER  ──►  SAM2AutomaticMaskGenerator
                       + CLIP zero-shot classifier
                       → object_class, confidence, bbox, lat/lon
         │
         ▼
SENSOR DB (SQLite)
  ├─ image_records       (image metadata)
  └─ detection_records   (object_id, class, confidence, lat, lon, time)
         │
         ▼
TEMPORAL PAIRING  ──►  current frame ↔ most-recent past frame
                        status: new / matched / disappeared
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

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# SAM2 (from Meta AI)
pip install git+https://github.com/facebookresearch/sam2.git

# CLIP (from OpenAI)
pip install git+https://github.com/openai/CLIP.git
```

### 2. Download SAM2 checkpoint

```bash
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
# Rename to match config:
mv checkpoints/sam2.1_hiera_large.pt checkpoints/sam2_hiera_large.pt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env: set LLM_BACKEND, model paths, etc.
```

### 4. Generate sample data and run pipeline

```bash
# Generate synthetic satellite/drone test images
python main.py --generate-samples

# Run full pipeline (detection → pairing → report)
python main.py --metadata data/images/metadata.json \
               --report-output data/reports/report.txt
```

### 5. Use Ollama backend (lighter weight)

```bash
# Pull EXAONE model via Ollama
ollama pull exaone4:32b

# Run with Ollama backend
LLM_BACKEND=ollama python main.py --generate-samples
```

---

## Image Metadata Format

Place your satellite/drone image files and a `metadata.json` in `data/images/`:

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

---

## Military Object Classes

The system detects these classes using CLIP zero-shot classification:

```
military tank · armored personnel carrier · military truck · military jeep
fighter aircraft · helicopter · military ship · missile launcher · artillery
military building · radar installation · military personnel
supply depot · fuel storage · command post
civilian vehicle · civilian building · road · runway · unknown object
```

---

## Notes

- **SAM3**: As of 2025, Meta AI's latest release is SAM2. This system uses SAM2.
  The architecture is forward-compatible with SAM3 when released.
- **EXAONE4-32b**: Uses LG AI Research's EXAONE 4.0 32B Instruct model.
  Falls back to Ollama if HuggingFace backend is unavailable.
- **Fallback mode**: If SAM2/CLIP are not installed, the system uses a
  grid-based pseudo-detector for development/testing purposes.

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
                       → object_class, confidence, bbox, mask, lat/lon
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

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

SAM3 is available via HuggingFace Transformers (≥ 4.48.0) — no separate installation needed.

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env: set LLM_BACKEND, SAM3_DEVICE, etc.
```

### 3. Generate sample data and run pipeline

```bash
# Generate synthetic satellite/drone test images
python main.py --generate-samples

# Run full pipeline (detection → pairing → report)
python main.py --metadata data/images/metadata.json \
               --report-output data/reports/report.txt
```

### 4. Use Ollama backend for LLM (lighter weight)

```bash
# Pull EXAONE model via Ollama
ollama pull exaone4:32b

# Run with Ollama backend
LLM_BACKEND=ollama python main.py --generate-samples
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

## Notes

- **SAM3 resolution**: SAM3 requires a fixed inference resolution of 1008×1008.
  Images are automatically resized by `Sam3Processor` before inference.
- **EXAONE4-32b**: Uses LG AI Research's EXAONE 4.0 32B Instruct model.
  Falls back to Ollama if HuggingFace backend is unavailable.
- **Fallback mode**: If SAM3 weights are unavailable (no GPU / offline),
  the system uses a grid-based pseudo-detector for development/testing.

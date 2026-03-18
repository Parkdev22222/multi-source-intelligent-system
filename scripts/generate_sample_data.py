"""
Generate synthetic satellite/drone images and metadata.json for testing
the MSIS pipeline without real sensor data.

Generates two time-point "snapshots" of the same geographic region:
  - T-1 (past):    Saved into Sensor DB as historical reference.
  - T0  (current): Processed by the pipeline's detection + pairing steps.

Image content is synthetic:
  - Coloured rectangles representing military objects (vehicles, buildings)
  - Random noise background simulating terrain texture
  - Objects differ between T-1 and T0 to trigger change-detection signals
"""

import json
import logging
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


SAMPLE_IMAGES_DIR = Path("data/images/sample")
METADATA_PATH = Path("data/images/metadata.json")

# Geographic region of interest (example: a military base region)
REGION_LAT_CENTER = 37.5665   # Seoul area as example
REGION_LON_CENTER = 126.9780
REGION_HALF_DEG = 0.05

IMG_WIDTH = 512
IMG_HEIGHT = 512

# Colour palette for different object types (RGB)
OBJECT_COLOURS = {
    "military tank":             (80,  60,  40),
    "armored personnel carrier": (70,  80,  50),
    "military truck":            (60,  70,  60),
    "military building":         (150, 140, 130),
    "fighter aircraft":          (200, 200, 220),
    "missile launcher":          (90,  50,  50),
    "radar installation":        (180, 180, 100),
    "supply depot":              (160, 120, 80),
    "command post":              (140, 100, 80),
    "military jeep":             (50,  80,  50),
}
OBJECT_TYPES = list(OBJECT_COLOURS.keys())


def _noisy_background(h: int, w: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Earthy terrain colours
    base = rng.integers(60, 120, (h, w, 3), dtype=np.uint8)
    noise = rng.integers(0, 20, (h, w, 3), dtype=np.uint8)
    return np.clip(base.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)


def _place_objects(
    image: np.ndarray,
    objects: List[Tuple[str, int, int, int, int]],  # (class, x1, y1, x2, y2)
) -> np.ndarray:
    img = image.copy()
    for obj_class, x1, y1, x2, y2 in objects:
        colour = OBJECT_COLOURS.get(obj_class, (200, 200, 200))
        img[y1:y2, x1:x2] = colour
        # Add a dark outline
        img[y1:y1+2, x1:x2] = (20, 20, 20)
        img[y2-2:y2, x1:x2] = (20, 20, 20)
        img[y1:y2, x1:x1+2] = (20, 20, 20)
        img[y1:y2, x2-2:x2] = (20, 20, 20)
    return img


def _random_objects(n: int, seed: int, w: int = IMG_WIDTH, h: int = IMG_HEIGHT):
    rng = random.Random(seed)
    objects = []
    for _ in range(n):
        obj_class = rng.choice(OBJECT_TYPES)
        obj_w = rng.randint(20, 60)
        obj_h = rng.randint(15, 40)
        x1 = rng.randint(10, w - obj_w - 10)
        y1 = rng.randint(10, h - obj_h - 10)
        objects.append((obj_class, x1, y1, x1 + obj_w, y1 + obj_h))
    return objects


def generate_sample_data(metadata_output: str = str(METADATA_PATH)) -> None:
    """
    Generate two sets of synthetic images (past and current time points)
    and write metadata.json.
    """
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    Path(metadata_output).parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image as PILImage
    except ImportError:
        logger.error("Pillow not installed. Run: pip install Pillow")
        return

    now = datetime.now(tz=timezone.utc).replace(microsecond=0)
    past_time = now - timedelta(hours=6)

    metadata_entries = []

    # Two sub-regions within the same area
    sub_regions = [
        {
            "name": "alpha",
            "lat": REGION_LAT_CENTER + 0.01,
            "lon": REGION_LON_CENTER - 0.01,
            "half": 0.01,
        },
        {
            "name": "bravo",
            "lat": REGION_LAT_CENTER - 0.01,
            "lon": REGION_LON_CENTER + 0.01,
            "half": 0.01,
        },
    ]

    for region in sub_regions:
        lat_c = region["lat"]
        lon_c = region["lon"]
        half = region["half"]
        name = region["name"]

        for time_label, capture_time, seed_base in [
            ("past", past_time, 100),
            ("current", now, 200),
        ]:
            # Generate image
            bg = _noisy_background(IMG_HEIGHT, IMG_WIDTH, seed=seed_base)
            n_objects = random.randint(4, 8)
            objects = _random_objects(n_objects, seed=seed_base + 1)
            img_array = _place_objects(bg, objects)

            filename = f"{name}_{time_label}_{capture_time.strftime('%Y%m%dT%H%M%S')}.png"
            img_path = SAMPLE_IMAGES_DIR / filename

            PILImage.fromarray(img_array.astype(np.uint8), "RGB").save(str(img_path))
            logger.info(f"[SampleGen] Saved {img_path}  objects={n_objects}")

            metadata_entries.append({
                "image_file": str(img_path.relative_to(Path("data/images"))),
                "capture_time": capture_time.isoformat(),
                "source_type": "satellite" if time_label == "past" else random.choice(["satellite", "drone"]),
                "lat_center": lat_c,
                "lon_center": lon_c,
                "lat_min": lat_c - half,
                "lat_max": lat_c + half,
                "lon_min": lon_c - half,
                "lon_max": lon_c + half,
                "resolution_m": 0.5,
                "sensor_platform": "WorldView-3" if time_label == "past" else "MQ-9 Reaper",
                "region_name": name,
            })

    with open(metadata_output, "w") as f:
        json.dump(metadata_entries, f, indent=2, default=str)

    logger.info(f"[SampleGen] Written metadata to {metadata_output} ({len(metadata_entries)} entries)")
    print(f"Sample data generated: {len(metadata_entries)} images in {SAMPLE_IMAGES_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_sample_data()

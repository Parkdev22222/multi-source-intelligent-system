"""
Project Maven-inspired Multi-Source Intelligent System Configuration
Based on DoD Project Maven architecture for AI-driven aerial imagery analysis.
"""

import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
DB_DIR = DATA_DIR / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)

# --- Database ---
SENSOR_DB_PATH = str(DB_DIR / "sensor_detections.db")
PAIRING_DB_PATH = str(DB_DIR / "object_pairings.db")

# --- SAM3 Model (Segment Anything Model 3 by Meta AI) ---
# SAM3 performs text-prompted concept segmentation in a single forward pass,
# replacing the SAM2 + CLIP two-stage pipeline.
# Model card: https://huggingface.co/facebook/sam3
SAM3_MODEL_NAME = os.getenv("SAM3_MODEL_NAME", "facebook/sam3")
SAM3_DEVICE = os.getenv("SAM3_DEVICE", "cuda")  # "cuda" or "cpu"
# SAM3 requires a fixed inference resolution of 1008×1008
SAM3_INFERENCE_SIZE = 1008

# --- Military Object Classes (aerial/satellite imagery) ---
MILITARY_OBJECT_CLASSES = [
    "military tank",
    "armored personnel carrier",
    "military truck",
    "military jeep",
    "fighter aircraft",
    "helicopter",
    "military ship",
    "missile launcher",
    "artillery",
    "military building",
    "radar installation",
    "military personnel",
    "supply depot",
    "fuel storage",
    "command post",
    "civilian vehicle",
    "civilian building",
    "road",
    "runway",
    "unknown object",
]

# --- Coordinate Matching ---
# Radius (in degrees) to consider two detections from the "same region"
# ~1 degree ≈ 111 km; 0.001 degree ≈ 111 m
COORDINATE_MATCH_RADIUS_DEG = float(os.getenv("COORD_MATCH_RADIUS", "0.01"))

# --- LLM (EXAONE4-32b) ---
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "LGAI-EXAONE/EXAONE-4.0-32B-Instruct")
LLM_DEVICE = os.getenv("LLM_DEVICE", "cuda")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "2048"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
# Ollama endpoint alternative
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone4:32b")
# Use "huggingface" or "ollama"
LLM_BACKEND = os.getenv("LLM_BACKEND", "huggingface")

# --- Detection Confidence ---
DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv("DETECTION_CONFIDENCE", "0.3"))
NMS_IOU_THRESHOLD = float(os.getenv("NMS_IOU", "0.5"))

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

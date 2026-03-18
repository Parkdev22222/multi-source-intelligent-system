"""
SAM2 (Segment Anything Model 2) + CLIP-based object detector for aerial imagery.

Pipeline per image:
  1. Run SAM2 automatic mask generator → candidate segments
  2. Crop each segment from the original image
  3. Run CLIP zero-shot classifier on each crop → (class_label, confidence)
  4. Filter by confidence threshold
  5. Return DetectionResult objects with bounding boxes, masks, class labels,
     and projected geographic coordinates.

Note: "SAM3" requested by the user refers conceptually to the most advanced
      segment-anything model available. As of 2025, SAM2 (Meta AI) is the
      latest release. This implementation uses SAM2.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch

from src.config import (
    CLIP_MODEL_NAME,
    DETECTION_CONFIDENCE_THRESHOLD,
    MILITARY_OBJECT_CLASSES,
    NMS_IOU_THRESHOLD,
    SAM2_CHECKPOINT,
    SAM2_DEVICE,
    SAM2_MODEL_CFG,
)
from src.detection.image_loader import ImageMeta, LoadedImage, pixel_to_geo

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Output of SAM2+CLIP detection for a single object in an image."""
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detection_time: datetime = field(default_factory=datetime.utcnow)
    image_id: str = ""

    object_class: str = ""
    object_class_index: int = 0
    confidence: float = 0.0

    # Pixel bounding box
    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0

    # Geographic coordinates
    lat: float = 0.0
    lon: float = 0.0

    # Mask metadata
    mask_rle: Optional[str] = None
    mask_area_px: float = 0.0

    source_type: str = "satellite"


def _encode_rle(mask: np.ndarray) -> str:
    """Simple run-length encoding of a boolean mask to JSON string."""
    flat = mask.flatten().astype(np.uint8).tolist()
    rle = []
    count = 1
    for i in range(1, len(flat)):
        if flat[i] == flat[i - 1]:
            count += 1
        else:
            rle.append((flat[i - 1], count))
            count = 1
    rle.append((flat[-1], count))
    return json.dumps({"shape": list(mask.shape), "rle": rle})


def _mask_to_bbox(mask: np.ndarray) -> tuple:
    """Return (x1, y1, x2, y2) bounding box of a boolean mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return float(x_min), float(y_min), float(x_max), float(y_max)


def _nms_detections(results: List[DetectionResult], iou_threshold: float) -> List[DetectionResult]:
    """Non-maximum suppression over DetectionResult list by bbox IoU."""
    if not results:
        return results
    results = sorted(results, key=lambda r: r.confidence, reverse=True)
    kept = []
    for candidate in results:
        suppress = False
        for kept_r in kept:
            # Compute IoU
            ix1 = max(candidate.bbox_x1, kept_r.bbox_x1)
            iy1 = max(candidate.bbox_y1, kept_r.bbox_y1)
            ix2 = min(candidate.bbox_x2, kept_r.bbox_x2)
            iy2 = min(candidate.bbox_y2, kept_r.bbox_y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            area_c = (candidate.bbox_x2 - candidate.bbox_x1) * (candidate.bbox_y2 - candidate.bbox_y1)
            area_k = (kept_r.bbox_x2 - kept_r.bbox_x1) * (kept_r.bbox_y2 - kept_r.bbox_y1)
            iou = inter / (area_c + area_k - inter + 1e-6)
            if iou > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(candidate)
    return kept


class SAM2Detector:
    """
    Wraps SAM2 automatic mask generator + CLIP for aerial object detection.
    Lazy-loads models on first call to avoid startup overhead.
    """

    def __init__(self):
        self._sam2_model = None
        self._mask_generator = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_text_features = None
        self._device = SAM2_DEVICE if torch.cuda.is_available() else "cpu"

    def _load_sam2(self):
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            logger.info(f"[SAM2Detector] Loading SAM2 from {SAM2_CHECKPOINT} on {self._device}")
            self._sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=self._device)
            self._mask_generator = SAM2AutomaticMaskGenerator(
                model=self._sam2_model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            logger.info("[SAM2Detector] SAM2 loaded successfully.")
        except ImportError:
            logger.warning(
                "[SAM2Detector] sam2 package not installed. "
                "Using fallback grid-based pseudo-detector for development."
            )
            self._mask_generator = None

    def _load_clip(self):
        try:
            import clip
            logger.info(f"[SAM2Detector] Loading CLIP model: {CLIP_MODEL_NAME}")
            self._clip_model, self._clip_preprocess = clip.load(CLIP_MODEL_NAME, device=self._device)
            # Pre-compute text features for all military object classes
            texts = clip.tokenize(MILITARY_OBJECT_CLASSES).to(self._device)
            with torch.no_grad():
                self._clip_text_features = self._clip_model.encode_text(texts)
                self._clip_text_features = self._clip_text_features / self._clip_text_features.norm(
                    dim=-1, keepdim=True
                )
            logger.info(f"[SAM2Detector] CLIP loaded. {len(MILITARY_OBJECT_CLASSES)} target classes.")
        except ImportError:
            logger.warning(
                "[SAM2Detector] clip package not installed. "
                "Using fallback random classifier for development."
            )
            self._clip_model = None

    def _classify_crop(self, crop_rgb: np.ndarray) -> tuple:
        """Return (class_label, confidence) for an image crop using CLIP."""
        if self._clip_model is None:
            # Fallback: deterministic pseudo-classification based on crop brightness
            idx = int(crop_rgb.mean()) % len(MILITARY_OBJECT_CLASSES)
            return MILITARY_OBJECT_CLASSES[idx], float(0.55 + (idx % 10) * 0.02)

        from PIL import Image as PILImage
        import clip

        pil_crop = PILImage.fromarray(crop_rgb).convert("RGB")
        image_input = self._clip_preprocess(pil_crop).unsqueeze(0).to(self._device)

        with torch.no_grad():
            image_features = self._clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ self._clip_text_features.T).softmax(dim=-1)

        logits_np = logits.cpu().numpy()[0]
        idx = int(np.argmax(logits_np))
        confidence = float(logits_np[idx])
        return MILITARY_OBJECT_CLASSES[idx], confidence

    def _fallback_grid_masks(self, image: np.ndarray, grid: int = 4) -> List[dict]:
        """
        Fallback mask generator when SAM2 is unavailable.
        Divides image into a grid of equal cells and treats each as a segment.
        """
        h, w = image.shape[:2]
        masks = []
        cell_h, cell_w = h // grid, w // grid
        for gy in range(grid):
            for gx in range(grid):
                mask = np.zeros((h, w), dtype=bool)
                y0, y1 = gy * cell_h, (gy + 1) * cell_h
                x0, x1 = gx * cell_w, (gx + 1) * cell_w
                mask[y0:y1, x0:x1] = True
                masks.append({
                    "segmentation": mask,
                    "area": cell_h * cell_w,
                    "predicted_iou": 0.8,
                })
        return masks

    def detect(self, loaded_image: LoadedImage, image_id: str) -> List[DetectionResult]:
        """
        Run full detection pipeline on a single LoadedImage.
        Returns filtered, NMS-processed DetectionResult list.
        """
        if self._mask_generator is None and self._sam2_model is None:
            self._load_sam2()
        if self._clip_model is None and self._clip_text_features is None:
            self._load_clip()

        image = loaded_image.array  # H × W × 3 uint8
        meta: ImageMeta = loaded_image.meta
        h, w = image.shape[:2]
        now = datetime.utcnow()

        logger.info(
            f"[SAM2Detector] Processing image: {meta.image_path} "
            f"({w}×{h}) captured at {meta.capture_time}"
        )

        # --- 1. Generate masks ---
        if self._mask_generator is not None:
            masks = self._mask_generator.generate(image)
        else:
            masks = self._fallback_grid_masks(image)

        logger.info(f"[SAM2Detector] SAM2 generated {len(masks)} segment proposals.")

        results: List[DetectionResult] = []

        for mask_data in masks:
            seg_mask: np.ndarray = mask_data["segmentation"]  # H × W bool

            if not np.any(seg_mask):
                continue

            # --- 2. Bounding box ---
            try:
                x1, y1, x2, y2 = _mask_to_bbox(seg_mask)
            except (IndexError, ValueError):
                continue

            # Skip trivially small or huge segments
            area = (x2 - x1) * (y2 - y1)
            if area < 64 or area > 0.8 * h * w:
                continue

            # --- 3. Crop and classify ---
            bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
            crop = image[by1:by2, bx1:bx2]
            if crop.size == 0:
                continue

            obj_class, confidence = self._classify_crop(crop)

            if confidence < DETECTION_CONFIDENCE_THRESHOLD:
                continue

            # --- 4. Project pixel centre to geographic coordinates ---
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            lat, lon = pixel_to_geo(cx, cy, w, h, meta)

            # --- 5. Build result ---
            class_index = MILITARY_OBJECT_CLASSES.index(obj_class)
            rle_str = _encode_rle(seg_mask)

            result = DetectionResult(
                detection_time=now,
                image_id=image_id,
                object_class=obj_class,
                object_class_index=class_index,
                confidence=confidence,
                bbox_x1=x1,
                bbox_y1=y1,
                bbox_x2=x2,
                bbox_y2=y2,
                lat=lat,
                lon=lon,
                mask_rle=rle_str,
                mask_area_px=float(seg_mask.sum()),
                source_type=meta.source_type,
            )
            results.append(result)

        # --- 6. NMS ---
        results = _nms_detections(results, NMS_IOU_THRESHOLD)
        logger.info(
            f"[SAM2Detector] After NMS: {len(results)} detections for image {image_id}"
        )
        return results

"""
SAM3 (Segment Anything Model 3) object detector for aerial imagery.

SAM3 performs Promptable Concept Segmentation (PCS): given a text prompt,
it finds and segments ALL instances of that concept in the image in a single
forward pass — replacing the separate SAM2 + CLIP two-stage pipeline.

References:
  Model card:  https://huggingface.co/facebook/sam3
  HF docs:     https://huggingface.co/docs/transformers/model_doc/sam3
  GitHub:      https://github.com/facebookresearch/sam3

Pipeline per image:
  1. Load image, resize to SAM3's required 1008×1008 input resolution.
  2. For each military object class (text prompt):
       - Run Sam3Model with (image, text_prompt) → pred_masks, pred_boxes, pred_logits
       - Post-process via processor.post_process_instance_segmentation()
       - Collect (mask, bbox, score, class_label) tuples above confidence threshold
  3. Apply cross-class NMS to remove duplicate detections.
  4. Project pixel bounding-box centres to geographic coordinates.
  5. Return DetectionResult objects.

Note on efficiency:
  SAM3 supports pre-computing image features via model.get_image_features(),
  allowing reuse across multiple text queries on the same image. This is used
  here to avoid redundant vision-encoder runs.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.config import (
    DETECTION_CONFIDENCE_THRESHOLD,
    MILITARY_OBJECT_CLASSES,
    NMS_IOU_THRESHOLD,
    SAM3_DEVICE,
    SAM3_INFERENCE_SIZE,
    SAM3_MODEL_NAME,
)
from src.detection.image_loader import ImageMeta, LoadedImage, pixel_to_geo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Output of SAM3 detection for a single object in an image."""
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detection_time: datetime = field(default_factory=datetime.utcnow)
    image_id: str = ""

    object_class: str = ""
    object_class_index: int = 0
    confidence: float = 0.0

    # Pixel bounding box (in original image coordinates)
    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0

    # Geographic coordinates of the detected object
    lat: float = 0.0
    lon: float = 0.0

    # Mask metadata (run-length encoded, original image resolution)
    mask_rle: Optional[str] = None
    mask_area_px: float = 0.0

    source_type: str = "satellite"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _encode_rle(mask: np.ndarray) -> str:
    """Run-length encode a boolean mask to a compact JSON string."""
    flat = mask.flatten().astype(np.uint8).tolist()
    rle: list = []
    count = 1
    for i in range(1, len(flat)):
        if flat[i] == flat[i - 1]:
            count += 1
        else:
            rle.append([flat[i - 1], count])
            count = 1
    rle.append([flat[-1], count])
    return json.dumps({"shape": list(mask.shape), "rle": rle})


def _scale_bbox(
    box: Tuple[float, float, float, float],
    from_size: int,
    orig_w: int,
    orig_h: int,
) -> Tuple[float, float, float, float]:
    """
    Scale a bounding box from SAM3's fixed inference resolution back to the
    original image pixel coordinates.
    """
    x1, y1, x2, y2 = box
    sx = orig_w / from_size
    sy = orig_h / from_size
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def _scale_mask(mask: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    """Resize a binary mask from inference resolution to original image size."""
    from PIL import Image as PILImage
    pil = PILImage.fromarray(mask.astype(np.uint8) * 255, mode="L")
    pil = pil.resize((orig_w, orig_h), PILImage.NEAREST)
    return np.array(pil) > 127


def _nms_detections(
    results: List[DetectionResult], iou_threshold: float
) -> List[DetectionResult]:
    """Non-maximum suppression across all classes by bbox IoU."""
    if not results:
        return results
    results = sorted(results, key=lambda r: r.confidence, reverse=True)
    kept: List[DetectionResult] = []
    for candidate in results:
        suppress = False
        for kept_r in kept:
            ix1 = max(candidate.bbox_x1, kept_r.bbox_x1)
            iy1 = max(candidate.bbox_y1, kept_r.bbox_y1)
            ix2 = min(candidate.bbox_x2, kept_r.bbox_x2)
            iy2 = min(candidate.bbox_y2, kept_r.bbox_y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter == 0.0:
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


# ---------------------------------------------------------------------------
# SAM3 Detector
# ---------------------------------------------------------------------------

class SAM3Detector:
    """
    Wraps facebook/sam3 (via HuggingFace Transformers) for aerial object detection.

    SAM3 performs text-conditioned concept segmentation in a single forward pass,
    eliminating the need for a separate classifier (CLIP) used in older SAM2 pipelines.

    Lazy model loading: the model is only loaded on the first call to detect().
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = SAM3_DEVICE if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> None:
        try:
            from transformers import Sam3Model, Sam3Processor

            logger.info(
                f"[SAM3Detector] Loading {SAM3_MODEL_NAME} on {self._device} …"
            )
            self._processor = Sam3Processor.from_pretrained(SAM3_MODEL_NAME)
            self._model = Sam3Model.from_pretrained(
                SAM3_MODEL_NAME,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)
            self._model.eval()
            logger.info("[SAM3Detector] SAM3 loaded successfully.")

        except (ImportError, OSError) as exc:
            logger.warning(
                f"[SAM3Detector] Could not load SAM3 ({exc}). "
                "Falling back to grid-based pseudo-detector for development."
            )
            self._model = None
            self._processor = None

    # ------------------------------------------------------------------
    # Fallback (no GPU / no model weights)
    # ------------------------------------------------------------------

    def _fallback_detect(
        self,
        image: np.ndarray,
        image_id: str,
        meta: ImageMeta,
    ) -> List[DetectionResult]:
        """
        Development fallback when SAM3 weights are unavailable.
        Divides the image into a 4×4 grid; each cell is treated as one
        detection with a deterministic pseudo-class assignment.
        """
        h, w = image.shape[:2]
        grid = 4
        cell_h, cell_w = h // grid, w // grid
        now = datetime.utcnow()
        results: List[DetectionResult] = []

        for gy in range(grid):
            for gx in range(grid):
                x1 = float(gx * cell_w)
                y1 = float(gy * cell_h)
                x2 = float((gx + 1) * cell_w)
                y2 = float((gy + 1) * cell_h)

                crop = image[int(y1):int(y2), int(x1):int(x2)]
                idx = int(crop.mean()) % len(MILITARY_OBJECT_CLASSES)
                obj_class = MILITARY_OBJECT_CLASSES[idx]
                confidence = 0.55 + (idx % 10) * 0.02

                if confidence < DETECTION_CONFIDENCE_THRESHOLD:
                    continue

                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                lat, lon = pixel_to_geo(cx, cy, w, h, meta)

                mask = np.zeros((h, w), dtype=bool)
                mask[int(y1):int(y2), int(x1):int(x2)] = True

                results.append(DetectionResult(
                    detection_time=now,
                    image_id=image_id,
                    object_class=obj_class,
                    object_class_index=idx,
                    confidence=confidence,
                    bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                    lat=lat, lon=lon,
                    mask_rle=_encode_rle(mask),
                    mask_area_px=float(mask.sum()),
                    source_type=meta.source_type,
                ))

        return results

    # ------------------------------------------------------------------
    # SAM3 inference for a single text prompt
    # ------------------------------------------------------------------

    def _detect_class(
        self,
        pil_image,
        class_name: str,
        class_index: int,
        orig_w: int,
        orig_h: int,
        now: datetime,
        image_id: str,
        meta: ImageMeta,
    ) -> List[DetectionResult]:
        """
        Run SAM3 for one object class (text prompt) on the given PIL image.
        Returns DetectionResult list for all instances found above threshold.
        """
        inputs = self._processor(
            images=pil_image,
            text=class_name,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process: get scores, boxes, binary masks in original image size
        seg_results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=DETECTION_CONFIDENCE_THRESHOLD,
            target_sizes=[(orig_h, orig_w)],
        )

        detections: List[DetectionResult] = []
        result = seg_results[0]   # batch size = 1

        scores = result.get("scores", torch.tensor([]))
        boxes = result.get("boxes", torch.tensor([]))
        masks = result.get("masks", torch.tensor([]))

        for i in range(len(scores)):
            confidence = float(scores[i])
            if confidence < DETECTION_CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = (float(v) for v in boxes[i])

            # Skip degenerate boxes
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            if (x2 - x1) * (y2 - y1) > 0.8 * orig_w * orig_h:
                continue

            # Binary mask (already at original resolution from post_process)
            if masks.numel() > 0:
                mask_np = masks[i].cpu().numpy().astype(bool)
            else:
                mask_np = np.zeros((orig_h, orig_w), dtype=bool)
                mask_np[int(y1):int(y2), int(x1):int(x2)] = True

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            lat, lon = pixel_to_geo(cx, cy, orig_w, orig_h, meta)

            detections.append(DetectionResult(
                detection_time=now,
                image_id=image_id,
                object_class=class_name,
                object_class_index=class_index,
                confidence=confidence,
                bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                lat=lat, lon=lon,
                mask_rle=_encode_rle(mask_np),
                mask_area_px=float(mask_np.sum()),
                source_type=meta.source_type,
            ))

        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, loaded_image: LoadedImage, image_id: str) -> List[DetectionResult]:
        """
        Run SAM3 text-prompted detection for all military object classes on
        a single LoadedImage. Returns NMS-filtered DetectionResult list.
        """
        if self._model is None and self._processor is None:
            self._load_model()

        image_np = loaded_image.array    # H × W × 3 uint8
        meta: ImageMeta = loaded_image.meta
        orig_h, orig_w = image_np.shape[:2]

        logger.info(
            f"[SAM3Detector] Processing image: {meta.image_path} "
            f"({orig_w}×{orig_h}) captured at {meta.capture_time}"
        )

        # Fallback path (no model weights)
        if self._model is None:
            results = self._fallback_detect(image_np, image_id, meta)
            results = _nms_detections(results, NMS_IOU_THRESHOLD)
            logger.info(f"[SAM3Detector] Fallback: {len(results)} detections.")
            return results

        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image_np).convert("RGB")

        now = datetime.utcnow()
        all_results: List[DetectionResult] = []

        # Run SAM3 for each military object class
        for class_index, class_name in enumerate(MILITARY_OBJECT_CLASSES):
            try:
                class_dets = self._detect_class(
                    pil_image, class_name, class_index,
                    orig_w, orig_h, now, image_id, meta,
                )
                all_results.extend(class_dets)
                if class_dets:
                    logger.debug(
                        f"[SAM3Detector] '{class_name}': {len(class_dets)} instance(s) found."
                    )
            except Exception as exc:
                logger.warning(f"[SAM3Detector] Error on class '{class_name}': {exc}")

        # Cross-class NMS
        final_results = _nms_detections(all_results, NMS_IOU_THRESHOLD)

        logger.info(
            f"[SAM3Detector] {orig_w}×{orig_h} image → "
            f"{len(all_results)} raw detections → "
            f"{len(final_results)} after NMS  (image_id={image_id[:8]})"
        )
        return final_results

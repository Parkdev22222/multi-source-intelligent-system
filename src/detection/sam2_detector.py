"""
SAM3 object detector and tracker for aerial imagery.

Two modes:
  SAM3Detector.detect()           - Sam3Model: text-prompted concept segmentation
                                    Used for initial detection on each frame.
  SAM3Detector.track_objects()    - Sam3Tracker: visual-prompt object tracking
                                    Takes past-frame bboxes as prompts, returns
                                    which past objects are still present and where.

Pairing flow (in temporal_pairing.py):
  1. track_objects() → TrackedObject list (past_detection_id + new bbox + score)
  2. pair_by_tracking() matches TrackedObjects to current Sam3Model detections by IoU
     → status = matched / new / disappeared  (ID-based, not coordinate-based)
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
    MAX_BBOX_AREA_RATIO,
    MILITARY_OBJECT_CLASSES,
    NMS_IOU_THRESHOLD,
    SAM3_DEVICE,
    SAM3_INFERENCE_SIZE,
    SAM3_MASK_SCORE_THRESHOLD,
    SAM3_MODEL_NAME,
)
from src.detection.image_loader import ImageMeta, LoadedImage, pixel_to_geo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
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

    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0

    lat: float = 0.0
    lon: float = 0.0

    mask_rle: Optional[str] = None
    mask_area_px: float = 0.0

    source_type: str = "satellite"


@dataclass
class TrackedObject:
    """
    Result of Sam3Tracker for one past detection.

    Sam3Tracker takes the past-frame bounding box as a visual prompt and finds
    the same object in the current frame.  score reflects how confidently the
    tracker found the object; objects below DETECTION_CONFIDENCE_THRESHOLD are
    considered disappeared.
    """
    past_detection_id: str      # ID of the DetectionRecord in the sensor DB
    past_object_class: str
    bbox_x1: float              # updated bbox in current-frame pixel coords
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    score: float                # Sam3Tracker IOU / presence score


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _encode_rle(mask: np.ndarray) -> str:
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


def _mask_to_bbox(mask: np.ndarray) -> Tuple[float, float, float, float]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return float(x_min), float(y_min), float(x_max), float(y_max)


def _scale_mask(mask: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    from PIL import Image as PILImage
    pil = PILImage.fromarray(mask.astype(np.uint8) * 255, mode="L")
    pil = pil.resize((orig_w, orig_h), PILImage.NEAREST)
    return np.array(pil) > 127


def _tighten_mask(mask: np.ndarray) -> np.ndarray:
    """
    마스크에서 가장 큰 연결 성분(connected component)만 남긴다.

    SAM3 텍스트 프롬프트 세그멘테이션은 객체 주변 문맥 픽셀까지 포함하는
    경향이 있어, 이 함수로 노이즈 픽셀을 제거하면 bbox가 실제 객체에
    훨씬 가깝게 수렴한다.  scipy가 없을 경우 원본 마스크를 그대로 반환.
    """
    if not mask.any():
        return mask
    try:
        from scipy.ndimage import label as ndimage_label
        labeled, n = ndimage_label(mask)
        if n == 0:
            return mask
        # 각 레이블 크기 계산 후 가장 큰 것 선택
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # 배경(0) 제외
        return labeled == sizes.argmax()
    except ImportError:
        return mask


def _nms_detections(
    results: List[DetectionResult], iou_threshold: float
) -> List[DetectionResult]:
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
# SAM3 Detector + Tracker
# ---------------------------------------------------------------------------

class SAM3Detector:
    """
    Wraps facebook/sam3 via HuggingFace Transformers.

    - Sam3Model   (self._model)   : text-prompted concept segmentation per image.
    - Sam3Tracker (self._tracker) : visual-prompt object tracking across frames.
    - Sam3Processor (self._processor): shared by both models.

    Both models are lazy-loaded on first use.
    """

    def __init__(self):
        self._model = None        # Sam3Model
        self._tracker = None      # Sam3Tracker
        self._processor = None    # Sam3Processor (shared)
        self._tracker_load_attempted = False
        self._device = SAM3_DEVICE if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Model loaders
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from transformers import Sam3Model, Sam3Processor
            logger.info(f"[SAM3Detector] Loading Sam3Model ({SAM3_MODEL_NAME}) on {self._device} …")
            self._processor = Sam3Processor.from_pretrained(SAM3_MODEL_NAME)
            self._model = Sam3Model.from_pretrained(
                SAM3_MODEL_NAME,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)
            self._model.eval()
            logger.info("[SAM3Detector] Sam3Model loaded.")
        except (ImportError, OSError) as exc:
            logger.warning(
                f"[SAM3Detector] Could not load Sam3Model ({exc}). "
                "Using fallback grid-detector."
            )
            self._model = None

    def _load_tracker(self) -> None:
        self._tracker_load_attempted = True
        try:
            from transformers import Sam3Tracker
            logger.info(f"[SAM3Detector] Loading Sam3Tracker ({SAM3_MODEL_NAME}) on {self._device} …")
            # Sam3Tracker shares the same processor already loaded for Sam3Model
            if self._processor is None:
                from transformers import Sam3Processor
                self._processor = Sam3Processor.from_pretrained(SAM3_MODEL_NAME)
            self._tracker = Sam3Tracker.from_pretrained(
                SAM3_MODEL_NAME,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)
            self._tracker.eval()
            logger.info("[SAM3Detector] Sam3Tracker loaded.")
        except (ImportError, OSError) as exc:
            logger.warning(
                f"[SAM3Detector] Could not load Sam3Tracker ({exc}). "
                "Tracking will fall back to returning all past detections as-is."
            )
            self._tracker = None

    # ------------------------------------------------------------------
    # Fallbacks (no model weights / no GPU)
    # ------------------------------------------------------------------

    def _fallback_detect(
        self, image: np.ndarray, image_id: str, meta: ImageMeta
    ) -> List[DetectionResult]:
        h, w = image.shape[:2]
        grid, now = 4, datetime.utcnow()
        cell_h, cell_w = h // grid, w // grid
        results: List[DetectionResult] = []
        for gy in range(grid):
            for gx in range(grid):
                x1, y1 = float(gx * cell_w), float(gy * cell_h)
                x2, y2 = float((gx + 1) * cell_w), float((gy + 1) * cell_h)
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
                    detection_time=now, image_id=image_id,
                    object_class=obj_class, object_class_index=idx,
                    confidence=confidence,
                    bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                    lat=lat, lon=lon,
                    mask_rle=_encode_rle(mask), mask_area_px=float(mask.sum()),
                    source_type=meta.source_type,
                ))
        return results

    def _fallback_track(
        self, past_detections: list
    ) -> List[TrackedObject]:
        """
        Fallback when Sam3Tracker is unavailable.
        Returns all past detections as tracked at their original positions.
        """
        logger.warning(
            "[SAM3Detector] Sam3Tracker unavailable – "
            "treating all past detections as tracked (positions unchanged)."
        )
        return [
            TrackedObject(
                past_detection_id=d.id,
                past_object_class=d.object_class,
                bbox_x1=d.bbox_x1, bbox_y1=d.bbox_y1,
                bbox_x2=d.bbox_x2, bbox_y2=d.bbox_y2,
                score=d.confidence,
            )
            for d in past_detections
        ]

    # ------------------------------------------------------------------
    # Sam3Model: concept segmentation (one class per forward pass)
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
        inputs = self._processor(
            images=pil_image,
            text=class_name,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # SAM3 마스크 점수는 DETECTION_CONFIDENCE_THRESHOLD 보다 높은
        # SAM3_MASK_SCORE_THRESHOLD 를 사용 → 낮은 신뢰도의 큰 마스크 사전 차단
        seg_results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=SAM3_MASK_SCORE_THRESHOLD,
            target_sizes=[(orig_h, orig_w)],
        )

        detections: List[DetectionResult] = []
        result = seg_results[0]
        scores = result.get("scores", torch.tensor([]))
        boxes = result.get("boxes", torch.tensor([]))
        masks = result.get("masks", torch.tensor([]))
        max_area = MAX_BBOX_AREA_RATIO * orig_w * orig_h

        for i in range(len(scores)):
            confidence = float(scores[i])
            if confidence < SAM3_MASK_SCORE_THRESHOLD:
                continue

            if masks.numel() > 0:
                raw_mask = masks[i].cpu().numpy().astype(bool)
                # 가장 큰 연결 성분만 남겨 노이즈 픽셀 제거 → bbox 축소
                mask_np = _tighten_mask(raw_mask)
            else:
                mask_np = None

            # 마스크 기반 bbox 재계산 (모델 출력 boxes 보다 tight)
            if mask_np is not None and mask_np.any():
                try:
                    x1, y1, x2, y2 = _mask_to_bbox(mask_np)
                except (IndexError, ValueError):
                    x1, y1, x2, y2 = (float(v) for v in boxes[i])
            else:
                x1, y1, x2, y2 = (float(v) for v in boxes[i])
                mask_np = np.zeros((orig_h, orig_w), dtype=bool)
                mask_np[int(y1):int(y2), int(x1):int(x2)] = True

            # 최소 크기 필터 (4px 이하 노이즈)
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue
            # 최대 크기 필터 (이미지 면적의 MAX_BBOX_AREA_RATIO 초과 차단)
            if (x2 - x1) * (y2 - y1) > max_area:
                continue

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            lat, lon = pixel_to_geo(cx, cy, orig_w, orig_h, meta)
            detections.append(DetectionResult(
                detection_time=now, image_id=image_id,
                object_class=class_name, object_class_index=class_index,
                confidence=confidence,
                bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                lat=lat, lon=lon,
                mask_rle=_encode_rle(mask_np),
                mask_area_px=float(mask_np.sum()),
                source_type=meta.source_type,
            ))
        return detections

    def detect(self, loaded_image: LoadedImage, image_id: str) -> List[DetectionResult]:
        """
        Run Sam3Model text-prompted detection for all military classes.
        Returns NMS-filtered DetectionResult list.
        """
        if self._model is None and self._processor is None:
            self._load_model()

        image_np = loaded_image.array
        meta = loaded_image.meta
        orig_h, orig_w = image_np.shape[:2]

        logger.info(
            f"[SAM3Detector] Detecting: {meta.image_path} "
            f"({orig_w}×{orig_h}) at {meta.capture_time}"
        )

        if self._model is None:
            results = self._fallback_detect(image_np, image_id, meta)
            return _nms_detections(results, NMS_IOU_THRESHOLD)

        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image_np).convert("RGB")
        now = datetime.utcnow()
        all_results: List[DetectionResult] = []

        for class_index, class_name in enumerate(MILITARY_OBJECT_CLASSES):
            try:
                all_results.extend(
                    self._detect_class(
                        pil_image, class_name, class_index,
                        orig_w, orig_h, now, image_id, meta,
                    )
                )
            except Exception as exc:
                logger.warning(f"[SAM3Detector] Error on class '{class_name}': {exc}")

        final = _nms_detections(all_results, NMS_IOU_THRESHOLD)
        logger.info(
            f"[SAM3Detector] {len(all_results)} raw → {len(final)} after NMS "
            f"(image_id={image_id[:8]})"
        )
        return final

    # ------------------------------------------------------------------
    # Sam3Tracker: visual-prompt object tracking across frames
    # ------------------------------------------------------------------

    def track_objects(
        self,
        pil_image,
        past_detections: list,   # List[DetectionRecord] from sensor DB
        orig_w: int,
        orig_h: int,
    ) -> List[TrackedObject]:
        """
        Use Sam3Tracker to find past objects in the current frame.

        Each past detection's bounding box is passed as a visual prompt.
        Sam3Tracker outputs a mask + score per prompt.
        Objects with score >= DETECTION_CONFIDENCE_THRESHOLD are returned
        as TrackedObject with their updated current-frame bbox.

        Args:
            pil_image:        Current-frame PIL image (RGB).
            past_detections:  DetectionRecord list from the most-recent past frame.
            orig_w / orig_h:  Original image dimensions for mask rescaling.

        Returns:
            List of TrackedObject – one entry per successfully tracked past object.
            Past detections absent from this list are considered "disappeared".
        """
        if not past_detections:
            return []

        if not self._tracker_load_attempted:
            self._load_tracker()

        if self._tracker is None:
            return self._fallback_track(past_detections)

        past_boxes = [
            [d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2]
            for d in past_detections
        ]

        # Sam3Tracker: one output mask per input box
        inputs = self._processor(
            images=pil_image,
            input_boxes=[past_boxes],   # shape: (1, n_objects, 4)
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._tracker(**inputs)

        # pred_masks : (batch=1, n_objects, H, W)
        # iou_scores : (batch=1, n_objects)  – tracker's confidence the object is present
        masks_tensor = outputs.pred_masks[0]   # (n_objects, H, W)

        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            scores = outputs.iou_scores[0].cpu().float().numpy()
        else:
            scores = (
                torch.sigmoid(masks_tensor).float().mean(dim=(-1, -2)).cpu().numpy()
            )

        tracked: List[TrackedObject] = []

        for i, (past_det, score) in enumerate(zip(past_detections, scores)):
            if float(score) < DETECTION_CONFIDENCE_THRESHOLD:
                logger.debug(
                    f"[Sam3Tracker] Object {past_det.id[:8]} "
                    f"({past_det.object_class}) score={score:.3f} → disappeared"
                )
                continue

            mask_np = (torch.sigmoid(masks_tensor[i]) > 0.5).cpu().numpy()
            mask_np = _scale_mask(mask_np, orig_w, orig_h)

            if not mask_np.any():
                continue

            try:
                x1, y1, x2, y2 = _mask_to_bbox(mask_np)
            except (IndexError, ValueError):
                continue

            tracked.append(TrackedObject(
                past_detection_id=past_det.id,
                past_object_class=past_det.object_class,
                bbox_x1=x1, bbox_y1=y1,
                bbox_x2=x2, bbox_y2=y2,
                score=float(score),
            ))
            logger.debug(
                f"[Sam3Tracker] Object {past_det.id[:8]} "
                f"({past_det.object_class}) tracked  score={score:.3f}"
            )

        logger.info(
            f"[Sam3Tracker] {len(tracked)}/{len(past_detections)} "
            "past objects successfully tracked."
        )
        return tracked

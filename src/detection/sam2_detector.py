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

    @staticmethod
    def _import_sam2():
        """
        transformers 버전에 따라 SAM2 관련 클래스를 반환.
        Sam2Model / Sam2Processor (transformers >= 4.47) 우선 시도,
        실패 시 구버전 SamModel / SamProcessor 사용.
        """
        try:
            from transformers import Sam2Model, Sam2Processor
            return Sam2Model, Sam2Processor
        except ImportError:
            pass
        try:
            from transformers import SamModel, SamProcessor
            return SamModel, SamProcessor
        except ImportError:
            raise ImportError(
                "transformers에서 SAM 모델을 찾을 수 없습니다. "
                "pip install 'transformers>=4.47.0' 를 실행하세요."
            )

    def _load_model(self) -> None:
        try:
            ModelCls, ProcessorCls = self._import_sam2()
            logger.info(
                f"[SAM3Detector] Loading {ModelCls.__name__} "
                f"({SAM3_MODEL_NAME}) on {self._device} …"
            )
            self._processor = ProcessorCls.from_pretrained(SAM3_MODEL_NAME)
            self._model = ModelCls.from_pretrained(
                SAM3_MODEL_NAME,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)
            self._model.eval()
            logger.info(f"[SAM3Detector] {ModelCls.__name__} loaded.")
        except (ImportError, OSError) as exc:
            logger.warning(
                f"[SAM3Detector] Could not load SAM model ({exc}). "
                "Using fallback grid-detector."
            )
            self._model = None

    def _load_tracker(self) -> None:
        self._tracker_load_attempted = True
        try:
            # SAM2VideoPredictor: 프레임 간 객체 추적 전용 클래스
            from transformers import Sam2VideoPredictor
            logger.info(
                f"[SAM3Detector] Loading Sam2VideoPredictor "
                f"({SAM3_MODEL_NAME}) on {self._device} …"
            )
            if self._processor is None:
                _, ProcessorCls = self._import_sam2()
                self._processor = ProcessorCls.from_pretrained(SAM3_MODEL_NAME)
            self._tracker = Sam2VideoPredictor.from_pretrained(
                SAM3_MODEL_NAME,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)
            self._tracker.eval()
            logger.info("[SAM3Detector] Sam2VideoPredictor loaded.")
        except (ImportError, OSError) as exc:
            logger.warning(
                f"[SAM3Detector] Could not load SAM2VideoPredictor ({exc}). "
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
    # SAM2 기반 탐지 (그리드 포인트 프롬프트 방식)
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
        # SAM2는 텍스트 프롬프트를 지원하지 않습니다.
        # 이미지를 격자로 나눠 각 셀 중심점을 positive point 프롬프트로 전달 →
        # SAM2가 해당 위치에 있는 객체를 세그멘테이션합니다.
        # (텍스트 기반 분류는 confidence 값으로 대체 - 마스크 IOU score 사용)
        grid_n = 4  # 4×4 = 16 포인트 프롬프트
        step_x = orig_w / grid_n
        step_y = orig_h / grid_n
        input_points = [
            [[int(step_x * (gx + 0.5)), int(step_y * (gy + 0.5))]]
            for gy in range(grid_n)
            for gx in range(grid_n)
        ]  # shape: (16, 1, 2)
        input_labels = [[1]] * len(input_points)  # 모두 positive point

        inputs = self._processor(
            images=pil_image,
            input_points=[input_points],
            input_labels=[input_labels],
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # pred_masks:  (1, num_points, num_masks_per_point, H, W)
        # iou_scores:  (1, num_points, num_masks_per_point)
        # multimask_output=True 이면 num_masks_per_point=3, False 이면 1
        # best mask per point: iou_scores 가 가장 높은 것 선택
        pred_masks  = outputs.pred_masks[0]   # (num_points, K, H, W)
        iou_scores  = outputs.iou_scores[0]   # (num_points, K)

        # 각 포인트에서 best mask 선택
        best_k      = iou_scores.argmax(dim=-1)   # (num_points,)
        best_scores = iou_scores.gather(1, best_k.unsqueeze(1)).squeeze(1)  # (num_points,)
        best_masks  = pred_masks[
            torch.arange(pred_masks.shape[0]), best_k
        ]  # (num_points, H, W)

        # 원본 해상도로 복원
        orig_sizes = inputs.get("original_sizes", torch.tensor([[orig_h, orig_w]]))
        reshaped   = inputs.get("reshaped_input_sizes", orig_sizes)
        masks_upsampled = self._processor.post_process_masks(
            best_masks.unsqueeze(0).cpu(),
            orig_sizes.cpu(),
            reshaped.cpu(),
        )[0]  # (num_points, H, W)  bool tensor

        # pseudo seg_results 구조로 통일
        seg_results = [{
            "scores": best_scores.cpu(),
            "masks":  masks_upsampled.cpu(),
            "boxes":  torch.zeros(len(input_points), 4),  # 마스크에서 재계산
        }]

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
        SAM2 그리드 포인트 프롬프트로 객체를 탐지한다.
        각 군사 클래스별로 _detect_class()를 호출하고 NMS로 중복 제거 후 반환.
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

        # Sam2VideoPredictor: 과거 bbox를 input_boxes 프롬프트로 전달
        # 각 박스에 대해 현재 프레임에서 같은 객체를 세그멘테이션
        inputs = self._processor(
            images=pil_image,
            input_boxes=[past_boxes],   # (1, n_objects, 4)
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._tracker(**inputs)

        # pred_masks : (1, n_objects, K, H, W)  K = 후보 마스크 수
        # iou_scores : (1, n_objects, K)
        raw_masks  = outputs.pred_masks[0]   # (n_objects, K, H, W)
        raw_scores = outputs.iou_scores[0]   # (n_objects, K)

        # 각 객체의 best mask 선택
        best_k       = raw_scores.argmax(dim=-1)          # (n_objects,)
        scores_np    = raw_scores.gather(
            1, best_k.unsqueeze(1)
        ).squeeze(1).cpu().float().numpy()                # (n_objects,)
        masks_tensor = raw_masks[
            torch.arange(raw_masks.shape[0]), best_k
        ]  # (n_objects, H, W)

        tracked: List[TrackedObject] = []

        for i, (past_det, score) in enumerate(zip(past_detections, scores_np)):
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

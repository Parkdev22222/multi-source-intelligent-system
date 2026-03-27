"""
단일 이미지 SAM3 객체탐지 + 결과 이미지 저장 스크립트

Usage:
    python detect_single.py --image <이미지경로> [옵션]

Examples:
    python detect_single.py --image data/images/sample/test.jpg
    python detect_single.py --image data/images/sample/test.jpg --output result.jpg
    python detect_single.py --image data/images/sample/test.jpg --classes "military tank" "helicopter"
    python detect_single.py --image data/images/sample/test.jpg --sam3-path /path/to/sam3
"""

import argparse
import sys
import os
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── 기본 설정 ──────────────────────────────────────────────────────────────
DEFAULT_SAM3_PATH = (
    Path(__file__).parent
    / "src/detection/sam3"
)
DEFAULT_MODEL_NAME = "facebook/sam3"
DEFAULT_CONFIDENCE  = 0.3
DEFAULT_MASK_SCORE  = 0.5
DEFAULT_NMS_IOU     = 0.3
DEFAULT_CLASSES = [
    "military tank",
    "armored personnel carrier",
    "military truck",
    "fighter aircraft",
    "helicopter",
    "military ship",
    "missile launcher",
    "artillery",
    "military building",
    "radar installation",
]
COLORS = [
    "#FF4040", "#FF8C00", "#FFD700", "#00FF88", "#00BFFF",
    "#DA70D6", "#FF69B4", "#7CFC00", "#FF6347", "#40E0D0",
]


# ── 헬퍼 ──────────────────────────────────────────────────────────────────

def _nms(detections, iou_threshold=0.3):
    """간단한 NMS – confidence 내림차순 greedy."""
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for d in dets:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        suppress = False
        for k in kept:
            ix1 = max(x1, k["x1"]); iy1 = max(y1, k["y1"])
            ix2 = min(x2, k["x2"]); iy2 = min(y2, k["y2"])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_d = (x2 - x1) * (y2 - y1)
            area_k = (k["x2"] - k["x1"]) * (k["y2"] - k["y1"])
            iou = inter / (area_d + area_k - inter + 1e-6)
            if iou > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(d)
    return kept


def _draw_results(image_path: str, detections: list, output_path: str):
    """탐지 결과를 이미지에 그려 저장."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # 폰트 로드 (없으면 기본 폰트)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = font_sm = ImageFont.load_default()

    class_list = list({d["class"] for d in detections})

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cls   = det["class"]
        conf  = det["confidence"]
        color = COLORS[class_list.index(cls) % len(COLORS)]

        # bbox 반투명 채우기
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 40), outline=color, width=2)

        # 마스크 오버레이 (있을 경우)
        if det.get("mask") is not None:
            mask = det["mask"]
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            ys, xs = np.where(mask)
            if len(xs) > 0:
                for px, py in zip(xs.tolist(), ys.tolist()):
                    overlay_draw.point((px, py), fill=(r, g, b, 80))
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")

        # 레이블
        label = f"{cls}  {conf:.2f}"
        bbox_text = draw.textbbox((x1, y1 - 18), label, font=font)
        draw.rectangle(bbox_text, fill=(r, g, b, 200))
        draw.text((x1, y1 - 18), label, fill="white", font=font)

    # 범례
    legend_x, legend_y = 10, 10
    for i, cls in enumerate(class_list):
        color = COLORS[i % len(COLORS)]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        draw.rectangle([legend_x, legend_y, legend_x + 12, legend_y + 12],
                       fill=(r, g, b, 220))
        draw.text((legend_x + 16, legend_y), cls, fill="white", font=font_sm)
        legend_y += 18

    img.save(output_path)
    logger.info(f"결과 이미지 저장: {output_path}")


# ── SAM3 탐지 ──────────────────────────────────────────────────────────────

def detect_with_sam3(image_path: str, classes: list, model_name: str,
                     confidence_thr: float, mask_score_thr: float,
                     nms_iou: float, sam3_path: str, device: str) -> list:
    """SAM3로 이미지 탐지. 결과 dict 리스트 반환."""

    # SAM3 경로 추가
    if sam3_path and Path(sam3_path).exists():
        if str(sam3_path) not in sys.path:
            sys.path.insert(0, str(sam3_path))
        logger.info(f"SAM3 경로 추가: {sam3_path}")

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        logger.error(f"SAM3 import 실패: {e}")
        logger.error("SAM3 설치 필요: pip install -e <sam3 경로>")
        sys.exit(1)

    import torch

    logger.info(f"SAM3 모델 로딩: {model_name} on {device}")
    model = build_sam3_image_model(model_name)
    model = model.to(device).eval()
    processor = Sam3Processor(model)
    logger.info("SAM3 모델 로딩 완료")

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    max_area = 0.9 * orig_w * orig_h

    all_detections = []

    for cls in classes:
        logger.info(f"  탐지 중: {cls}")
        try:
            state  = processor.set_image(img)
            output = processor.set_text_prompt(state=state, prompt=cls)

            masks_out  = output.get("masks",  [])
            boxes_out  = output.get("boxes",  [])
            scores_out = output.get("scores", [])

            if hasattr(masks_out,  "cpu"): masks_out  = masks_out.cpu().numpy()
            if hasattr(boxes_out,  "cpu"): boxes_out  = boxes_out.cpu().numpy()
            if hasattr(scores_out, "cpu"): scores_out = scores_out.cpu().numpy()

            masks_out  = np.asarray(masks_out)
            boxes_out  = np.asarray(boxes_out)
            scores_out = np.asarray(scores_out).flatten()

            for i, score in enumerate(scores_out):
                score = float(score)
                if score < mask_score_thr:
                    continue

                # bbox
                if boxes_out.ndim >= 2 and i < len(boxes_out):
                    x1, y1, x2, y2 = (float(v) for v in boxes_out[i])
                else:
                    continue

                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                if (x2 - x1) * (y2 - y1) > max_area:
                    continue

                # 마스크
                mask = None
                if masks_out.ndim >= 3 and i < len(masks_out):
                    raw = np.squeeze(masks_out[i]).astype(bool)
                    if raw.shape != (orig_h, orig_w):
                        pm = Image.fromarray(raw.astype(np.uint8) * 255, "L")
                        pm = pm.resize((orig_w, orig_h), Image.NEAREST)
                        raw = np.array(pm) > 127
                    mask = raw

                all_detections.append({
                    "class": cls, "confidence": score,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "mask": mask,
                })

        except Exception as e:
            logger.warning(f"  {cls} 탐지 실패: {e}")

    # NMS
    detections = _nms(all_detections, nms_iou)
    logger.info(f"탐지 완료: {len(all_detections)}개 → NMS 후 {len(detections)}개")
    return detections


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM3 단일 이미지 객체탐지")
    parser.add_argument("--image",      required=True,  help="입력 이미지 경로")
    parser.add_argument("--output",     default=None,   help="결과 이미지 저장 경로 (기본: <입력명>_detected.jpg)")
    parser.add_argument("--classes",    nargs="+",      default=DEFAULT_CLASSES, help="탐지할 클래스 목록")
    parser.add_argument("--model",      default=DEFAULT_MODEL_NAME, help="SAM3 모델명")
    parser.add_argument("--sam3-path",  default=str(DEFAULT_SAM3_PATH), help="SAM3 라이브러리 경로")
    parser.add_argument("--device",     default="cuda", choices=["cuda", "cpu"], help="추론 디바이스")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="신뢰도 임계값")
    parser.add_argument("--mask-score", type=float, default=DEFAULT_MASK_SCORE,  help="마스크 점수 임계값")
    parser.add_argument("--nms-iou",    type=float, default=DEFAULT_NMS_IOU,     help="NMS IoU 임계값")
    args = parser.parse_args()

    # 입력 파일 확인
    if not Path(args.image).exists():
        logger.error(f"이미지 파일 없음: {args.image}")
        sys.exit(1)

    # 출력 경로 결정
    if args.output is None:
        p = Path(args.image)
        args.output = str(p.parent / f"{p.stem}_detected{p.suffix}")

    # CUDA 사용 가능 여부 확인
    try:
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 사용 불가 → CPU로 전환")
            args.device = "cpu"
    except ImportError:
        args.device = "cpu"

    logger.info(f"입력: {args.image}")
    logger.info(f"클래스: {args.classes}")
    logger.info(f"디바이스: {args.device}")

    detections = detect_with_sam3(
        image_path=args.image,
        classes=args.classes,
        model_name=args.model,
        confidence_thr=args.confidence,
        mask_score_thr=args.mask_score,
        nms_iou=args.nms_iou,
        sam3_path=args.sam3_path,
        device=args.device,
    )

    # 결과 출력
    print(f"\n{'─'*60}")
    print(f"탐지 결과: {len(detections)}개 객체")
    print(f"{'─'*60}")
    for i, d in enumerate(detections, 1):
        print(f"  [{i:02d}] {d['class']:<30}  conf={d['confidence']:.3f}"
              f"  bbox=({d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f})")
    print(f"{'─'*60}\n")

    _draw_results(args.image, detections, args.output)


if __name__ == "__main__":
    main()

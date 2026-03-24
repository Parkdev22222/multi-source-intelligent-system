"""
Super-Resolution preprocessing for satellite/aerial imagery.

업스케일 우선순위:
  1. Real-ESRGAN (basicsr + realesrgan 패키지 설치된 경우)
  2. PIL LANCZOS (폴백)

목표 해상도: config.SR_TARGET_W × SR_TARGET_H (기본 8000×6000).
이미지 비율을 유지하며 목표 크기 내에서 최대한 확대.
이미 목표 크기 이상이면 SR 없이 원본 반환.
"""

import logging

import numpy as np
from PIL import Image

from src.config import SR_TARGET_H, SR_TARGET_W

logger = logging.getLogger(__name__)


def _sr_output_size(w: int, h: int) -> tuple[int, int]:
    """비율 유지하며 SR_TARGET 내 최대 크기 계산. 이미 크면 원본 그대로."""
    scale = min(SR_TARGET_W / w, SR_TARGET_H / h)
    if scale <= 1.0:
        return w, h
    return int(w * scale), int(h * scale)


def _upscale_realesrgan(image_np: np.ndarray, esrgan_scale: int) -> np.ndarray:
    """Real-ESRGAN x{esrgan_scale} 업스케일."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=esrgan_scale,
    )
    model_url = (
        f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
        f"RealESRGAN_x{esrgan_scale}plus.pth"
    )
    upsampler = RealESRGANer(
        scale=esrgan_scale,
        model_path=model_url,
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )
    out, _ = upsampler.enhance(image_np, outscale=esrgan_scale)
    return out


def super_resolve(image_np: np.ndarray) -> np.ndarray:
    """
    위성/항공 이미지를 SR_TARGET(8000×6000) 기준으로 업스케일.

    Args:
        image_np: H×W×3 uint8 RGB numpy array

    Returns:
        업스케일된 H×W×3 uint8 RGB numpy array
    """
    h, w = image_np.shape[:2]
    target_w, target_h = _sr_output_size(w, h)

    if target_w == w and target_h == h:
        logger.debug(f"[SR] 이미지({w}×{h}) 이미 목표 크기 이상, SR 건너뜀.")
        return image_np

    scale_needed = max(target_w / w, target_h / h)
    esrgan_scale = 4 if scale_needed > 2.0 else 2

    # ── Real-ESRGAN 시도 ────────────────────────────────────────────────
    try:
        import basicsr    # noqa: F401
        import realesrgan # noqa: F401

        logger.info(
            f"[SR] Real-ESRGAN x{esrgan_scale} 적용: "
            f"{w}×{h} → ~{w * esrgan_scale}×{h * esrgan_scale}"
        )
        sr_np = _upscale_realesrgan(image_np, esrgan_scale)

        # SR 후 정확히 target 크기로 맞추기 (소수점 오차 보정)
        sr_h, sr_w = sr_np.shape[:2]
        if (sr_w, sr_h) != (target_w, target_h):
            pil = Image.fromarray(sr_np)
            pil = pil.resize((target_w, target_h), Image.LANCZOS)
            sr_np = np.array(pil, dtype=np.uint8)

        logger.info(f"[SR] Real-ESRGAN 완료: {sr_np.shape[1]}×{sr_np.shape[0]}")
        return sr_np

    except (ImportError, Exception) as exc:
        logger.warning(f"[SR] Real-ESRGAN 사용 불가 ({exc}). PIL LANCZOS 폴백.")

    # ── PIL LANCZOS 폴백 ────────────────────────────────────────────────
    pil = Image.fromarray(image_np)
    pil = pil.resize((target_w, target_h), Image.LANCZOS)
    result = np.array(pil, dtype=np.uint8)
    logger.info(f"[SR] PIL LANCZOS 완료: {result.shape[1]}×{result.shape[0]}")
    return result

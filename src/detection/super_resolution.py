"""
Super-Resolution preprocessing for satellite/aerial imagery.

업스케일 우선순위 (SR_BACKEND 환경변수로 선택):
  1. EDSR     (basicsr 패키지 — 한국 SNU, 2017 NTIRE SR 1위, 기본값)
  2. Real-ESRGAN (basicsr + realesrgan 패키지 — SR_BACKEND="realesrgan" 선택 시)
  3. PIL LANCZOS (폴백 — 두 모델 모두 사용 불가 시)

목표 해상도: config.SR_TARGET_W × SR_TARGET_H (기본 8000×6000).
이미지 비율을 유지하며 목표 크기 내에서 최대한 확대.
이미 목표 크기 이상이면 SR 없이 원본 반환.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import (
    SR_TARGET_H, SR_TARGET_W,
    SR_BACKEND,
    EDSR_X4_PATH, EDSR_X2_PATH,
    REALESRGAN_X4_PATH, REALESRGAN_X2_PATH,
)

logger = logging.getLogger(__name__)


def _sr_output_size(w: int, h: int) -> tuple[int, int]:
    """비율 유지하며 SR_TARGET 내 최대 크기 계산. 이미 크면 원본 그대로."""
    scale = min(SR_TARGET_W / w, SR_TARGET_H / h)
    if scale <= 1.0:
        return w, h
    return int(w * scale), int(h * scale)


def _upscale_edsr(image_np: np.ndarray, scale: int) -> np.ndarray:
    """EDSR x{scale} 업스케일 (basicsr, 한국 SNU 모델, 로컬 가중치 파일 사용).

    EDSR-L (Large) 설정: num_feat=256, num_block=32 — DIV2K 공식 가중치와 일치.
    """
    import torch
    from basicsr.archs.edsr_arch import EDSR

    model_path = EDSR_X4_PATH if scale == 4 else EDSR_X2_PATH

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"EDSR 가중치 파일을 찾을 수 없습니다: {model_path}\n"
            f"다운로드 명령:\n"
            f"  wget -P {Path(model_path).parent} "
            f"https://github.com/XPixelGroup/BasicSR/releases/download/V1.1/"
            f"EDSR_Lx{scale}_f256b32_DIV2K_official-"
            f"{'76ee1c8f' if scale == 4 else 'be38e77d'}.pth"
        )

    model = EDSR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=256,
        num_block=32,
        upscale=scale,
        res_scale=0.1,
        img_range=255.0,
        rgb_mean=[0.4488, 0.4371, 0.4040],
    )

    state = torch.load(model_path, map_location="cpu")
    # basicsr 가중치는 'params_ema' 또는 'params' 키 아래에 있을 수 있음
    if isinstance(state, dict):
        state = state.get("params_ema", state.get("params", state))
    model.load_state_dict(state, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # numpy uint8 → float32 tensor [1, 3, H, W], 0–1 정규화
    img_t = (
        torch.from_numpy(image_np.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        out_t = model(img_t)

    # [1, 3, H, W] → numpy uint8
    out_np = (
        out_t.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .cpu()
        .numpy()
    )
    return out_np


def _upscale_realesrgan(image_np: np.ndarray, esrgan_scale: int) -> np.ndarray:
    """Real-ESRGAN x{esrgan_scale} 업스케일 (로컬 가중치 파일 사용)."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model_path = REALESRGAN_X4_PATH if esrgan_scale == 4 else REALESRGAN_X2_PATH

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Real-ESRGAN 가중치 파일을 찾을 수 없습니다: {model_path}\n"
            f"다운로드 명령:\n"
            f"  wget -P {Path(model_path).parent} "
            f"https://github.com/xinntao/Real-ESRGAN/releases/download/"
            f"{'v0.1.0' if esrgan_scale == 4 else 'v0.2.1'}/"
            f"RealESRGAN_x{esrgan_scale}plus.pth"
        )

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=esrgan_scale,
    )
    upsampler = RealESRGANer(
        scale=esrgan_scale,
        model_path=model_path,
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

    SR_BACKEND 환경변수(config.SR_BACKEND)로 백엔드를 선택합니다.
      "edsr"       → EDSR (한국 SNU, basicsr 필요) [기본값]
      "realesrgan" → Real-ESRGAN (basicsr + realesrgan 필요)
      "lanczos"    → PIL LANCZOS 강제 사용 (SR 모델 없이 빠른 폴백)

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
    sr_scale = 4 if scale_needed > 2.0 else 2

    backend = SR_BACKEND.lower().strip()

    # ── EDSR (기본값, 한국 SNU) ─────────────────────────────────────────────
    if backend == "edsr":
        try:
            import basicsr  # noqa: F401

            logger.info(
                f"[SR] EDSR x{sr_scale} 적용 (한국 SNU): "
                f"{w}×{h} → ~{w * sr_scale}×{h * sr_scale}"
            )
            sr_np = _upscale_edsr(image_np, sr_scale)

            # SR 후 정확히 target 크기로 맞추기 (소수점 오차 보정)
            sr_h, sr_w = sr_np.shape[:2]
            if (sr_w, sr_h) != (target_w, target_h):
                pil = Image.fromarray(sr_np)
                pil = pil.resize((target_w, target_h), Image.LANCZOS)
                sr_np = np.array(pil, dtype=np.uint8)

            logger.info(f"[SR] EDSR 완료: {sr_np.shape[1]}×{sr_np.shape[0]}")
            return sr_np

        except (ImportError, Exception) as exc:
            logger.warning(f"[SR] EDSR 사용 불가 ({exc}). PIL LANCZOS 폴백.")

    # ── Real-ESRGAN ──────────────────────────────────────────────────────────
    elif backend == "realesrgan":
        try:
            import basicsr    # noqa: F401
            import realesrgan # noqa: F401

            logger.info(
                f"[SR] Real-ESRGAN x{sr_scale} 적용: "
                f"{w}×{h} → ~{w * sr_scale}×{h * sr_scale}"
            )
            sr_np = _upscale_realesrgan(image_np, sr_scale)

            sr_h, sr_w = sr_np.shape[:2]
            if (sr_w, sr_h) != (target_w, target_h):
                pil = Image.fromarray(sr_np)
                pil = pil.resize((target_w, target_h), Image.LANCZOS)
                sr_np = np.array(pil, dtype=np.uint8)

            logger.info(f"[SR] Real-ESRGAN 완료: {sr_np.shape[1]}×{sr_np.shape[0]}")
            return sr_np

        except (ImportError, Exception) as exc:
            logger.warning(f"[SR] Real-ESRGAN 사용 불가 ({exc}). PIL LANCZOS 폴백.")

    elif backend != "lanczos":
        logger.warning(f"[SR] 알 수 없는 SR_BACKEND='{SR_BACKEND}'. PIL LANCZOS 사용.")

    # ── PIL LANCZOS 폴백 ─────────────────────────────────────────────────────
    pil = Image.fromarray(image_np)
    pil = pil.resize((target_w, target_h), Image.LANCZOS)
    result = np.array(pil, dtype=np.uint8)
    logger.info(f"[SR] PIL LANCZOS 완료: {result.shape[1]}×{result.shape[0]}")
    return result

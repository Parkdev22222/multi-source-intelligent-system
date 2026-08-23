"""
Run SAM3 (+ CLIP embeddings) over a benchmark split once and cache the result.

    python -m icce.eval.cache_detections --dataset levir_cd --split train \
        --out data/cache/levir_cd_train --limit 445

This is the only GPU-heavy stage of the detection experiments. Everything after
it -- head training, threshold sweeps, assignment ablations, report grounding --
reads the cache, so the expensive pass happens exactly once per split.

Self-supervised labels are written at cache time: `coverage` is the fraction of
each detection's mask that falls inside the ground-truth change mask. That
single number is what supervises both the matcher (stable objects have low
coverage in both frames) and the change verifier.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# image + mask IO
# ---------------------------------------------------------------------------
def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def load_mask(path: Optional[Path], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if path is None or not Path(path).is_file():
        return None
    from PIL import Image
    with Image.open(path) as im:
        m = np.array(im.convert("L"))
    if m.shape != shape:
        from PIL import Image as I
        m = np.array(I.fromarray(m).resize((shape[1], shape[0]), I.NEAREST))
    return m > 127


def decode_mask(mask_rle: Optional[str], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Decode the pipeline's RLE mask; returns None when unavailable."""
    if not mask_rle:
        return None
    try:
        from src.pairing.temporal_pairing import _decode_rle
        m = _decode_rle(mask_rle)
        return m.astype(bool) if m is not None and m.shape == shape else None
    except Exception:
        return None


def coverage_of(
    det_mask: Optional[np.ndarray],
    bbox: Sequence[float],
    change_mask: Optional[np.ndarray],
) -> float:
    """Fraction of the detection inside the annotated change region."""
    if change_mask is None:
        return 0.0
    if det_mask is not None and det_mask.shape == change_mask.shape:
        area = int(det_mask.sum())
        return float((det_mask & change_mask).sum() / area) if area else 0.0

    h, w = change_mask.shape
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = change_mask[y1:y2, x1:x2]
    return float(patch.mean())


# ---------------------------------------------------------------------------
# CLIP embeddings of mask-cropped objects
# ---------------------------------------------------------------------------
class ClipEmbedder:
    """Mirrors the production `_CLIPEmbedder`: background masked out, then CLIP."""

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._proc = None

    def _load(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        logger.info("loading CLIP %s", self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self._proc = CLIPProcessor.from_pretrained(self.model_name)

    def embed(self, image: np.ndarray, dets, masks) -> np.ndarray:
        if not dets:
            return np.zeros((0, 512), np.float32)
        if self._model is None:
            self._load()

        import torch
        from PIL import Image as PILImage

        crops = []
        for d, m in zip(dets, masks):
            x1, y1, x2, y2 = (int(round(d.bbox_x1)), int(round(d.bbox_y1)),
                              int(round(d.bbox_x2)), int(round(d.bbox_y2)))
            h, w = image.shape[:2]
            x1, x2 = max(0, min(w - 1, x1)), max(1, min(w, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(1, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                crops.append(PILImage.new("RGB", (8, 8)))
                continue
            patch = image[y1:y2, x1:x2].copy()
            if m is not None and m.shape == image.shape[:2]:
                patch[~m[y1:y2, x1:x2]] = 0        # background suppressed
            crops.append(PILImage.fromarray(patch))

        with torch.no_grad():
            inputs = self._proc(images=crops, return_tensors="pt").to(self.device)
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return feats.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cache SAM3 detections for a benchmark split")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-gt-area", type=int, default=32,
                    help="ignore GT change blobs smaller than this many pixels")
    ap.add_argument("--cache-masks", action="store_true", default=True,
                    help="store SAM3 instance masks (RLE) for pixel-level scoring")
    ap.add_argument("--no-cache-masks", dest="cache_masks", action="store_false")
    ap.add_argument("--no-clip", action="store_true",
                    help="skip CLIP embeddings (geometry-only ablation)")
    ap.add_argument("--attach-cd-masks", action="store_true",
                    help="LEVIR-CC only: borrow LEVIR-CD masks for the same tiles")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from src.config import CLIP_MODEL_NAME, OBJECT_CLASSES
    from src.detection.image_loader import ImageMeta, LoadedImage
    from src.detection.sam3_detector import SAM3Detector
    from icce.convert.mask_to_instances import connected_components
    from icce.convert.to_msis import build_scenes
    from icce.datasets.registry import GSD, load as load_dataset
    from icce.pairing_head.cache import CacheWriter, CachedDet, CachedSample

    pairs = load_dataset(args.dataset, split=args.split, limit=args.limit)
    if args.attach_cd_masks:
        from icce.datasets.levir_cc import attach_cd_masks
        n = attach_cd_masks(pairs)
        logger.info("attached LEVIR-CD masks to %d/%d LEVIR-CC pairs", n, len(pairs))

    gsd = GSD.get(args.dataset.replace("-", "_"), 0.5)
    scenes = build_scenes(pairs, gsd, Path("."))
    by_id = {p.pair_id: p for p in pairs}
    logger.info("%s/%s: %d pairs, gsd=%.3f m", args.dataset, args.split, len(scenes), gsd)

    class_index = {c: i for i, c in enumerate(OBJECT_CLASSES)}
    detector = SAM3Detector()
    embedder = None if args.no_clip else ClipEmbedder(CLIP_MODEL_NAME, args.device)
    writer = CacheWriter(args.out)

    t_start = time.time()
    timings: List[float] = []

    for n, scene in enumerate(scenes, 1):
        pair = by_id[scene.pair_id]
        t0 = time.time()
        try:
            img_a, img_b = load_rgb(pair.image_a), load_rgb(pair.image_b)
        except Exception as exc:
            logger.warning("%s: unreadable images (%s), skipped", scene.pair_id, exc)
            continue

        h, w = img_a.shape[:2]
        gt_mask = load_mask(pair.mask, (h, w))

        cached: Dict[str, List[CachedDet]] = {}
        embeds: Dict[str, np.ndarray] = {}

        for phase, img, capture in (("past", img_a, scene.past_time),
                                    ("cur", img_b, scene.current_time)):
            meta = ImageMeta(
                image_path=str(pair.image_a if phase == "past" else pair.image_b),
                capture_time=capture, source_type="satellite",
                lat_center=(scene.grid.lat_min + scene.grid.lat_max) / 2,
                lon_center=(scene.grid.lon_min + scene.grid.lon_max) / 2,
                lat_min=scene.grid.lat_min, lat_max=scene.grid.lat_max,
                lon_min=scene.grid.lon_min, lon_max=scene.grid.lon_max,
                resolution_m=gsd, sensor_platform="BENCHMARK",
            )
            image_id = f"{scene.pair_id}::{phase}"
            dets = detector.detect(LoadedImage(meta=meta, array=img), image_id)
            masks = [decode_mask(d.mask_rle, (h, w)) for d in dets]

            rows: List[CachedDet] = []
            for d, m in zip(dets, masks):
                bbox = [d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2]
                lat, lon = scene.grid.centre_of(bbox)
                rows.append(CachedDet(
                    det_id=d.detection_id,
                    object_class=d.object_class,
                    class_id=class_index.get(d.object_class, len(class_index)),
                    confidence=float(d.confidence),
                    bbox_px=[float(v) for v in bbox],
                    lat=lat, lon=lon,
                    geo_bbox=list(scene.grid.bbox_to_geo(bbox)),
                    mask_area=int(d.mask_area_px) if d.mask_area_px else None,
                    coverage=coverage_of(m, bbox, gt_mask),
                    mask_rle=d.mask_rle if args.cache_masks else None,
                ))
            cached[phase] = rows
            if embedder is not None:
                embeds[phase] = embedder.embed(img, dets, masks)

        gt_instances = ([list(c["bbox"]) for c in connected_components(gt_mask, args.min_gt_area)]
                        if gt_mask is not None else [])

        writer.add(
            CachedSample(
                pair_id=scene.pair_id, dataset=pair.dataset, split=args.split,
                image_size=(w, h), past=cached["past"], current=cached["cur"],
                gt_instances=gt_instances,
                gt_change_present=(pair.change_flag if pair.change_flag is not None
                                   else (bool(gt_instances) if gt_mask is not None else None)),
                captions=list(pair.captions),
                parent_scene=scene.parent_scene,
            ),
            embeds.get("past"), embeds.get("cur"),
        )

        timings.append(time.time() - t0)
        if n % 20 == 0 or n == len(scenes):
            rate = float(np.mean(timings[-20:]))
            eta = rate * (len(scenes) - n) / 60.0
            logger.info("%d/%d cached (%.2fs/pair, ETA %.1f min)", n, len(scenes), rate, eta)

    writer.close()
    (Path(args.out) / "cache_info.json").write_text(json.dumps({
        "dataset": args.dataset, "split": args.split, "n_pairs": writer.n,
        "gsd_m": gsd, "clip": not args.no_clip,
        "object_classes": OBJECT_CLASSES,
        "seconds_per_pair_mean": float(np.mean(timings)) if timings else None,
        "seconds_per_pair_p95": float(np.percentile(timings, 95)) if timings else None,
        "total_seconds": time.time() - t_start,
    }, indent=2), encoding="utf-8")

    logger.info("done: %d pairs in %.1f min", writer.n, (time.time() - t_start) / 60.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

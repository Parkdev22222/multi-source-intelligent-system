"""
Vision-language baseline: show a VLM both images and ask what changed.

This is the comparison that matters. A model trained on LEVIR-CC's own training
captions is a specialist with in-domain supervision and will win on n-gram
overlap; that tells us nothing about whether our design is a good one. The
honest question is whether routing imagery through explicit detection, pairing
and a knowledge graph beats simply handing both pictures to a capable VLM --
because "just use a VLM" is what a practitioner would actually try first.

It is also the comparison we have reason to win on factuality rather than
fluency. A VLM reading 256x256 aerial tiles has no instance-level grounding: it
cannot count roofs reliably, and when it is unsure it produces confident,
well-formed prose anyway. Change-Fact-Score is built to catch exactly that.

Run with a multimodal model served by vLLM, e.g.
    --vlm Qwen/Qwen2.5-VL-7B-Instruct
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

VLM_SYSTEM = (
    "You are an urban change-monitoring analyst. You are shown two satellite "
    "images of the same place: the first was taken earlier, the second later.\n"
    "Describe what changed between them in ONE or TWO short sentences.\n"
    "Rules:\n"
    "  - Describe only changes you can actually see. Do not speculate.\n"
    "  - If nothing changed, say the scene is the same as before.\n"
    "  - No preamble, no bullet points, no headings. Output the sentence only."
)


def _data_uri(path: Path, max_side: int = 512) -> str:
    """Encode an image as a data URI, downscaled if very large."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_side:
            scale = max_side / max(im.size)
            im = im.resize((int(im.width * scale), int(im.height * scale)),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class VlmCaptioner:
    """Batched two-image captioning through vLLM's chat API."""

    def __init__(
        self,
        model: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        temperature: float = 0.0,
        max_images: int = 2,
        image_side: int = 512,
    ) -> None:
        self.name = model
        self.temperature = temperature
        self.image_side = image_side
        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        from vllm import LLM

        logger.info("loading VLM %s", model)
        self._llm = LLM(
            model=model,
            dtype="auto",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": max_images},
            trust_remote_code=True,
        )

    def _conversation(self, path_a: Path, path_b: Path) -> List[Dict]:
        return [
            {"role": "system", "content": VLM_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Earlier image:"},
                {"type": "image_url",
                 "image_url": {"url": _data_uri(path_a, self.image_side)}},
                {"type": "text", "text": "Later image:"},
                {"type": "image_url",
                 "image_url": {"url": _data_uri(path_b, self.image_side)}},
                {"type": "text",
                 "text": "In one or two sentences, what changed between them?"},
            ]},
        ]

    def caption_batch(
        self,
        pairs: Sequence[tuple],          # (pair_id, path_a, path_b)
        max_tokens: int = 96,
    ) -> Dict[str, str]:
        from vllm import SamplingParams

        usable, skipped = [], 0
        for pid, pa, pb in pairs:
            if pa and pb and Path(pa).is_file() and Path(pb).is_file():
                usable.append((pid, Path(pa), Path(pb)))
            else:
                skipped += 1
        if skipped:
            logger.warning("%d pairs skipped: image files not found on disk", skipped)
        if not usable:
            return {}

        convs = [self._conversation(pa, pb) for _, pa, pb in usable]
        params = SamplingParams(temperature=self.temperature, max_tokens=max_tokens)
        outputs = self._llm.chat(convs, sampling_params=params)
        return {pid: o.outputs[0].text.strip()
                for (pid, _, _), o in zip(usable, outputs)}

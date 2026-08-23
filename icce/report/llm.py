"""
Batched LLM wrapper for the report experiments.

The production reporter generates one report per pipeline run; the paper needs
~5 grounding conditions x ~1000 LEVIR-CC test pairs x 2 model sizes. Serial
`llm.chat()` calls would take days, so this wrapper drives vLLM's continuous
batching directly and streams results to disk as they land, letting a run be
resumed after a pod restart.

Decoding is greedy (temperature 0) everywhere: the ablation must measure the
grounding, not sampling noise.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GenRequest:
    key: str                 # unique id, used for resume
    system: str
    user: str
    max_tokens: int = 512


class BaseLLM:
    name = "base"

    def generate_batch(self, requests: Sequence[GenRequest]) -> List[str]:
        raise NotImplementedError

    def unload(self) -> None:
        pass


class VllmLLM(BaseLLM):
    """vLLM with continuous batching -- the workhorse for the RunPod runs."""

    def __init__(
        self,
        model: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        temperature: float = 0.0,
        dtype: str = "auto",
    ) -> None:
        self.name = model
        self.temperature = temperature
        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        from vllm import LLM

        logger.info("loading %s via vLLM", model)
        t0 = time.time()
        self._llm = LLM(
            model=model,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        self.load_seconds = time.time() - t0
        logger.info("loaded %s in %.1fs", model, self.load_seconds)

    def generate_batch(self, requests: Sequence[GenRequest]) -> List[str]:
        from vllm import SamplingParams

        if not requests:
            return []
        conversations = [
            [{"role": "system", "content": r.system},
             {"role": "user", "content": r.user}]
            for r in requests
        ]
        params = [
            SamplingParams(temperature=self.temperature, max_tokens=r.max_tokens)
            for r in requests
        ]
        outputs = self._llm.chat(conversations, sampling_params=params)
        return [o.outputs[0].text.strip() for o in outputs]

    def unload(self) -> None:
        try:
            import gc

            import torch
            del self._llm
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


class OllamaLLM(BaseLLM):
    """Serial fallback for a workstation without vLLM."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 temperature: float = 0.0) -> None:
        self.name = model
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def generate_batch(self, requests: Sequence[GenRequest]) -> List[str]:
        import urllib.request

        out = []
        for r in requests:
            payload = json.dumps({
                "model": self.model,
                "messages": [{"role": "system", "content": r.system},
                             {"role": "user", "content": r.user}],
                "stream": False,
                "options": {"temperature": self.temperature, "num_predict": r.max_tokens},
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/chat", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                out.append(json.loads(resp.read().decode())["message"]["content"].strip())
        return out


class EchoLLM(BaseLLM):
    """Deterministic stand-in so the harness is testable without a GPU.

    Returns the change inventory already embedded in the prompt. That makes it a
    genuine (weak) system rather than a mock: it grounds perfectly on whatever
    the prompt states and never adds anything, which is a useful lower bound
    for the hallucination column.
    """

    name = "echo"

    def generate_batch(self, requests: Sequence[GenRequest]) -> List[str]:
        out = []
        for r in requests:
            lines = [l.strip() for l in r.user.splitlines()
                     if l.strip().startswith(("- ", "* ")) or "CHANGE_" in l]
            out.append(" ".join(lines[:6]) if lines else "The scene is the same as before.")
        return out


def build_llm(spec: str, **kw) -> BaseLLM:
    """`spec` is 'echo', 'ollama:<model>' or a HuggingFace model id for vLLM."""
    if spec == "echo":
        return EchoLLM()
    if spec.startswith("ollama:"):
        return OllamaLLM(spec.split(":", 1)[1], **{k: v for k, v in kw.items()
                                                   if k in ("base_url", "temperature")})
    allowed = {"max_model_len", "gpu_memory_utilization", "tensor_parallel_size",
               "temperature", "dtype"}
    return VllmLLM(spec, **{k: v for k, v in kw.items() if k in allowed})


# ---------------------------------------------------------------------------
# resumable batched generation
# ---------------------------------------------------------------------------
def generate_with_cache(
    llm: BaseLLM,
    requests: Sequence[GenRequest],
    out_path: Path,
    batch_size: int = 128,
) -> Dict[str, str]:
    """Generate, skipping keys already present in `out_path` (JSONL).

    A LEVIR-CC sweep is long enough that pod pre-emption is a real risk; this
    makes a restart cost only the current batch.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: Dict[str, str] = {}
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done[rec["key"]] = rec["text"]
            except (json.JSONDecodeError, KeyError):
                continue
        logger.info("resuming: %d/%d already generated", len(done), len(requests))

    todo = [r for r in requests if r.key not in done]
    with out_path.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            t0 = time.time()
            texts = llm.generate_batch(chunk)
            for r, t in zip(chunk, texts):
                done[r.key] = t
                fh.write(json.dumps({"key": r.key, "text": t}, ensure_ascii=False) + "\n")
            fh.flush()
            logger.info("generated %d/%d (%.1fs for %d)",
                        min(start + batch_size, len(todo)), len(todo),
                        time.time() - t0, len(chunk))
    return done

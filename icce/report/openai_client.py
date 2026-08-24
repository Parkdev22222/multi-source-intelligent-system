"""
Minimal OpenAI-compatible chat client for a standalone vLLM server.

Why a server instead of an in-process ``vllm.LLM``
--------------------------------------------------
The decisive reason is not latency, it is dependency isolation. vLLM pulls
torch>=2.13, numpy>=2 and transformers>=5; SAM3 requires numpy<2 and this
project pins transformers<5. The two cannot share one interpreter. Running
vLLM behind HTTP lets the detection side keep torch 2.8 / numpy 1.26 while the
generation side runs whatever it wants.

Two further wins fall out of it:

  * ``scripts/icce_runpod.sh`` starts a fresh Python process five times
    (three ``run_report_eval`` calls, one report stage, one efficiency stage).
    In-process, each start reloads the 32B model. Against a server the weights
    are loaded once for the whole sweep.
  * ``VlmCaptioner`` and ``VllmLLM`` both ask for gpu_memory_utilization=0.85,
    so they cannot coexist in one process. As separate servers -- or one
    server restarted between stages -- that conflict disappears.

Raw token throughput is *not* expected to improve: vLLM does continuous
batching either way, and HTTP adds a little overhead. What improves is that
the model is loaded once rather than five times, and that requests are issued
concurrently so the server's scheduler always has a full queue.

Deliberately stdlib-only (urllib + concurrent.futures): adding the ``openai``
package to the detection environment is exactly the coupling this module
exists to avoid.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")

# Transient conditions worth another attempt: the server is still warming up,
# the scheduler queue is full, or a proxy dropped the connection.
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ServerError(RuntimeError):
    pass


def parse_spec(spec: str, default_base_url: str = DEFAULT_BASE_URL):
    """``server:<model>`` or ``server:<model>@<base_url>`` -> (model, base_url).

    The ``server:`` prefix is assumed to have been stripped or not; both forms
    are accepted so callers can pass the raw CLI value through.
    """
    if spec.startswith("server:"):
        spec = spec[len("server:"):]
    if "@" in spec:
        model, base_url = spec.rsplit("@", 1)
        return model.strip(), base_url.strip().rstrip("/")
    return spec.strip(), default_base_url.rstrip("/")


class OpenAIChatClient:
    """Concurrent client against ``/v1/chat/completions``.

    ``concurrency`` is how many requests are in flight at once, not a batch
    size: vLLM's scheduler merges whatever has arrived into one running batch,
    so the job here is simply to keep the queue non-empty.
    """

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        concurrency: int = 32,
        timeout: int = 600,
        retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("VLLM_API_KEY") or "EMPTY"
        self.temperature = temperature
        self.concurrency = max(1, concurrency)
        self.timeout = timeout
        self.retries = max(1, retries)

    # -- plumbing ---------------------------------------------------------
    def _post(self, path: str, payload: Dict) -> Dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _one(self, messages: List[Dict], max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                data = self._post("/chat/completions", payload)
                return (data["choices"][0]["message"]["content"] or "").strip()
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in _RETRY_STATUS:
                    body = exc.read().decode("utf-8", "replace")[:400]
                    raise ServerError(f"HTTP {exc.code} from {self.base_url}: {body}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                last = exc
            if attempt < self.retries - 1:
                time.sleep(2 ** attempt)
        raise ServerError(f"{self.base_url} failed after {self.retries} attempts: {last}")

    # -- public API -------------------------------------------------------
    def chat_many(
        self,
        conversations: Sequence[List[Dict]],
        max_tokens: Sequence[int] | int = 512,
    ) -> List[str]:
        """Run conversations concurrently, returning texts in input order."""
        if not conversations:
            return []
        if isinstance(max_tokens, int):
            max_tokens = [max_tokens] * len(conversations)
        if len(max_tokens) != len(conversations):
            raise ValueError("max_tokens length must match conversations")

        workers = min(self.concurrency, len(conversations))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # executor.map preserves input order and re-raises in order.
            return list(pool.map(self._one, conversations, max_tokens))

    def wait_until_ready(self, timeout: int = 1800, poll: float = 3.0) -> str:
        """Block until ``/models`` answers; returns the served model id.

        A 32B checkpoint can take minutes to load, so the caller should not
        have to guess how long to sleep before the first request.
        """
        deadline = time.time() + timeout
        url = f"{self.base_url}/models"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.api_key}"})
        last: Optional[Exception] = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                served = [m["id"] for m in data.get("data", [])]
                if served:
                    logger.info("vLLM server ready at %s (serving %s)",
                                self.base_url, ", ".join(served))
                    return served[0]
            except Exception as exc:      # not up yet, keep waiting
                last = exc
            time.sleep(poll)
        raise ServerError(f"{url} not ready within {timeout}s (last error: {last})")

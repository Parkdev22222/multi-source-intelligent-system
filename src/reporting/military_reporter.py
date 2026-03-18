"""
Military Change Detection Report Generator using EXAONE4-32b.

Input:  Latest pairing records from the Pairing DB.
Output: Structured military intelligence report (Korean/English) covering:
         - Observation summary
         - Detected object inventory (current vs past)
         - Change analysis (new assets, disappeared assets, repositioned assets)
         - Threat assessment
         - Recommended actions

LLM backend can be switched via config:
  LLM_BACKEND=huggingface  → loads EXAONE4-32b via HuggingFace Transformers
  LLM_BACKEND=ollama       → calls local Ollama server (model: exaone4:32b)
"""

import json
import logging
import textwrap
from datetime import datetime, timezone
from typing import List, Optional

from src.config import (
    LLM_BACKEND,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from src.database.models import PairingRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    return textwrap.dedent("""
        You are an elite military intelligence analyst specializing in
        multi-source imagery intelligence (IMINT) and change detection analysis.
        You produce formal military intelligence reports based on automated
        AI-driven object detection results from satellite and drone imagery.

        Your reports are written in a concise, factual, and classified military
        format following standard intelligence product conventions (e.g., DIA/NGA
        product standards). Use clear section headers. Assess military significance.
        Highlight anomalies, force movements, and potential threat indicators.
        If the data indicates no significant change, state so explicitly.
    """).strip()


def _build_user_prompt(pairings: List[PairingRecord], report_time: datetime) -> str:
    """Serialise pairing records into a structured prompt for the LLM."""

    def fmt_dt(dt: Optional[datetime]) -> str:
        if dt is None:
            return "N/A"
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Summarise pairings by status
    new_objs = [p for p in pairings if p.status == "new"]
    matched_objs = [p for p in pairings if p.status == "matched"]
    disappeared_objs = [p for p in pairings if p.status == "disappeared"]

    # Region info
    lats = [p.lat_center for p in pairings]
    lons = [p.lon_center for p in pairings]
    lat_c = sum(lats) / len(lats) if lats else 0.0
    lon_c = sum(lons) / len(lons) if lons else 0.0

    # Capture time range
    all_times = [
        p.current_capture_time for p in pairings if p.current_capture_time
    ] + [
        p.past_capture_time for p in pairings if p.past_capture_time
    ]
    time_past = min(all_times).strftime("%Y-%m-%dT%H:%M:%SZ") if all_times else "UNKNOWN"
    time_current = max(all_times).strftime("%Y-%m-%dT%H:%M:%SZ") if all_times else "UNKNOWN"

    lines = [
        f"REPORT GENERATION TIME: {fmt_dt(report_time)}",
        f"REGION OF INTEREST: LAT {lat_c:.4f}, LON {lon_c:.4f}",
        f"TEMPORAL WINDOW: {time_past} → {time_current}",
        f"TOTAL PAIRING RECORDS: {len(pairings)}",
        "",
        "=== NEWLY APPEARED OBJECTS (not in past frame) ===",
    ]

    for p in new_objs:
        lines.append(
            f"  - ID={p.current_detection_id[:8]}  CLASS={p.current_object_class}"
            f"  CONF={p.current_confidence:.2f}"
            f"  LAT={p.current_lat:.4f}  LON={p.current_lon:.4f}"
            f"  TIME={fmt_dt(p.current_capture_time)}"
        )

    lines += ["", "=== MATCHED OBJECTS (present in both frames) ==="]
    for p in matched_objs:
        class_change = (
            f"  [CLASS CHANGED: {p.past_object_class} → {p.current_object_class}]"
            if p.past_object_class != p.current_object_class else ""
        )
        lat_delta = (
            abs(p.current_lat - p.past_lat)
            if p.current_lat is not None and p.past_lat is not None else 0
        )
        lon_delta = (
            abs(p.current_lon - p.past_lon)
            if p.current_lon is not None and p.past_lon is not None else 0
        )
        moved = "  [POSITION CHANGED]" if (lat_delta + lon_delta) > 0.0005 else ""
        lines.append(
            f"  - CUR_ID={p.current_detection_id[:8] if p.current_detection_id else 'N/A'}"
            f"  PAST_ID={p.past_detection_id[:8] if p.past_detection_id else 'N/A'}"
            f"  CLASS={p.current_object_class}"
            f"  CUR_LAT={p.current_lat:.4f}  CUR_LON={p.current_lon:.4f}"
            f"  PAST_LAT={p.past_lat:.4f}  PAST_LON={p.past_lon:.4f}"
            f"{class_change}{moved}"
        )

    lines += ["", "=== DISAPPEARED OBJECTS (in past frame, absent in current) ==="]
    for p in disappeared_objs:
        lines.append(
            f"  - ID={p.past_detection_id[:8] if p.past_detection_id else 'N/A'}"
            f"  CLASS={p.past_object_class}"
            f"  CONF={p.past_confidence:.2f}"
            f"  LAT={p.past_lat:.4f}  LON={p.past_lon:.4f}"
            f"  LAST_SEEN={fmt_dt(p.past_capture_time)}"
        )

    lines += [
        "",
        "=== INSTRUCTIONS ===",
        "Based on the above AI-generated object detection pairing data from satellite/drone imagery,",
        "produce a formal military intelligence report with the following sections:",
        "1. CLASSIFICATION / HANDLING INSTRUCTIONS",
        "2. EXECUTIVE SUMMARY",
        "3. SITUATION (current observed state)",
        "4. CHANGE ANALYSIS (new, disappeared, repositioned objects and their military significance)",
        "5. THREAT ASSESSMENT",
        "6. INTELLIGENCE GAPS",
        "7. RECOMMENDED ACTIONS",
        "8. APPENDIX: Full Object Inventory (table format)",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------------------------

class _HuggingFaceBackend:
    def __init__(self):
        self._model = None
        self._tokenizer = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"[Reporter] Loading {LLM_MODEL_NAME} via HuggingFace Transformers...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info(f"[Reporter] {LLM_MODEL_NAME} loaded.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._model is None:
            self._load()

        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # EXAONE uses standard chat template
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                do_sample=LLM_TEMPERATURE > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = output_ids[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)


class _OllamaBackend:
    """Calls a locally running Ollama server."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request

        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_MAX_NEW_TOKENS,
            },
        }).encode()

        url = f"{OLLAMA_BASE_URL}/api/chat"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        logger.info(f"[Reporter] Sending request to Ollama at {url} model={OLLAMA_MODEL}")
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode())
        return body["message"]["content"]


def _get_backend():
    if LLM_BACKEND == "ollama":
        return _OllamaBackend()
    return _HuggingFaceBackend()


# ---------------------------------------------------------------------------
# Main reporter class
# ---------------------------------------------------------------------------

class MilitaryReporter:
    """Generates military change-detection reports using EXAONE4-32b."""

    def __init__(self):
        self._backend = _get_backend()

    def generate_report(
        self,
        pairings: List[PairingRecord],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a military intelligence report from the given pairing records.

        Args:
            pairings:    List of PairingRecord objects (latest session).
            output_path: If provided, write the report text to this file path.

        Returns:
            Report text as a string.
        """
        if not pairings:
            logger.warning("[Reporter] No pairing records provided – returning empty report.")
            return "NO DATA: No pairing records available for report generation."

        report_time = datetime.now(tz=timezone.utc)
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(pairings, report_time)

        logger.info(f"[Reporter] Generating military report for {len(pairings)} pairings...")
        report_text = self._backend.generate(system_prompt, user_prompt)

        # Prepend metadata header
        header = (
            f"{'='*72}\n"
            f"  MILITARY INTELLIGENCE REPORT\n"
            f"  Generated by: Multi-Source Intelligent System (MSIS)\n"
            f"  Model: {LLM_MODEL_NAME}\n"
            f"  Generated: {report_time.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
            f"  Records analysed: {len(pairings)}\n"
            f"{'='*72}\n\n"
        )
        full_report = header + report_text

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_report)
            logger.info(f"[Reporter] Report written to {output_path}")

        return full_report

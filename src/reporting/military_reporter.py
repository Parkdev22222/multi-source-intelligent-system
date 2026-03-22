"""
Military Change Detection Report Generator using EXAONE4-32b.

Input:  Latest pairing records from the Pairing DB.
Output: Structured military intelligence report (Korean/English) covering:
         - Observation summary
         - Change analysis (new assets and disappeared assets only)
         - Threat assessment
         - Recommended actions
         NOTE: Stationary and repositioned/moved objects are excluded from the report.

LLM backend can be switched via config:
  LLM_BACKEND=vllm    → loads EXAONE4-32b via vLLM (default)
  LLM_BACKEND=ollama  → calls local Ollama server (model: exaone4:32b)
"""

import json
import logging
import textwrap
from datetime import datetime, timezone
from typing import List, Optional

from src.config import (
    LLM_BACKEND,
    LLM_GPU_MEMORY_UTILIZATION,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from src.database.models import PairingRecord
from src.database.reports_db import insert_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    return (
        "You are a military IMINT analyst. Produce a concise formal intelligence report "
        "from AI-based satellite/drone object detection data. "
        "Use standard section headers. Be factual. "
        "Focus exclusively on newly appeared and disappeared objects as indicators of activity. "
        "Do not analyse or comment on stationary or repositioned objects. "
        "IMPORTANT: 'DISAPPEARED' means the object was observed in the PAST imagery but was "
        "NOT detected in the CURRENT (most recent) imagery. It does NOT mean the object is "
        "confirmed destroyed or permanently gone — it may have moved outside the sensor FOV, "
        "be obscured, or relocated. Always qualify disappearance as 'no longer observed in "
        "current imagery' rather than implying confirmed destruction or elimination."
    )


_MAX_DETAIL = 20   # max individual records shown per change category


def _class_counts(objs) -> str:
    """Return 'TANK:3 APC:2 ...' summary string from a list of pairing records."""
    from collections import Counter
    counts = Counter(
        p.current_object_class if p.current_object_class else p.past_object_class
        for p in objs
    )
    return "  ".join(f"{cls}:{n}" for cls, n in counts.most_common())


def _build_user_prompt(pairings: List[PairingRecord]) -> str:
    """Serialise pairing records into a compact prompt for the LLM."""

    def fmt_dt(dt: Optional[datetime]) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else "N/A"

    new_objs = [p for p in pairings if p.status == "new"]
    disappeared_objs = [p for p in pairings if p.status == "disappeared"]

    # matched/moved 객체는 보고서에서 제외 — 집계 참고용으로만 카운트
    n_matched = sum(1 for p in pairings if p.status == "matched")
    n_moved   = sum(1 for p in pairings if p.status == "moved")

    # 현재 프레임 탐지 건수 = new + matched + moved (visualize_detections.py 기준과 동일)
    n_current_detections = len(new_objs) + n_matched + n_moved

    lats = [p.lat_center for p in pairings]
    lons = [p.lon_center for p in pairings]
    lat_c = sum(lats) / len(lats) if lats else 0.0
    lon_c = sum(lons) / len(lons) if lons else 0.0

    past_times = [p.past_capture_time for p in pairings if p.past_capture_time]
    current_times = [p.current_capture_time for p in pairings if p.current_capture_time]
    time_past = min(past_times).strftime("%Y-%m-%dT%H:%M:%SZ") if past_times else "UNKNOWN"
    time_current = max(current_times).strftime("%Y-%m-%dT%H:%M:%SZ") if current_times else "UNKNOWN"

    lines = [
        f"PAST_OBS: {time_past}  CURRENT_OBS: {time_current}  ROI: {lat_c:.3f},{lon_c:.3f}",
        f"CURRENT_FRAME_DETECTIONS: {n_current_detections}"
        f"  (NEW:{len(new_objs)}  STATIONARY:{n_matched}  MOVED:{n_moved})",
        f"PAST_ONLY (disappeared from current): {len(disappeared_objs)}",
        "NOTE: NEW = detected in CURRENT_OBS but absent in PAST_OBS.",
        "NOTE: DISAPPEARED = detected in PAST_OBS but NOT observed in CURRENT_OBS"
        " (location unknown — may have relocated or exited sensor coverage).",
        "NOTE: Report covers only NEW and DISAPPEARED objects.",
    ]

    # --- NEW objects (high-value: list top _MAX_DETAIL by confidence) ---
    lines.append(f"\n=== NEW OBJECTS (first observed at CURRENT_OBS: {time_current}) ===")
    top_new = sorted(new_objs, key=lambda p: p.current_confidence or 0, reverse=True)
    for p in top_new[:_MAX_DETAIL]:
        lines.append(
            f"  {p.current_object_class} CONF={p.current_confidence:.2f}"
            f" ({p.current_lat:.3f},{p.current_lon:.3f})"
            f" DETECTED={fmt_dt(p.current_capture_time)}"
        )
    if len(new_objs) > _MAX_DETAIL:
        lines.append(f"  ... +{len(new_objs) - _MAX_DETAIL} more: {_class_counts(new_objs[_MAX_DETAIL:])}")
    if not new_objs:
        lines.append("  (none)")

    # --- DISAPPEARED objects ---
    lines.append(
        f"\n=== DISAPPEARED OBJECTS"
        f" (present at PAST_OBS: {time_past}, NOT observed at CURRENT_OBS: {time_current}) ==="
    )
    top_gone = sorted(disappeared_objs, key=lambda p: p.past_confidence or 0, reverse=True)
    for p in top_gone[:_MAX_DETAIL]:
        lines.append(
            f"  {p.past_object_class} CONF={p.past_confidence:.2f}"
            f" ({p.past_lat:.3f},{p.past_lon:.3f})"
            f" LAST_SEEN={fmt_dt(p.past_capture_time)}"
        )
    if len(disappeared_objs) > _MAX_DETAIL:
        lines.append(f"  ... +{len(disappeared_objs) - _MAX_DETAIL} more: {_class_counts(disappeared_objs[_MAX_DETAIL:])}")
    if not disappeared_objs:
        lines.append("  (none)")

    lines += [
        "\n=== TASK ===",
        "Write a military intelligence report based ONLY on the NEW and DISAPPEARED objects above.",
        "Use PAST_OBS and CURRENT_OBS timestamps (not today's date) as the observation times.",
        "For DISAPPEARED objects, state they were 'no longer observed in current imagery' — "
        "do NOT imply they are destroyed, eliminated, or permanently gone.",
        "Do NOT mention stationary or repositioned/moved objects.",
        "Sections: 1.CLASSIFICATION 2.EXECUTIVE SUMMARY 3.SITUATION 4.CHANGE ANALYSIS"
        " 5.THREAT ASSESSMENT 6.INTELLIGENCE GAPS 7.RECOMMENDED ACTIONS 8.APPENDIX",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------------------------

class _VllmBackend:
    def __init__(self):
        self._llm = None

    def _load(self):
        from vllm import LLM

        logger.info(f"[Reporter] Loading {LLM_MODEL_NAME} via vLLM (AWQ)...")
        self._llm = LLM(
            model=LLM_MODEL_NAME,
            quantization="awq",
            dtype="float16",
            gpu_memory_utilization=LLM_GPU_MEMORY_UTILIZATION,
            max_model_len=4096,
        )
        logger.info(f"[Reporter] {LLM_MODEL_NAME} loaded.")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from vllm import SamplingParams

        if self._llm is None:
            self._load()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        sampling_params = SamplingParams(
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_NEW_TOKENS,
        )
        outputs = self._llm.chat(messages, sampling_params=sampling_params)
        return outputs[0].outputs[0].text


class _FallbackBackend:
    """
    Rule-based report generator used when no LLM backend is available.
    Produces a structured summary directly from pairing records.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Extract the data section from the user_prompt (everything before INSTRUCTIONS)
        data_section = user_prompt.split("=== INSTRUCTIONS ===")[0].strip()
        return textwrap.dedent(f"""
            1. CLASSIFICATION: UNCLASSIFIED // FOR OFFICIAL USE ONLY (AUTO-GENERATED)

            2. EXECUTIVE SUMMARY
            This report was generated by the rule-based fallback engine because no LLM
            backend (HuggingFace / Ollama) was available. The raw detection pairing data
            is reproduced verbatim below for manual analysis.

            3. SITUATION
            AI-based object detection and temporal pairing have completed successfully.
            Refer to the CHANGE ANALYSIS section for object-level status breakdown.

            4. CHANGE ANALYSIS
            {data_section}

            5. THREAT ASSESSMENT
            Manual review required. LLM-based threat assessment unavailable.

            6. INTELLIGENCE GAPS
            - LLM threat analysis not performed (model not loaded).
            - Install transformers + EXAONE model or configure Ollama for full analysis.

            7. RECOMMENDED ACTIONS
            - Review moved/new/disappeared objects in the pairing database.
            - Configure LLM backend and re-run for full intelligence assessment.

            8. APPENDIX
            See pairing database (data/db/object_pairings.db) for complete object inventory.
        """).strip()


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
    if LLM_BACKEND == "vllm":
        return _VllmBackend()
    logger.warning(f"[Reporter] 알 수 없는 LLM_BACKEND='{LLM_BACKEND}', vLLM으로 대체합니다.")
    return _VllmBackend()


class _SafeBackend:
    """
    Wraps any backend and catches load/generate errors,
    falling back to _FallbackBackend if the primary backend fails.
    """

    def __init__(self, primary):
        self._primary = primary
        self._fallback = _FallbackBackend()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            return self._primary.generate(system_prompt, user_prompt)
        except Exception as exc:
            logger.warning(
                f"[Reporter] LLM backend failed ({exc}). "
                "Using rule-based fallback report."
            )
            return self._fallback.generate(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Main reporter class
# ---------------------------------------------------------------------------

class MilitaryReporter:
    """Generates military change-detection reports using EXAONE4-32b."""

    def __init__(self):
        self._backend = _SafeBackend(_get_backend())

    def generate_report(
        self,
        pairings: List[PairingRecord],
        output_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Generate a military intelligence report from the given pairing records,
        then persist it to the Reports DB (saved_time, file_path, full content).

        Args:
            pairings:    List of PairingRecord objects (latest session).
            output_path: If provided, write the report text to this file path.
            session_id:  Pipeline session UUID (stored in the Reports DB row).

        Returns:
            Report text as a string.
        """
        if not pairings:
            logger.warning("[Reporter] No pairing records provided – returning empty report.")
            return "NO DATA: No pairing records available for report generation."

        report_time = datetime.now(tz=timezone.utc)
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(pairings)

        logger.info(f"[Reporter] Generating military report for {len(pairings)} pairings...")
        report_text = self._backend.generate(system_prompt, user_prompt)

        # Derive observation period from pairing records (metadata.json capture times)
        past_times = [p.past_capture_time for p in pairings if p.past_capture_time]
        current_times = [p.current_capture_time for p in pairings if p.current_capture_time]
        obs_past = min(past_times).strftime("%Y-%m-%dT%H:%M:%SZ") if past_times else "UNKNOWN"
        obs_current = max(current_times).strftime("%Y-%m-%dT%H:%M:%SZ") if current_times else "UNKNOWN"

        # Prepend metadata header
        n_current = sum(1 for p in pairings if p.status in ('new', 'matched', 'moved'))
        n_new_rep = sum(1 for p in pairings if p.status == 'new')
        n_matched_rep = sum(1 for p in pairings if p.status == 'matched')
        n_moved_rep = sum(1 for p in pairings if p.status == 'moved')
        n_disappeared_rep = sum(1 for p in pairings if p.status == 'disappeared')
        header = (
            f"{'='*72}\n"
            f"  MILITARY INTELLIGENCE REPORT\n"
            f"  Generated by: Multi-Source Intelligent System (MSIS)\n"
            f"  Model: {LLM_MODEL_NAME}\n"
            f"  Past observation:    {obs_past}\n"
            f"  Current observation: {obs_current}\n"
            f"  Report generated:    {report_time.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
            f"  Current frame detections: {n_current}"
            f"  (new={n_new_rep} / stationary={n_matched_rep} / moved={n_moved_rep})\n"
            f"  Disappeared (past only):  {n_disappeared_rep}\n"
            f"  Total pairing records:    {len(pairings)}\n"
            f"{'='*72}\n\n"
        )
        full_report = header + report_text

        # Write to file if requested
        saved_file_path: Optional[str] = None
        if output_path:
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_report)
            saved_file_path = str(Path(output_path).resolve())
            logger.info(f"[Reporter] Report written to {saved_file_path}")

        # Persist to Reports DB
        db_record = insert_report(
            report_time=report_time.replace(tzinfo=None),  # store as naive UTC
            report_content=full_report,
            llm_model=LLM_MODEL_NAME,
            llm_backend=LLM_BACKEND,
            pairing_count=len(pairings),
            session_id=session_id,
            file_path=saved_file_path,
        )
        logger.info(
            f"[Reporter] Report saved to DB  "
            f"id={db_record.id}  saved_time={db_record.saved_time}"
        )

        return full_report

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
        "Do not analyse or comment on stationary or repositioned objects."
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


def _build_user_prompt(pairings: List[PairingRecord], report_time: datetime) -> str:
    """Serialise pairing records into a compact prompt for the LLM."""

    def fmt_dt(dt: Optional[datetime]) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else "N/A"

    new_objs = [p for p in pairings if p.status == "new"]
    disappeared_objs = [p for p in pairings if p.status == "disappeared"]

    # matched/moved 객체는 보고서에서 제외 — 집계 참고용으로만 카운트
    n_matched = sum(1 for p in pairings if p.status == "matched")
    n_moved   = sum(1 for p in pairings if p.status == "moved")

    lats = [p.lat_center for p in pairings]
    lons = [p.lon_center for p in pairings]
    lat_c = sum(lats) / len(lats) if lats else 0.0
    lon_c = sum(lons) / len(lons) if lons else 0.0

    all_times = [
        p.current_capture_time for p in pairings if p.current_capture_time
    ] + [
        p.past_capture_time for p in pairings if p.past_capture_time
    ]
    time_past = min(all_times).strftime("%Y-%m-%dT%H:%M:%SZ") if all_times else "UNKNOWN"
    time_current = max(all_times).strftime("%Y-%m-%dT%H:%M:%SZ") if all_times else "UNKNOWN"

    lines = [
        f"TIME: {fmt_dt(report_time)}  ROI: {lat_c:.3f},{lon_c:.3f}"
        f"  WINDOW: {time_past}→{time_current}",
        f"TOTAL: {len(pairings)}  NEW:{len(new_objs)}  DISAPPEARED:{len(disappeared_objs)}"
        f"  (EXCLUDED — STATIONARY:{n_matched}  MOVED:{n_moved})",
        "NOTE: Report covers only NEW and DISAPPEARED objects.",
    ]

    # --- NEW objects (high-value: list top _MAX_DETAIL by confidence) ---
    lines.append("\n=== NEW OBJECTS ===")
    top_new = sorted(new_objs, key=lambda p: p.current_confidence or 0, reverse=True)
    for p in top_new[:_MAX_DETAIL]:
        lines.append(
            f"  {p.current_object_class} CONF={p.current_confidence:.2f}"
            f" ({p.current_lat:.3f},{p.current_lon:.3f})"
        )
    if len(new_objs) > _MAX_DETAIL:
        lines.append(f"  ... +{len(new_objs) - _MAX_DETAIL} more: {_class_counts(new_objs[_MAX_DETAIL:])}")
    if not new_objs:
        lines.append("  (none)")

    # --- DISAPPEARED objects ---
    lines.append("\n=== DISAPPEARED OBJECTS ===")
    top_gone = sorted(disappeared_objs, key=lambda p: p.past_confidence or 0, reverse=True)
    for p in top_gone[:_MAX_DETAIL]:
        lines.append(
            f"  {p.past_object_class} CONF={p.past_confidence:.2f}"
            f" ({p.past_lat:.3f},{p.past_lon:.3f})"
            f" LAST={fmt_dt(p.past_capture_time)}"
        )
    if len(disappeared_objs) > _MAX_DETAIL:
        lines.append(f"  ... +{len(disappeared_objs) - _MAX_DETAIL} more: {_class_counts(disappeared_objs[_MAX_DETAIL:])}")
    if not disappeared_objs:
        lines.append("  (none)")

    lines += [
        "\n=== TASK ===",
        "Write a military intelligence report based ONLY on the NEW and DISAPPEARED objects above.",
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

        logger.info(f"[Reporter] Loading {LLM_MODEL_NAME} via vLLM...")
        self._llm = LLM(model=LLM_MODEL_NAME, trust_remote_code=True)
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
            f"  Records analysed: {len(pairings)} total  "
        f"({sum(1 for p in pairings if p.status in ('new','disappeared'))} new/disappeared used)\n"
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

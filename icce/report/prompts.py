"""
Prompt construction for the grounding ablation (contribution C3).

Five conditions, each seeing strictly more structure than the last:

  template      no LLM at all -- deterministic prose from the change inventory.
                Establishes what the *structure alone* is worth, and bounds
                hallucination at zero by construction.
  llm_raw       LLM + the unaggregated detection dump. This is what "just throw
                the detections at the model" gets you.
  llm_struct    LLM + the aggregated change inventory (the current MSIS prompt,
                minus any history). Isolates aggregation from retrieval.
  llm_flat_rag  llm_struct + top-k retrieved past observation texts. The
                standard RAG baseline: history, but no structure over it.
  llm_graphrag  llm_struct + GraphRAG local (entity history) and global
                (community) context. Ours.

Two output styles share these conditions:
  caption  one or two sentences, LEVIR-CC register, for BLEU/METEOR/ROUGE/CIDEr
  report   the full structured urban-monitoring report, for CFS and the rubric

Style exemplars for caption mode are drawn from the LEVIR-CC **train** split
only. Without them a general-purpose LLM writes fluent prose in the wrong
register and loses n-gram overlap for reasons that have nothing to do with
grounding; with them, all conditions are anchored identically.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from icce.report.evidence import ChangeEvidence, ObservedChange

GROUNDING_MODES = ("template", "llm_raw", "llm_struct", "llm_flat_rag", "llm_graphrag")
OUTPUT_STYLES = ("caption", "report")

MAX_DETAIL = 20


# ---------------------------------------------------------------------------
# evidence rendering
# ---------------------------------------------------------------------------
def render_raw_dump(ev: ChangeEvidence) -> str:
    """Ungrouped detection list -- the `llm_raw` condition."""
    w, h = ev.image_size
    lines = [f"DETECTIONS ({len(ev.changes)} rows), tile {w}x{h}px, "
             f"{ev.interval_days()} day interval:"]
    for c in ev.changes[:80]:
        lines.append(
            f"  status={c.status} class={c.object_class} conf={c.confidence:.2f} "
            f"bbox=({c.bbox_px[0]:.0f},{c.bbox_px[1]:.0f},{c.bbox_px[2]:.0f},{c.bbox_px[3]:.0f}) "
            f"pos=({c.lat:.5f},{c.lon:.5f})"
        )
    if len(ev.changes) > 80:
        lines.append(f"  ... +{len(ev.changes) - 80} more rows")
    return "\n".join(lines)


def render_inventory(ev: ChangeEvidence) -> str:
    """Aggregated change inventory shared by every LLM condition but `llm_raw`."""
    w, h = ev.image_size
    lines = [
        "=== CHANGE INVENTORY ===",
        f"AREA: {ev.scene}  CENTRE: {ev.lat:.5f},{ev.lon:.5f}",
        f"PAST_OBS: {ev.past_time:%Y-%m-%d}  CURRENT_OBS: {ev.current_time:%Y-%m-%d}"
        f"  INTERVAL: {ev.interval_days()} days",
        f"CHANGE_PRESENT: {'yes' if ev.has_change else 'no'}",
        f"CHANGE_COUNTS: appeared={len(ev.appeared)} disappeared={len(ev.disappeared)}"
        f" modified={len(ev.modified)} unchanged={len(ev.stable)}",
    ]

    for title, objs in (("APPEARED", ev.appeared),
                        ("DISAPPEARED", ev.disappeared),
                        ("MODIFIED", ev.modified)):
        lines.append(f"{title} ({len(objs)}): {ev.class_counts(objs)}")
        for o in sorted(objs, key=lambda x: -x.confidence)[:MAX_DETAIL]:
            lines.append(f"  - {o.object_class} in the {o.bearing(w, h)} part "
                         f"(conf {o.confidence:.2f})")
        if len(objs) > MAX_DETAIL:
            lines.append(f"  - ... +{len(objs) - MAX_DETAIL} more")

    lines.append(f"UNCHANGED: {len(ev.stable)} object(s) matched across both dates")
    lines.append("=== END CHANGE INVENTORY ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# template condition (no LLM)
# ---------------------------------------------------------------------------
_NUM_WORD = {2: "two", 3: "three", 4: "four", 5: "five"}


def _plural(n: int, word: str) -> str:
    """Spell out small counts: LEVIR-CC references never use digits."""
    if n == 1:
        return f"one {word}"
    if n > 5:
        return f"many {word}s"
    return f"{_NUM_WORD.get(n, str(n))} {word}s"


def template_caption(ev: ChangeEvidence) -> str:
    if not ev.has_change:
        return "The scene is the same as before and nothing has changed."

    w, h = ev.image_size
    clauses: List[str] = []
    for objs, verb in ((ev.appeared, "appeared"),
                       (ev.disappeared, "disappeared"),
                       (ev.modified, "were changed")):
        if not objs:
            continue
        head = objs[0]
        by_class: Dict[str, int] = {}
        for o in objs:
            by_class[o.object_class] = by_class.get(o.object_class, 0) + 1
        parts = [_plural(n, cls) for cls, n in sorted(by_class.items(), key=lambda kv: -kv[1])[:2]]
        clauses.append(f"{' and '.join(parts)} {verb} in the {head.bearing(w, h)} part of the scene")
    return (" and ".join(clauses)).capitalize() + "."


def template_report(ev: ChangeEvidence) -> str:
    w, h = ev.image_size
    lines = [
        f"URBAN CHANGE MONITORING REPORT -- {ev.scene}",
        f"Observation window: {ev.past_time:%Y-%m-%d} to {ev.current_time:%Y-%m-%d} "
        f"({ev.interval_days()} days)",
        "",
        "1. SUMMARY",
        f"   {template_caption(ev)}",
        "",
        "2. CHANGE INVENTORY",
        f"   Appeared: {len(ev.appeared)} ({ev.class_counts(ev.appeared)})",
        f"   Disappeared: {len(ev.disappeared)} ({ev.class_counts(ev.disappeared)})",
        f"   Modified: {len(ev.modified)} ({ev.class_counts(ev.modified)})",
        f"   Unchanged: {len(ev.stable)}",
        "",
        "3. DEVELOPMENT ASSESSMENT",
    ]
    if ev.appeared and not ev.disappeared:
        lines.append("   Net construction: the built-up footprint expanded over the window.")
    elif ev.disappeared and not ev.appeared:
        lines.append("   Net demolition or clearance: built-up footprint contracted.")
    elif ev.appeared and ev.disappeared:
        lines.append("   Redevelopment: structures were removed and rebuilt in the same area.")
    else:
        lines.append("   No development activity detected in this window.")
    lines += ["", "4. RECOMMENDED ACTIONS",
              "   Continue routine monitoring at the next revisit."
              if not ev.has_change else
              "   Verify the newly detected structures against the permit record."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# system prompts
# ---------------------------------------------------------------------------
_CAPTION_SYSTEM = (
    "You are an urban change-monitoring analyst describing what changed between "
    "two satellite images of the same place, taken at different times.\n"
    "Write ONE or TWO short sentences in plain English, present or past tense, "
    "describing only the changes.\n"
    "Rules:\n"
    "  - State only what the CHANGE INVENTORY supports. Never invent objects, "
    "counts, or locations that are not listed.\n"
    "  - If the inventory says no change is present, say that the scene is the "
    "same as before.\n"
    "  - No preamble, no bullet points, no headings. Output the sentence only."
)

_REPORT_SYSTEM = (
    "You are an urban change-monitoring analyst writing a short interpretation "
    "report for a consumer property-monitoring service.\n"
    "Use exactly these sections:\n"
    "  1. SUMMARY\n  2. CHANGE INVENTORY\n  3. DEVELOPMENT ASSESSMENT\n"
    "  4. INFORMATION GAPS\n  5. RECOMMENDED ACTIONS\n"
    "Rules:\n"
    "  - Every factual claim must trace to the CHANGE INVENTORY or the "
    "historical context block. Do not invent objects, counts or locations.\n"
    "  - If the evidence is thin, say so under INFORMATION GAPS rather than "
    "filling the gap with speculation.\n"
    "  - Be concise: at most roughly 200 words."
)


def system_prompt(style: str) -> str:
    if style not in OUTPUT_STYLES:
        raise ValueError(f"unknown style '{style}', expected one of {OUTPUT_STYLES}")
    return _CAPTION_SYSTEM if style == "caption" else _REPORT_SYSTEM


# ---------------------------------------------------------------------------
# user prompts
# ---------------------------------------------------------------------------
def user_prompt(
    ev: ChangeEvidence,
    mode: str,
    style: str,
    graph_context: str = "",
    rag_context: str = "",
    style_examples: Optional[Sequence[str]] = None,
) -> str:
    if mode not in GROUNDING_MODES:
        raise ValueError(f"unknown grounding mode '{mode}', expected one of {GROUNDING_MODES}")

    blocks: List[str] = []

    if style == "caption" and style_examples:
        blocks.append(
            "Reference style -- write in the same register as these examples "
            "(they describe OTHER scenes; do not copy their content):\n"
            + "\n".join(f"  - {e}" for e in style_examples)
        )

    blocks.append(render_raw_dump(ev) if mode == "llm_raw" else render_inventory(ev))

    if mode == "llm_flat_rag" and rag_context:
        blocks.append(rag_context)
    if mode == "llm_graphrag" and graph_context:
        blocks.append(graph_context)

    blocks.append(
        "Write the one-or-two sentence change description now."
        if style == "caption" else
        "Write the report now, using the five section headings."
    )
    return "\n\n".join(blocks)


def max_tokens_for(style: str) -> int:
    return 96 if style == "caption" else 512

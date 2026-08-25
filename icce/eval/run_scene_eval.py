"""
Scene-level change reporting -- E4 asked at the unit GraphRAG is built for.

Why this exists
---------------
The crop-level ladder (`run_report_eval`) cannot separate `llm_struct`,
`llm_flat_rag` and `llm_graphrag`, and the reason is structural rather than a
bug: all three receive the *same* change inventory for the crop being
described, and CFS scores only claims about that crop. Retrieved history
concerns other crops, so it can move phrasing but never a claim. Measured on
128 crops: 77/128 generations differed, 128/128 claim sets were identical.

At scene level the comparison becomes real. A LEVIR-CD tile is cut into 16
disjoint 256px crops; one report must describe the whole neighbourhood, so the
question stops being "does more context help" and becomes "which representation
of the same 16 observations survives into the report":

    llm_struct     all 16 inventories, concatenated and unaggregated
    llm_flat_rag   top-k retrieved crop summaries -- k < 16, so lossy by design
    llm_graphrag   graph entity/community aggregate over all 16

This is deliberately *not* the additive ladder of the crop-level experiment,
and a paper using it has to say so. Handing `llm_flat_rag` the full inventory
*and* its retrieval would reproduce the crop-level null result, because
retrieval adds nothing to a prompt that already contains everything.

The falsifiable prediction is about counting: top-k truncation drops crops, so
a scene's instance total cannot be recovered from k < 16 summaries, while a
graph aggregate carries it. If `llm_flat_rag` does not lose counts relative to
`llm_graphrag`, the GraphRAG contribution is not supported and the paper should
say that instead.

Ground truth per scene is the union of its crops' majority-vote reference
claims, and the sum of its crops' GT mask instances.

    python -m icce.eval.run_scene_eval \\
        --cache data/cache/pilot/levir_cc_scenes8 \\
        --checkpoint data/checkpoints/pilot_head.pt \\
        --llm server:LGAI-EXAONE/EXAONE-4.0-32B \\
        --out results/pilot_cc_scenes8/scene_level
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from icce.metrics.change_fact import (
    ChangeClaim,
    ChangeFactEvaluator,
    extract_claims,
    gt_claims_from_captions,
)
from icce.pairing_head.cache import load_cache
from icce.report.evidence import ChangeEvidence
from icce.report.flat_rag import FlatRagStore
from icce.report.llm import GenRequest, build_llm, generate_with_cache

logger = logging.getLogger(__name__)

SCENE_MODES = ("template", "llm_struct", "llm_flat_rag", "llm_graphrag")

CHANGE_DIRECTIONS = ("appeared", "disappeared", "modified")

SYSTEM = (
    "You are a satellite monitoring service writing a short neighbourhood "
    "change report for a subscriber. State only what the evidence supports. "
    "Give explicit counts when the evidence gives you counts."
)

TASK = (
    "Write a 3-5 sentence report on what changed across this neighbourhood "
    "between the two visits. State the total number of changed buildings "
    "explicitly as a number."
)


# ---------------------------------------------------------------------------
# scene assembly
# ---------------------------------------------------------------------------
def group_by_scene(
    evidences: Sequence[ChangeEvidence],
) -> "OrderedDict[str, List[ChangeEvidence]]":
    by_scene: "OrderedDict[str, List[ChangeEvidence]]" = OrderedDict()
    for ev in evidences:
        by_scene.setdefault(ev.scene, []).append(ev)
    for scene, evs in by_scene.items():
        by_scene[scene] = sorted(evs, key=lambda e: e.current_time)
    return by_scene


def scene_ground_truth(evs: Sequence[ChangeEvidence]) -> Tuple[Set[ChangeClaim], int, bool]:
    """Union of the crops' reference claims, and the summed GT instance count."""
    claims: Set[ChangeClaim] = set()
    count = 0
    changed = False
    for ev in evs:
        captions = ev.meta.get("captions") or []
        if captions:
            claims |= gt_claims_from_captions(captions)
        count += int(ev.meta.get("n_gt_instances") or 0)
        if ev.meta.get("gt_change_present"):
            changed = True
    # A neighbourhood where some crop changed is a changed neighbourhood, so a
    # "nothing changed" claim inherited from a quiet crop would contradict it.
    if changed:
        claims = {c for c in claims if c.direction != "none"}
    return claims, count, changed


def scene_inventory(evs: Sequence[ChangeEvidence]) -> str:
    """Every crop's inventory, unaggregated -- the `llm_struct` condition."""
    lines = [f"Neighbourhood {evs[0].scene}: {len(evs)} observed sectors."]
    for ev in evs:
        lines.append(f"  - {ev.summary_line()}")
    return "\n".join(lines)


def scene_totals(evs: Sequence[ChangeEvidence]) -> Dict[str, Counter]:
    return {
        "appeared": Counter(c.object_class for ev in evs for c in ev.appeared),
        "disappeared": Counter(c.object_class for ev in evs for c in ev.disappeared),
        "modified": Counter(c.object_class for ev in evs for c in ev.modified),
    }


def template_scene_report(evs: Sequence[ChangeEvidence]) -> str:
    """Deterministic aggregation: the counting floor, with no LLM involved."""
    totals = scene_totals(evs)
    n_total = sum(sum(c.values()) for c in totals.values())
    if not n_total:
        return f"The neighbourhood is unchanged across all {len(evs)} sectors."

    parts = []
    for direction, verb in (("appeared", "appeared"),
                            ("disappeared", "were removed"),
                            ("modified", "were modified")):
        counter = totals[direction]
        if not counter:
            continue
        items = ", ".join(f"{n} {cls}{'s' if n != 1 else ''}"
                          for cls, n in counter.most_common())
        parts.append(f"{items} {verb}")
    body = "; ".join(parts)
    return (f"Across {len(evs)} sectors of the neighbourhood, {body}. "
            f"{n_total} buildings changed in total.")


# ---------------------------------------------------------------------------
# contexts
# ---------------------------------------------------------------------------
def flat_rag_context(evs: Sequence[ChangeEvidence], k: int, radius_deg: float) -> str:
    """Top-k retrieval over the scene's own crops, queried at its centroid.

    k < len(evs) is the point: this condition is what a flat retriever can see,
    and the crops it does not retrieve are simply absent from the report.
    """
    store = FlatRagStore(radius_deg=radius_deg)
    for ev in evs:
        store.add(ev.pair_id, ev.summary_line(), ev.lat, ev.lon)
    lat = sum(e.lat for e in evs) / len(evs)
    lon = sum(e.lon for e in evs) / len(evs)
    query = f"{evs[0].scene} neighbourhood change summary"
    return store.context_block(query, lat, lon, k)


def graph_context(evs: Sequence[ChangeEvidence], db_path: Path, radius_deg: float) -> str:
    """Index every crop, then ask the graph for the neighbourhood's aggregate.

    Unlike the crop-level driver there is no retrieve-then-index ordering to
    honour: the question is about the scene as a whole, and every crop of it is
    legitimately in scope. Nothing outside this scene is ever indexed.
    """
    from icce.report.graph_context import GraphContextBuilder, evidence_to_pairing_records

    builder = GraphContextBuilder(db_path=db_path, radius_deg=radius_deg)
    records: List = []
    for ev in evs:
        recs = evidence_to_pairing_records(ev)
        records.extend(recs)
        try:
            builder._indexer.index_pairings(recs, session_id=ev.pair_id)
            builder.n_indexed += 1
        except Exception as exc:
            logger.warning("graph indexing failed for %s: %s", ev.pair_id, exc)
    try:
        ctx = builder._indexer.get_historical_context(records, radius_deg)
    except Exception as exc:
        logger.warning("graph context retrieval failed for %s: %s", evs[0].scene, exc)
        ctx = ""
    logger.info("scene %s: graph=%s", evs[0].scene, builder.stats())
    return ctx


def scene_prompt(mode: str, evs: Sequence[ChangeEvidence], context: str) -> str:
    header = f"Neighbourhood {evs[0].scene}, {len(evs)} sectors observed."
    if mode == "llm_struct":
        body = scene_inventory(evs)
    elif mode == "llm_flat_rag":
        body = ("Retrieved observations for this neighbourhood "
                f"(top matches only, not the full set):\n{context}")
    elif mode == "llm_graphrag":
        body = f"Knowledge-graph summary of this neighbourhood:\n{context}"
    else:
        raise ValueError(f"no prompt for mode '{mode}'")
    return "\n\n".join([header, body, TASK])


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def claimed_total(text: str) -> Optional[int]:
    """Total changed instances the report commits to.

    CFS's own count rule takes `max` over stated counts, which is right for a
    single-crop caption naming one class. A neighbourhood report says "12 new
    and 3 demolished", where the quantity under test is the sum, so this sums
    counts over change-direction claims instead. Reported alongside CFS's
    CountMAE, never silently substituted for it.
    """
    counts = [c.count for c in extract_claims(text)
              if c.count is not None and c.direction in CHANGE_DIRECTIONS]
    return sum(counts) if counts else None


def score_scene_condition(
    mode: str,
    texts: Dict[str, str],
    scenes: "OrderedDict[str, List[ChangeEvidence]]",
    gt: Dict[str, Tuple[Set[ChangeClaim], int, bool]],
) -> Dict:
    fact = ChangeFactEvaluator()
    abs_errs: List[float] = []
    n_no_count = 0

    for scene, evs in scenes.items():
        text = texts.get(scene, "")
        gt_claims, gt_count, gt_changed = gt[scene]
        fact.update(report=text, gt_claims=gt_claims,
                    gt_change_present=gt_changed, gt_instance_count=gt_count)
        total = claimed_total(text)
        if total is None:
            n_no_count += 1
        else:
            abs_errs.append(abs(total - gt_count))

    row: Dict = {"name": mode, "mode": mode, "n_scenes": len(scenes)}
    row.update(fact.compute().as_dict())
    row["scene_count_mae"] = (sum(abs_errs) / len(abs_errs)) if abs_errs else None
    row["n_scenes_without_a_count"] = n_no_count
    row["mean_length_words"] = (sum(len(t.split()) for t in texts.values())
                                / max(1, len(texts)))
    return row


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Scene-level change reporting")
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--llm", default="echo")
    ap.add_argument("--modes", nargs="*", default=list(SCENE_MODES))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rag-k", type=int, default=5)
    ap.add_argument("--graph-radius", type=float, default=0.05)
    ap.add_argument("--match-radius", type=float, default=0.001)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from datetime import datetime, timedelta, timezone
    from icce.eval.run_report_eval import build_evidences
    from icce.pairing_head.infer import LearnedPairer

    samples, emb = load_cache(args.cache)
    base = datetime(2026, 3, 1, tzinfo=timezone.utc)
    order = {s.pair_id: i for i, s in enumerate(samples)}
    past_of = lambda s: base + timedelta(hours=order[s.pair_id])
    cur_of = lambda s: past_of(s) + timedelta(days=30)

    pairer = LearnedPairer.from_checkpoint(
        args.checkpoint, device=args.device, match_radius_deg=args.match_radius
    )
    evidences = build_evidences(samples, emb, pairer, past_of, cur_of)
    scenes = group_by_scene(evidences)
    logger.info("%d crops -> %d scenes (%.1f crops/scene)", len(evidences), len(scenes),
                len(evidences) / max(1, len(scenes)))
    if args.rag_k >= min(len(e) for e in scenes.values()):
        logger.warning("rag-k=%d is not smaller than the smallest scene: flat RAG "
                       "sees everything and the comparison is vacuous",
                       args.rag_k)

    gt = {scene: scene_ground_truth(evs) for scene, evs in scenes.items()}
    logger.info("GT instances per scene: %s",
                {s: g[1] for s, g in list(gt.items())[:8]})

    contexts: Dict[str, Dict[str, str]] = {"llm_flat_rag": {}, "llm_graphrag": {}}
    for scene, evs in scenes.items():
        if "llm_flat_rag" in args.modes:
            contexts["llm_flat_rag"][scene] = flat_rag_context(
                evs, args.rag_k, args.graph_radius)
        if "llm_graphrag" in args.modes:
            contexts["llm_graphrag"][scene] = graph_context(
                evs, out / "graph" / f"{scene}.db", args.graph_radius)

    llm = None
    rows: List[Dict] = []
    for mode in args.modes:
        t0 = time.time()
        if mode == "template":
            gen = {scene: template_scene_report(evs) for scene, evs in scenes.items()}
        else:
            if llm is None:
                llm = build_llm(args.llm, max_model_len=args.max_model_len,
                                temperature=0.0)
            reqs = [
                GenRequest(key=scene, system=SYSTEM,
                           user=scene_prompt(mode, evs, contexts.get(mode, {}).get(scene, "")),
                           max_tokens=320)
                for scene, evs in scenes.items()
            ]
            ctx_words = [len(r.user.split()) for r in reqs]
            logger.info("%-14s prompt words: mean %.0f", mode,
                        sum(ctx_words) / max(1, len(ctx_words)))
            gen = generate_with_cache(llm, reqs, out / f"gen_{mode}_scene.jsonl")

        row = score_scene_condition(mode, gen, scenes, gt)
        row["seconds"] = time.time() - t0
        rows.append(row)
        logger.info("%-14s CFS-F1=%.4f SceneCountMAE=%s (no count: %d/%d)",
                    mode, row.get("cfs_f1", 0.0),
                    ("%.2f" % row["scene_count_mae"]) if row["scene_count_mae"] is not None else "n/a",
                    row["n_scenes_without_a_count"], row["n_scenes"])

    payload = {
        "cache": str(args.cache),
        "llm": args.llm,
        "rag_k": args.rag_k,
        "n_scenes": len(scenes),
        "crops_per_scene": {s: len(e) for s, e in scenes.items()},
        "gt_instances_per_scene": {s: g[1] for s, g in gt.items()},
        "results": rows,
    }
    (out / "scene_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", out / "scene_results.json")

    print("\nScene-level neighbourhood reports")
    print("-" * 78)
    print(f"{'method':<16}{'CFS-P':>9}{'CFS-R':>9}{'CFS-F1':>9}"
          f"{'CountMAE':>11}{'no-count':>10}{'words':>8}")
    for r in rows:
        mae = "n/a" if r["scene_count_mae"] is None else f"{r['scene_count_mae']:.2f}"
        print(f"{r['name']:<16}{r['cfs_precision']*100:>9.2f}{r['cfs_recall']*100:>9.2f}"
              f"{r['cfs_f1']*100:>9.2f}{mae:>11}"
              f"{r['n_scenes_without_a_count']:>10}{r['mean_length_words']:>8.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

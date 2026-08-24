"""
Experiment E3/E4: change-report generation quality and factuality.

    python -m icce.eval.run_report_eval \
        --cache data/cache/levir_cc_test \
        --checkpoint data/checkpoints/pairing_head.pt \
        --llm LGAI-EXAONE/EXAONE-4.0-32B-Instruct \
        --style caption --out results/levir_cc_caption

For each grounding condition the *same* pairing output is rendered into the
same evidence object and then into a condition-specific prompt, so differences
between rows are attributable to grounding alone.

Two scoring axes:
  * caption style -> BLEU-n / METEOR / ROUGE-L / CIDEr-D against the five
    LEVIR-CC human references, directly comparable with the RSICC leaderboard
  * both styles   -> Change-Fact-Score: claim precision/recall, hallucination
    rate, change-decision accuracy, count error

Retrieval for the flat-RAG and GraphRAG conditions is built strictly from
observations that precede the sample being described. A pair never contributes
to its own context.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from icce.metrics.caption_metrics import score_corpus
from icce.metrics.change_fact import (
    ChangeFactEvaluator,
    extract_claims,
    gt_claims_from_captions,
)
from icce.pairing_head.cache import load_cache, sample_dets
from icce.report.evidence import ChangeEvidence, from_pairing_result
from icce.report.flat_rag import FlatRagStore
from icce.report.graph_context import build_contexts
from icce.report.llm import GenRequest, build_llm, generate_with_cache
from icce.report.prompts import (
    GROUNDING_MODES,
    IMAGE_CONDITIONED_MODES,
    max_tokens_for,
    system_prompt,
    template_caption,
    template_report,
    user_prompt,
)

logger = logging.getLogger(__name__)

DEFAULT_STYLE_EXAMPLES = 4


def build_evidences(samples, emb, pairer, past_time_of, current_time_of) -> List[ChangeEvidence]:
    out = []
    for s in samples:
        past, cur = sample_dets(s, emb)
        res = pairer.pair(past, cur, image_size=s.image_size)
        lat = sum(d.lat for d in (cur or past)) / max(1, len(cur or past)) if (cur or past) else 0.0
        lon = sum(d.lon for d in (cur or past)) / max(1, len(cur or past)) if (cur or past) else 0.0
        out.append(from_pairing_result(
            res, pair_id=s.pair_id, scene=s.parent_scene or s.pair_id,
            lat=lat, lon=lon,
            past_time=past_time_of(s), current_time=current_time_of(s),
            image_size=s.image_size,
            meta={"captions": s.captions, "gt_change_present": s.gt_change_present,
                  "n_gt_instances": len(s.gt_instances)},
        ))
    return out


def build_rag_contexts(evidences: Sequence[ChangeEvidence], k: int = 5,
                       radius_deg: float = 0.05) -> Dict[str, str]:
    """Flat retrieval over strictly-earlier observations, one store per scene."""
    by_scene: "OrderedDict[str, List[ChangeEvidence]]" = OrderedDict()
    for ev in evidences:
        by_scene.setdefault(ev.scene, []).append(ev)

    out: Dict[str, str] = {}
    for evs in by_scene.values():
        store = FlatRagStore(radius_deg=radius_deg)
        for ev in sorted(evs, key=lambda e: e.current_time):
            out[ev.pair_id] = (store.context_block(ev.summary_line(), ev.lat, ev.lon, k)
                               if len(store) else "")
            store.add(ev.pair_id, ev.summary_line(), ev.lat, ev.lon)
    return out


def style_examples_from_train(dataset: str, n: int) -> List[str]:
    """Register anchors taken from the TRAIN split only -- never from test."""
    if n <= 0:
        return []
    try:
        from icce.datasets.registry import load as load_dataset
        pairs = load_dataset(dataset, split="train", limit=max(20, n * 4))
    except Exception as exc:
        logger.warning("no train-split style examples (%s); prompts run unanchored", exc)
        return []

    changed = [p.captions[0] for p in pairs if p.captions and p.change_flag is not False]
    same = [p.captions[0] for p in pairs if p.captions and p.change_flag is False]
    out = changed[: max(1, n - 1)] + same[:1]
    logger.info("style examples from train split: %d", len(out))
    return out


def score_condition(
    mode: str,
    style: str,
    texts: Dict[str, str],
    evidences: Sequence[ChangeEvidence],
) -> Dict:
    hyps, refs, ids = [], [], []
    fact = ChangeFactEvaluator()

    for ev in evidences:
        text = texts.get(ev.pair_id, "")
        captions = ev.meta.get("captions") or []
        if captions:
            hyps.append(text)
            refs.append(captions)
            ids.append(ev.pair_id)
        fact.update(
            report=text,
            gt_claims=gt_claims_from_captions(captions) if captions else set(),
            gt_change_present=ev.meta.get("gt_change_present"),
            gt_instance_count=ev.meta.get("n_gt_instances"),
        )

    row: Dict = {"name": mode, "mode": mode, "style": style,
                 "n_generated": len([e for e in evidences if texts.get(e.pair_id)])}
    if style == "caption" and hyps:
        row.update(score_corpus(hyps, refs))
    row.update(fact.compute().as_dict())
    row["mean_length_words"] = (sum(len(t.split()) for t in texts.values())
                                / max(1, len(texts)))
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Report generation + factuality evaluation")
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--dataset", default="levir_cc")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="pairing head; omitted -> production heuristic")
    ap.add_argument("--llm", default="echo",
                    help="'echo', 'ollama:<model>' or a HF model id for vLLM")
    ap.add_argument("--vlm", default=None,
                    help="multimodal model for the vlm_direct baseline, "
                         "e.g. Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--style", default="caption", choices=("caption", "report"))
    ap.add_argument("--modes", nargs="*", default=list(GROUNDING_MODES))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--match-radius", type=float, default=0.001)
    ap.add_argument("--graph-radius", type=float, default=0.05)
    ap.add_argument("--rag-k", type=int, default=5)
    ap.add_argument("--style-examples", type=int, default=DEFAULT_STYLE_EXAMPLES)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-leakage", action="store_true",
                    help="debug only: continue past an integrity violation and "
                         "stamp the results as unclean")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    samples, emb = load_cache(args.cache)
    if args.limit:
        samples = samples[: args.limit]
    logger.info("%d cached pairs", len(samples))

    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 3, 1, tzinfo=timezone.utc)
    order = {s.pair_id: i for i, s in enumerate(samples)}
    past_of = lambda s: base + timedelta(hours=order[s.pair_id])
    cur_of = lambda s: past_of(s) + timedelta(days=30)

    ckpt_extra: Optional[Dict] = None
    if args.checkpoint and Path(args.checkpoint).is_file():
        from icce.pairing_head.model import PairingHead
        _, ckpt_extra = PairingHead.load(args.checkpoint)

    from icce.eval.integrity import check_evaluation
    integrity = check_evaluation(
        eval_ids=[s.pair_id for s in samples],
        eval_split=(samples[0].split if samples else "test"),
        train_ids=(ckpt_extra or {}).get("train_pair_ids"),
        checkpoint_extra=ckpt_extra,
        style_example_split=("train" if args.style == "caption"
                             and args.style_examples > 0 else None),
    ).raise_if_dirty(allow=args.allow_leakage)

    from icce.pairing_head.infer import LearnedPairer
    pairer = LearnedPairer.from_checkpoint(
        args.checkpoint, device=args.device, match_radius_deg=args.match_radius
    )
    evidences = build_evidences(samples, emb, pairer, past_of, cur_of)
    logger.info("built evidence for %d pairs (%d with change)",
                len(evidences), sum(1 for e in evidences if e.has_change))

    graph_ctx = build_contexts(evidences, db_path=out / "graph" / "g.db",
                               radius_deg=args.graph_radius) \
        if "llm_graphrag" in args.modes else {}
    rag_ctx = build_rag_contexts(evidences, k=args.rag_k, radius_deg=args.graph_radius) \
        if "llm_flat_rag" in args.modes else {}

    examples = (style_examples_from_train(args.dataset, args.style_examples)
                if args.style == "caption" else [])

    llm = None
    rows: List[Dict] = []
    for mode in args.modes:
        t0 = time.time()
        if mode == "template":
            gen = {ev.pair_id: (template_caption(ev) if args.style == "caption"
                                else template_report(ev)) for ev in evidences}
        elif mode in IMAGE_CONDITIONED_MODES:
            if not args.vlm:
                logger.warning("skipping %s: --vlm not set", mode)
                continue
            from icce.report.vlm import build_captioner
            by_id = {s_.pair_id: s_ for s_ in samples}
            # 'server:<model>' routes to a standalone vLLM server; anything
            # else still loads the weights in this process.
            captioner = build_captioner(args.vlm, max_model_len=args.max_model_len,
                                        gpu_memory_utilization=args.gpu_mem)
            gen = captioner.caption_batch(
                [(ev.pair_id, by_id[ev.pair_id].image_a, by_id[ev.pair_id].image_b)
                 for ev in evidences if ev.pair_id in by_id])
            _dump_generations(gen, out / f"gen_{mode}_{args.style}.jsonl")
            del captioner
        else:
            if llm is None:
                llm = build_llm(args.llm, max_model_len=args.max_model_len,
                                gpu_memory_utilization=args.gpu_mem, temperature=0.0)
            reqs = [
                GenRequest(
                    key=ev.pair_id,
                    system=system_prompt(args.style),
                    user=user_prompt(ev, mode, args.style,
                                     graph_context=graph_ctx.get(ev.pair_id, ""),
                                     rag_context=rag_ctx.get(ev.pair_id, ""),
                                     style_examples=examples),
                    max_tokens=max_tokens_for(args.style),
                )
                for ev in evidences
            ]
            gen = generate_with_cache(llm, reqs, out / f"gen_{mode}_{args.style}.jsonl",
                                      batch_size=args.batch_size)

        row = score_condition(mode, args.style, gen, evidences)
        row["seconds"] = time.time() - t0
        rows.append(row)
        logger.info("%-14s CFS-F1=%.4f Hal=%.4f ChgAcc=%s BLEU-4=%s (%.1fs)",
                    mode, row.get("cfs_f1", 0.0), row.get("hallucination_rate", 0.0),
                    _r(row.get("change_accuracy")), _r(row.get("BLEU-4")), row["seconds"])

    from icce.eval.tables import latex_table, print_console_table, save_json, save_latex

    save_json({
        "dataset": args.dataset, "style": args.style, "llm": args.llm,
        "n_pairs": len(evidences), "checkpoint": str(args.checkpoint),
        "graph_radius_deg": args.graph_radius, "rag_k": args.rag_k,
        "style_examples": examples, "integrity": integrity.as_dict(),
        "vlm": args.vlm, "results": rows,
    }, out / f"report_results_{args.style}.json")

    if args.style == "caption":
        cols = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D"]
        print_console_table("LEVIR-CC caption metrics", cols, rows)
        save_latex(latex_table(
            "Change captioning on LEVIR-CC. Baselines are trained on the LEVIR-CC "
            "training split; our conditions are zero-shot apart from the "
            "pairing head.",
            "tab:levir_cc_caption", cols, rows, baseline_group="levir_cc",
        ), out / "table_levir_cc_caption.tex")

    fact_cols = ["cfs_precision", "cfs_recall", "cfs_f1",
                 "hallucination_rate", "change_accuracy"]
    print_console_table(f"Report factuality ({args.style})", fact_cols, rows)
    save_latex(latex_table(
        f"Report factuality by grounding condition ({args.style} style). "
        "CFS is claim-level precision/recall against the human references; "
        "Hal is the share of generated claims with no ground-truth support.",
        f"tab:factuality_{args.style}",
        ["CFS-P", "CFS-R", "CFS-F1", "Hal", "ChgAcc"], rows,
        metric_map={"CFS-P": "cfs_precision", "CFS-R": "cfs_recall",
                    "CFS-F1": "cfs_f1", "Hal": "hallucination_rate",
                    "ChgAcc": "change_accuracy"},
    ), out / f"table_factuality_{args.style}.tex")
    return 0


def _dump_generations(gen: Dict[str, str], path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for k, v in gen.items():
            fh.write(json.dumps({"key": k, "text": v}, ensure_ascii=False) + "\n")


def _r(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else "n/a"


if __name__ == "__main__":
    sys.exit(main())

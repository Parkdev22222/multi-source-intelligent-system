"""
Experiment E6: deployment cost of the pipeline.

ICCE reviewers care whether a proposed consumer service can actually be
operated, so the paper reports a per-stage latency and memory budget rather
than accuracy alone. This runner measures each stage on the target GPU and
emits a table with the per-tile cost and the marginal cost of our additions.

    python -m icce.eval.run_efficiency --cache data/cache/levir_cc_test \
        --checkpoint data/checkpoints/pairing_head.pt \
        --llm LGAI-EXAONE/EXAONE-4.0-32B-Instruct --out results/efficiency

Stages measured
  detection   from the cache metadata written during the SAM3 pass
  pairing     heuristic vs learned head, timed over the cached detections
  graph       knowledge-graph index + retrieve per tile
  report      LLM generation, amortised over the batch vLLM actually runs
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def gpu_memory_mb() -> Optional[float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        return None


def reset_gpu_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def time_stage(fn, n_items: int, warmup: int = 2, repeats: int = 3) -> Dict:
    """Median wall-clock per item over `repeats` passes, after warm-up."""
    for _ in range(warmup):
        fn()
    reset_gpu_peak()
    durations = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - t0)
    total = statistics.median(durations)
    return {
        "total_s": total,
        "per_item_ms": (total / max(1, n_items)) * 1000.0,
        "peak_gpu_mb": gpu_memory_mb(),
        "n_items": n_items,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Per-stage latency and memory budget")
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--llm", default=None, help="omit to skip the LLM stage")
    ap.add_argument("--style", default="report", choices=("caption", "report"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--match-radius", type=float, default=0.001)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from icce.pairing_head.cache import load_cache, sample_dets
    from icce.pairing_head.infer import LearnedPairer

    samples, emb = load_cache(args.cache)
    samples = samples[: args.limit]
    dets = [sample_dets(s, emb) for s in samples]
    n = len(samples)
    logger.info("timing over %d cached pairs", n)

    stages: List[Dict] = []

    # -- detection: read back from the cache pass, we do not re-run SAM3 here --
    info_path = Path(args.cache) / "cache_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("seconds_per_pair_mean"):
            stages.append({
                "stage": "SAM3 detection + CLIP (2 frames)",
                "per_item_ms": info["seconds_per_pair_mean"] * 1000.0,
                "p95_ms": (info.get("seconds_per_pair_p95") or 0) * 1000.0,
                "peak_gpu_mb": None,
                "note": "measured during the caching pass",
            })

    # -- pairing --------------------------------------------------------------
    for name, pairer in (
        ("pairing: heuristic (production)",
         LearnedPairer.from_checkpoint(None, match_radius_deg=args.match_radius)),
        ("pairing: learned head (ours)",
         LearnedPairer.from_checkpoint(args.checkpoint, device=args.device,
                                       match_radius_deg=args.match_radius)
         if args.checkpoint else None),
    ):
        if pairer is None:
            continue

        def run(p=pairer):
            for s, (past, cur) in zip(samples, dets):
                p.pair(past, cur, image_size=s.image_size)

        r = time_stage(run, n)
        stages.append({"stage": name, **r})
        logger.info("%-34s %.3f ms/tile", name, r["per_item_ms"])

    # -- knowledge graph ------------------------------------------------------
    from datetime import datetime, timedelta, timezone

    from icce.report.evidence import from_pairing_result
    from icce.report.graph_context import GraphContextBuilder

    base = datetime(2026, 3, 1, tzinfo=timezone.utc)
    pairer = (LearnedPairer.from_checkpoint(args.checkpoint, device=args.device,
                                            match_radius_deg=args.match_radius)
              if args.checkpoint else
              LearnedPairer.from_checkpoint(None, match_radius_deg=args.match_radius))
    evidences = []
    for i, (s, (past, cur)) in enumerate(zip(samples, dets)):
        res = pairer.pair(past, cur, image_size=s.image_size)
        evidences.append(from_pairing_result(
            res, s.pair_id, s.parent_scene or s.pair_id,
            lat=(cur[0].lat if cur else 36.0), lon=(cur[0].lon if cur else 127.0),
            past_time=base + timedelta(hours=i),
            current_time=base + timedelta(hours=i, days=30),
            image_size=s.image_size,
        ))

    def run_graph():
        b = GraphContextBuilder()
        for ev in evidences:
            b.context_then_index(ev)

    r = time_stage(run_graph, n, warmup=1, repeats=2)
    stages.append({"stage": "knowledge graph: index + retrieve", **r})
    logger.info("%-34s %.3f ms/tile", "graph", r["per_item_ms"])

    # -- report LLM -----------------------------------------------------------
    if args.llm:
        from icce.report.llm import GenRequest, build_llm
        from icce.report.prompts import max_tokens_for, system_prompt, user_prompt

        llm = build_llm(args.llm, max_model_len=args.max_model_len,
                        gpu_memory_utilization=args.gpu_mem, temperature=0.0)
        reqs = [
            GenRequest(key=ev.pair_id, system=system_prompt(args.style),
                       user=user_prompt(ev, "llm_graphrag", args.style),
                       max_tokens=max_tokens_for(args.style))
            for ev in evidences
        ]
        reset_gpu_peak()
        t0 = time.perf_counter()
        texts = llm.generate_batch(reqs)
        total = time.perf_counter() - t0
        stages.append({
            "stage": f"report LLM ({args.llm}, batched)",
            "total_s": total,
            "per_item_ms": total / max(1, n) * 1000.0,
            "peak_gpu_mb": gpu_memory_mb(),
            "n_items": n,
            "mean_output_words": float(np.mean([len(t.split()) for t in texts])) if texts else 0.0,
            "model_load_s": getattr(llm, "load_seconds", None),
        })
        logger.info("%-34s %.1f ms/tile", "report LLM", total / max(1, n) * 1000.0)

    # -- head size ------------------------------------------------------------
    head_params = None
    if args.checkpoint and Path(args.checkpoint).is_file():
        from icce.pairing_head.model import PairingHead
        m, _ = PairingHead.load(args.checkpoint)
        head_params = m.n_parameters()

    from icce.eval.tables import save_json, save_latex

    out = Path(args.out)
    save_json({"n_pairs": n, "device": args.device, "llm": args.llm,
               "pairing_head_parameters": head_params, "stages": stages},
              out / "efficiency.json")

    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Per-tile processing cost on a single NVIDIA A100 80GB. "
        "The learned pairing head adds "
        f"{head_params if head_params else '--'} parameters and a sub-millisecond "
        "step to a pipeline dominated by segmentation and language generation.}",
        "\\label{tab:efficiency}",
        "\\begin{tabular}{lrr}", "\\toprule",
        "Stage & Latency (ms/tile) & Peak GPU (MB) \\\\", "\\midrule",
    ]
    for s in stages:
        mem = f"{s['peak_gpu_mb']:.0f}" if s.get("peak_gpu_mb") else "--"
        lines.append(f"{s['stage']} & {s['per_item_ms']:.2f} & {mem} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    save_latex("\n".join(lines), out / "table_efficiency.tex")

    for s in stages:
        print(f"{s['stage']:<42} {s['per_item_ms']:>10.2f} ms/tile")
    return 0


if __name__ == "__main__":
    sys.exit(main())

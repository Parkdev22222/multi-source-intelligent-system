#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The report experiments, chained, paying for each model exactly once.
#
#   bash scripts/icce_chain.sh
#   STAGES="caption e5 e7" bash scripts/icce_chain.sh
#
# Why this exists next to icce_runpod.sh: run_report_eval given a bare HF id
# loads vLLM in-process, so five invocations mean five 60GB loads. This starts
# one standalone server, runs every text stage against it, swaps to the vision
# model once, and joins the two passes. On an A100 that is the difference
# between ~3 h and ~4 h for the same numbers.
#
#   E3/E4 text -> E5 heuristic arm -> E7 -> [swap] -> vlm_direct -> merge
#
# Each stage writes its own directory and is skipped when that directory
# already holds a result, so a failure costs one stage rather than the night.
#
# Three things here are load-bearing and were learned the hard way:
#
#   * Ports are chosen from a free list, never assumed. RunPod's nginx holds
#     :8001 on this pod.
#   * Readiness means "the endpoint is serving the model we asked for", not
#     "something answered". nginx returns a 502 page with a 200 status, which
#     satisfies `curl -sf` and nothing else.
#   * Stopping vLLM kills by pattern. vllm_server.sh ends in `exec ... | tee`,
#     so the pid we background is the pipeline's, not the server's.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

set -a; [ -f .env ] && . ./.env; set +a
export MSIS_DATA_ROOT="${MSIS_DATA_ROOT:-data/benchmarks}"
export PYTHONPATH="${PYTHONPATH:-.}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"
export SAM3_STRICT="${SAM3_STRICT:-1}"
export TOKENIZERS_PARALLELISM=false

CACHE="${CACHE:-data/cache/levir_cc_test}"
HEAD="${HEAD:-data/checkpoints/pairing_head.pt}"
R="${RESULT_DIR:-results}"
LLM_MODEL="${LLM:-LGAI-EXAONE/EXAONE-4.0-32B}"
VLM_MODEL="${VLM:-Qwen/Qwen2.5-VL-7B-Instruct}"
GPU_MEM="${GPU_MEM:-0.85}"
STAGES="${STAGES:-caption e5 e7 vlm merge}"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
fail() { printf '\n!! %s\n' "$*"; }
runs() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

pick_port() {
  for p in 8000 8005 8010 8011 8012 8013; do
    ss -ltn 2>/dev/null | grep -q ":${p} " || { echo "$p"; return 0; }
  done
  return 1
}

stop_serving() {
  pkill -f "vllm serve" 2>/dev/null
  pkill -f "VLLM::EngineCore" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "VLLM::EngineCore" >/dev/null || break; sleep 3; done
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  sleep 5
}
trap stop_serving EXIT

# serve <model> <limit_mm_image>; sets SERVE_PORT and SERVE_SPEC on success.
serve() {
  local model="$1" mm="${2:-0}" port body
  port="$(pick_port)" || { fail "no free port"; return 1; }
  log "serving ${model} on :${port}"
  MODEL="$model" PORT="$port" GPU_MEM="$GPU_MEM" LIMIT_MM_IMAGE="$mm" \
    bash scripts/vllm_server.sh > "/workspace/.venvs/vllm-chain-${port}.log" 2>&1 &
  for _ in $(seq 1 160); do
    body="$(curl -s --max-time 5 "http://127.0.0.1:${port}/v1/models" 2>/dev/null)"
    case "$body" in *"$model"*) SERVE_PORT="$port"
                                SERVE_SPEC="server:${model}@http://127.0.0.1:${port}/v1"
                                log "ready: ${model}"; return 0;; esac
    pgrep -f "vllm serve" >/dev/null || {
      fail "vLLM exited early -- /workspace/.venvs/vllm-chain-${port}.log"; return 1; }
    sleep 15
  done
  fail "vLLM never served ${model} on :${port}"; return 1
}

# --- preflight ------------------------------------------------------------
# A cache without cache_info.json is a killed run. load_cache accepts one with
# no embeddings.npz on a warning and then hands zeroed CLIP features to every
# experiment below, which is worse than stopping.
if [ ! -e "${CACHE}/cache_info.json" ]; then
  fail "no ${CACHE}/cache_info.json -- the cache is missing or incomplete."
  fail "Run the cache stage first: STAGES=cache bash scripts/icce_runpod.sh"
  exit 1
fi
log "cache: $(wc -l < "${CACHE}/samples.jsonl") crops"

# --- text conditions, one model load -------------------------------------
if runs caption || runs e5 || runs e7; then
  serve "$LLM_MODEL" 0 || exit 1
fi

if runs caption; then
  if [ -e "${R}/levir_cc_caption/report_results_caption.json" ]; then
    log "E3/E4 text conditions already done, skipping"
  else
    log "E3/E4: captioning + factuality, text conditions"
    python -m icce.eval.run_report_eval \
      --cache "$CACHE" --checkpoint "$HEAD" \
      --dataset levir_cc --llm "$SERVE_SPEC" --style caption \
      --modes template llm_raw llm_struct llm_flat_rag llm_graphrag \
      --out "${R}/levir_cc_caption" --device cuda || fail "E3/E4 failed"
  fi
fi

if runs e5; then
  if [ -e "${R}/levir_cc_caption_heuristic_pairing/report_results_caption.json" ]; then
    log "E5 heuristic arm already done, skipping"
  else
    # No --checkpoint: the production heuristic pairs instead. The learned arm
    # is the llm_graphrag row of the run above, so grounding is identical and
    # pairing is the only thing that differs.
    log "E5: heuristic pairing against the same grounding"
    python -m icce.eval.run_report_eval \
      --cache "$CACHE" \
      --dataset levir_cc --llm "$SERVE_SPEC" --style caption \
      --modes llm_graphrag \
      --out "${R}/levir_cc_caption_heuristic_pairing" --device cuda || fail "E5 failed"
  fi
fi

if runs e7; then
  if [ -e "${R}/levir_cc_scene/scene_results.json" ]; then
    log "E7 already done, skipping"
  else
    log "E7: neighbourhood-level grounding, all cached scenes"
    python -m icce.eval.run_scene_eval \
      --cache "$CACHE" --checkpoint "$HEAD" \
      --llm "$SERVE_SPEC" \
      --out "${R}/levir_cc_scene" --device cuda || fail "E7 failed"
  fi
fi

if runs efficiency; then
  # 512 pairs, not the default 64: the head-vs-heuristic delta is a difference
  # of two small numbers and both scale with a tile's detection count.
  log "E6: per-stage latency and memory"
  python -m icce.eval.run_efficiency \
    --cache "$CACHE" --checkpoint "$HEAD" \
    --llm "$SERVE_SPEC" --style report --limit 512 \
    --out "${R}/efficiency" --device cuda || fail "E6 failed"
fi

# --- the vision model, once ----------------------------------------------
if runs vlm; then
  if [ -e "${R}/levir_cc_caption_vlm/report_results_caption.json" ]; then
    log "vlm_direct already done, skipping"
  else
    stop_serving
    serve "$VLM_MODEL" 2 || exit 1
    log "E3/E4: vlm_direct, the external baseline"
    python -m icce.eval.run_report_eval \
      --cache "$CACHE" --checkpoint "$HEAD" \
      --dataset levir_cc --vlm "$SERVE_SPEC" --style caption \
      --modes vlm_direct \
      --out "${R}/levir_cc_caption_vlm" --device cuda || fail "vlm_direct failed"
  fi
fi

stop_serving

# --- join the two passes -------------------------------------------------
if runs merge; then
  if [ -e "${R}/levir_cc_caption_vlm/report_results_caption.json" ] \
     && [ -e "${R}/levir_cc_caption/report_results_caption.json" ]; then
    log "merging vlm_direct into the caption table"
    python -m icce.eval.merge_passes \
      --into "${R}/levir_cc_caption" \
      --from "${R}/levir_cc_caption_vlm" \
      --modes vlm_direct --style caption || fail "merge failed"
  else
    fail "skipping merge: one of the two passes has no result"
  fi
fi

log "CHAIN DONE"
for f in "${R}/levir_cc_caption/report_results_caption.json" \
         "${R}/levir_cc_caption_heuristic_pairing/report_results_caption.json" \
         "${R}/levir_cc_scene/scene_results.json"; do
  [ -e "$f" ] && echo "  ok      $f" || echo "  MISSING $f"
done

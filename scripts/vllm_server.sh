#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Start a standalone vLLM OpenAI-compatible server in its own virtualenv.
#
# Why a separate venv and not just `pip install vllm`:
#   vLLM requires torch>=2.13, numpy>=2 and transformers>=5.
#   SAM3 requires numpy<2, and requirements.txt pins transformers<5.
#   Installing vLLM into the main environment breaks detection. The venv keeps
#   the two stacks apart while they talk over HTTP.
#
#   bash scripts/vllm_server.sh                       # default LLM on :8000
#   PORT=8001 MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
#     LIMIT_MM_IMAGE=2 bash scripts/vllm_server.sh    # VLM on a second port
#
# Then point the harness at it:
#   python -m icce.eval.run_report_eval \
#     --llm "server:LGAI-EXAONE/EXAONE-4.0-32B-Instruct" \
#     --vlm "server:Qwen/Qwen2.5-VL-7B-Instruct@http://127.0.0.1:8001/v1" ...
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-LGAI-EXAONE/EXAONE-4.0-32B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
VENV="${VLLM_VENV:-/workspace/.venvs/vllm}"
GPU_MEM="${GPU_MEM:-0.85}"
MAX_LEN="${MAX_LEN:-8192}"
TP="${TP:-1}"
LIMIT_MM_IMAGE="${LIMIT_MM_IMAGE:-0}"     # >0 for a vision model
LOG="${LOG:-/workspace/.venvs/vllm-${PORT}.log}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

if [ ! -x "${VENV}/bin/vllm" ]; then
  log "creating isolated vLLM venv at ${VENV} (this is a multi-GB install)"
  python -m venv "${VENV}"
  "${VENV}/bin/pip" install --upgrade pip
  # --no-cache-dir: the wheels are large and the pod disk is the scarce resource
  "${VENV}/bin/pip" install --no-cache-dir vllm
fi

# Keep the server's HF cache pointed at the same place the datasets/weights
# already live, so nothing is downloaded twice.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

ARGS=(serve "${MODEL}"
      --host "${HOST}" --port "${PORT}"
      --gpu-memory-utilization "${GPU_MEM}"
      --max-model-len "${MAX_LEN}"
      --tensor-parallel-size "${TP}"
      --trust-remote-code)

if [ "${LIMIT_MM_IMAGE}" -gt 0 ]; then
  ARGS+=(--limit-mm-per-prompt "{\"image\": ${LIMIT_MM_IMAGE}}")
fi

log "serving ${MODEL} on http://${HOST}:${PORT}/v1  (log: ${LOG})"
exec "${VENV}/bin/vllm" "${ARGS[@]}" 2>&1 | tee -a "${LOG}"

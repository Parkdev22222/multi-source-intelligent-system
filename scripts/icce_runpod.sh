#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ICCE-Asia 2026 experiment pipeline, end to end, on a single A100 80GB.
#
#   bash scripts/icce_runpod.sh            # everything
#   bash scripts/icce_runpod.sh cache      # one stage only
#   STAGES="cache train cd" bash scripts/icce_runpod.sh
#
# Every stage is resumable: caches, checkpoints and generation JSONLs are
# skipped when they already exist, so a pre-empted pod costs one stage, not the
# whole run.
# ---------------------------------------------------------------------------
set -euo pipefail

export DOMAIN="${DOMAIN:-urban}"
export MSIS_DATA_ROOT="${MSIS_DATA_ROOT:-data/benchmarks}"
export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM=false

CACHE_DIR="${CACHE_DIR:-data/cache}"
CKPT_DIR="${CKPT_DIR:-data/checkpoints}"
RESULT_DIR="${RESULT_DIR:-results}"
LLM="${LLM:-LGAI-EXAONE/EXAONE-4.0-32B-Instruct}"
LLM_SMALL="${LLM_SMALL:-LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct}"
DEVICE="${DEVICE:-cuda}"

HEAD="${CKPT_DIR}/pairing_head.pt"
STAGES="${STAGES:-${*:-cache train cd transfer caption report efficiency}}"

mkdir -p "$CACHE_DIR" "$CKPT_DIR" "$RESULT_DIR"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
have() { [ -e "$1" ]; }
runs() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# --- 1. detection caches ---------------------------------------------------
if runs cache; then
  for spec in "levir_cd train" "levir_cd val" "levir_cd test" \
              "levir_cc test" "whu_cd test"; do
    set -- $spec
    ds=$1; sp=$2; out="${CACHE_DIR}/${ds}_${sp}"
    if have "${out}/samples.jsonl"; then
      log "cache ${ds}/${sp} exists, skipping"
      continue
    fi
    log "caching SAM3 detections: ${ds}/${sp}"
    extra=""
    [ "$ds" = "levir_cc" ] && extra="--attach-cd-masks"
    python -m icce.eval.cache_detections \
      --dataset "$ds" --split "$sp" --out "$out" --device "$DEVICE" $extra
  done
fi

# --- 2. train the pairing head --------------------------------------------
if runs train; then
  if have "$HEAD"; then
    log "pairing head exists, skipping training"
  else
    log "training the pairing head"
    python -m icce.pairing_head.train \
      --train-cache "${CACHE_DIR}/levir_cd_train" \
      --val-cache   "${CACHE_DIR}/levir_cd_val" \
      --out "$HEAD" --device "$DEVICE" --epochs 60
  fi
fi

# --- 3. E1: change detection on LEVIR-CD ----------------------------------
if runs cd; then
  log "E1: change detection on LEVIR-CD test"
  python -m icce.eval.run_cd_eval \
    --cache "${CACHE_DIR}/levir_cd_test" --checkpoint "$HEAD" \
    --dataset levir_cd --split test \
    --out "${RESULT_DIR}/levir_cd_test" --device "$DEVICE"
fi

# --- 4. E2: zero-shot transfer to WHU-CD ----------------------------------
if runs transfer; then
  log "E2: zero-shot transfer to WHU-CD (head trained on LEVIR-CD only)"
  python -m icce.eval.run_cd_eval \
    --cache "${CACHE_DIR}/whu_cd_test" --checkpoint "$HEAD" \
    --dataset whu_cd --split test \
    --out "${RESULT_DIR}/whu_cd_test" --device "$DEVICE"
fi

# --- 5. E3/E4: captions and factuality on LEVIR-CC ------------------------
if runs caption; then
  log "E3/E4: LEVIR-CC captioning + factuality (${LLM})"
  python -m icce.eval.run_report_eval \
    --cache "${CACHE_DIR}/levir_cc_test" --checkpoint "$HEAD" \
    --dataset levir_cc --llm "$LLM" --style caption \
    --out "${RESULT_DIR}/levir_cc_caption" --device "$DEVICE"

  log "E4b: LLM size ablation (${LLM_SMALL})"
  python -m icce.eval.run_report_eval \
    --cache "${CACHE_DIR}/levir_cc_test" --checkpoint "$HEAD" \
    --dataset levir_cc --llm "$LLM_SMALL" --style caption \
    --out "${RESULT_DIR}/levir_cc_caption_small" --device "$DEVICE"

  log "E4c: pairing ablation held against the same grounding (heuristic pairing)"
  python -m icce.eval.run_report_eval \
    --cache "${CACHE_DIR}/levir_cc_test" \
    --dataset levir_cc --llm "$LLM" --style caption \
    --modes llm_graphrag \
    --out "${RESULT_DIR}/levir_cc_caption_heuristic_pairing" --device "$DEVICE"
fi

# --- 6. full interpretation reports ---------------------------------------
if runs report; then
  log "E5: full urban interpretation reports"
  python -m icce.eval.run_report_eval \
    --cache "${CACHE_DIR}/levir_cc_test" --checkpoint "$HEAD" \
    --dataset levir_cc --llm "$LLM" --style report \
    --out "${RESULT_DIR}/levir_cc_report" --device "$DEVICE"
fi

# --- 7. E6: deployment cost -----------------------------------------------
if runs efficiency; then
  log "E6: per-stage latency and memory"
  python -m icce.eval.run_efficiency \
    --cache "${CACHE_DIR}/levir_cc_test" --checkpoint "$HEAD" \
    --llm "$LLM" --style report \
    --out "${RESULT_DIR}/efficiency" --device "$DEVICE"
fi

log "done. LaTeX tables:"
find "$RESULT_DIR" -name 'table_*.tex' | sort

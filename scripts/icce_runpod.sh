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
# Benchmark runs must never fall back silently: a swallowed SAM3 failure turns
# "zero detections" into a clean exit, and the paper's numbers then describe
# the fallback detector. Override with SAM3_STRICT=0 only for debugging.
export SAM3_STRICT="${SAM3_STRICT:-1}"

CACHE_DIR="${CACHE_DIR:-data/cache}"
CKPT_DIR="${CKPT_DIR:-data/checkpoints}"
RESULT_DIR="${RESULT_DIR:-results}"
# LGAI-EXAONE/EXAONE-4.0-32B-Instruct 404s on the Hub; the instruction-tuned
# release is published under the bare name.
LLM="${LLM:-LGAI-EXAONE/EXAONE-4.0-32B}"
LLM_SMALL="${LLM_SMALL:-LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct}"
VLM="${VLM:-Qwen/Qwen2.5-VL-7B-Instruct}"
DEVICE="${DEVICE:-cuda}"

HEAD="${CKPT_DIR}/pairing_head.pt"
HEAD_NOXF="${CKPT_DIR}/pairing_head_no_xf.pt"
# Whole LEVIR-CC neighbourhoods to cache. 8 scenes = 128 crops ~= 20 min.
CC_SCENES="${CC_SCENES:-8}"
STAGES="${STAGES:-${*:-cache train cd transfer caption report external efficiency}}"

mkdir -p "$CACHE_DIR" "$CKPT_DIR" "$RESULT_DIR"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
have() { [ -e "$1" ]; }
runs() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# --- 0. PILOT -------------------------------------------------------------
# Run this first, on day one. It processes a few dozen tiles instead of a few
# hundred and produces the same tables, so you learn where you actually stand
# against the published numbers before committing a week of GPU time. If the
# pilot says pixel F1 is far below what the paper needs, that is a decision to
# make on day 3, not day 9.
if runs pilot; then
  log "PILOT: small-sample run to get real numbers fast"
  P_CACHE="${CACHE_DIR}/pilot"
  # The LEVIR-CC limit is ignored -- that split is selected by whole scenes.
  for spec in "levir_cd train 60" "levir_cd val 20" "levir_cd test 24" \
              "levir_cc test -"; do
    set -- $spec
    ds=$1; sp=$2; lim=$3; out="${P_CACHE}/${ds}_${sp}"
    have "${out}/cache_info.json" && { log "pilot cache ${ds}/${sp} complete"; continue; }
    have "${out}/samples.jsonl" && { log "pilot cache ${ds}/${sp} partial, redoing"; rm -rf "$out"; }
    # LEVIR-CC is selected by scene, not by --limit: see the cache stage below.
    if [ "$ds" = "levir_cc" ]; then
      python -m icce.eval.cache_detections \
        --dataset "$ds" --split "$sp" --out "$out" --device "$DEVICE" \
        --attach-cd-masks --limit-scenes "$CC_SCENES"
    else
      python -m icce.eval.cache_detections \
        --dataset "$ds" --split "$sp" --limit "$lim" --out "$out" --device "$DEVICE"
    fi
  done

  P_HEAD="${CKPT_DIR}/pilot_head.pt"
  have "$P_HEAD" || python -m icce.pairing_head.train \
    --train-cache "${P_CACHE}/levir_cd_train" \
    --val-cache   "${P_CACHE}/levir_cd_val" \
    --out "$P_HEAD" --device "$DEVICE" --epochs 40

  python -m icce.eval.run_cd_eval \
    --cache "${P_CACHE}/levir_cd_test" --checkpoint "$P_HEAD" \
    --dataset levir_cd --split test \
    --out "${RESULT_DIR}/pilot_levir_cd" --device "$DEVICE"

  # EXAONE-4.0-32B is 64GB of weights, Qwen2.5-VL-7B another 16.6GB: on one
  # 80GB card they cannot be resident together, so the text conditions and the
  # image-conditioned baseline run as two passes.
  #
  # They must not share an --out. run_report_eval writes `results` from the
  # modes of *that* invocation and overwrites the JSON, so a second pass into
  # the same directory replaces the first pass's rows rather than adding to
  # them -- and vlm_direct regenerates rather than reusing its cached
  # generations, so ordering does not save it either. Separate directories,
  # then merge_passes, which checks the two passes describe the same crops
  # before joining them and records per row which pass it came from.
  python -m icce.eval.run_report_eval \
    --cache "${P_CACHE}/levir_cc_test" --checkpoint "$P_HEAD" \
    --dataset levir_cc --llm "$LLM" --style caption \
    --modes template llm_raw llm_struct llm_flat_rag llm_graphrag \
    --out "${RESULT_DIR}/pilot_levir_cc" --device "$DEVICE"

  python -m icce.eval.run_report_eval \
    --cache "${P_CACHE}/levir_cc_test" --checkpoint "$P_HEAD" \
    --dataset levir_cc --vlm "$VLM" --style caption \
    --modes vlm_direct \
    --out "${RESULT_DIR}/pilot_levir_cc_vlm" --device "$DEVICE"

  python -m icce.eval.merge_passes \
    --into "${RESULT_DIR}/pilot_levir_cc" \
    --from "${RESULT_DIR}/pilot_levir_cc_vlm" \
    --modes vlm_direct --style caption

  log "PILOT DONE -- read these before starting the full run:"
  echo "  ${RESULT_DIR}/pilot_levir_cd/cd_results.json"
  echo "  ${RESULT_DIR}/pilot_levir_cc/report_results_caption.json"
  exit 0
fi

# --- 1. detection caches ---------------------------------------------------
if runs cache; then
  for spec in "levir_cd train" "levir_cd val" "levir_cd test" \
              "levir_cc test" "whu_cd test"; do
    set -- $spec
    ds=$1; sp=$2; out="${CACHE_DIR}/${ds}_${sp}"
    # Completion marker is cache_info.json, written last. samples.jsonl appears
    # with the first pair, so treating it as "done" resumes into a truncated
    # cache -- and one with no embeddings.npz, which load_cache accepts with
    # only a warning and then serves zeroed CLIP features to every downstream
    # experiment.
    if have "${out}/cache_info.json"; then
      log "cache ${ds}/${sp} complete, skipping"
      continue
    fi
    if have "${out}/samples.jsonl"; then
      log "cache ${ds}/${sp} is a partial run, redoing it"
      rm -rf "$out"
    fi
    # A dataset that was never downloaded should cost its own stage, not the
    # whole pipeline: WHU-CD needs a manual fetch and set -e would abort here.
    if ! python -c "from icce.datasets.registry import load; load('$ds', split='$sp', limit=1)" \
         >/dev/null 2>&1; then
      log "SKIP ${ds}/${sp}: dataset not available (see icce/README.md -> Datasets)"
      continue
    fi
    log "caching SAM3 detections: ${ds}/${sp}"
    extra=""
    # Unbounded, LEVIR-CC test is 1929 crops (~5 h). Whole scenes rather than a
    # spread sample: --limit leaves ~1 crop per tile, which starves the
    # per-scene knowledge graph of the history E4/E7 are about.
    [ "$ds" = "levir_cc" ] && extra="--attach-cd-masks --limit-scenes ${CC_SCENES}"
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

  # Ablating cross-frame evidence requires retraining without it; zeroing the
  # feature at inference measures a broken model, not the feature.
  if have "$HEAD_NOXF"; then
    log "cross-frame ablation head exists, skipping"
  else
    log "training the cross-frame ablation head"
    python -m icce.pairing_head.train \
      --train-cache "${CACHE_DIR}/levir_cd_train" \
      --val-cache   "${CACHE_DIR}/levir_cd_val" \
      --out "$HEAD_NOXF" --device "$DEVICE" --epochs 60 --no-cross-frame
  fi
fi

# --- 3. E1: change detection on LEVIR-CD ----------------------------------
if runs cd; then
  log "E1: change detection on LEVIR-CD test"
  python -m icce.eval.run_cd_eval \
    --cache "${CACHE_DIR}/levir_cd_test" --checkpoint "$HEAD" \
    --checkpoint-no-xf "$HEAD_NOXF" \
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
  # Two passes for the same reason as the pilot: 60 GB of EXAONE and 16.6 GB of
  # Qwen do not co-exist on an 80 GB card. Asking one invocation for both is
  # how you find that out at hour six of a full run.
  log "E3/E4: LEVIR-CC captioning + factuality, text conditions (${LLM})"
  python -m icce.eval.run_report_eval \
    --cache "${CACHE_DIR}/levir_cc_test" --checkpoint "$HEAD" \
    --dataset levir_cc --llm "$LLM" --style caption \
    --modes template llm_raw llm_struct llm_flat_rag llm_graphrag \
    --out "${RESULT_DIR}/levir_cc_caption" --device "$DEVICE"

  log "E3/E4: the image-conditioned external baseline (${VLM})"
  python -m icce.eval.run_report_eval \
    --cache "${CACHE_DIR}/levir_cc_test" --checkpoint "$HEAD" \
    --dataset levir_cc --vlm "$VLM" --style caption \
    --modes vlm_direct \
    --out "${RESULT_DIR}/levir_cc_caption_vlm" --device "$DEVICE"

  python -m icce.eval.merge_passes \
    --into "${RESULT_DIR}/levir_cc_caption" \
    --from "${RESULT_DIR}/levir_cc_caption_vlm" \
    --modes vlm_direct --style caption

  log "E4b: LLM size ablation (${LLM_SMALL})"
  python -m icce.eval.run_report_eval \
    --cache "${CACHE_DIR}/levir_cc_test" --checkpoint "$HEAD" \
    --dataset levir_cc --llm "$LLM_SMALL" --style caption \
    --out "${RESULT_DIR}/levir_cc_caption_small" --device "$DEVICE"

  # This is E5 as the README defines it: grounding pinned, pairing swapped.
  # The learned arm is the llm_graphrag row of the run above, which uses $HEAD;
  # this run omits --checkpoint so the production heuristic pairs instead.
  log "E5: pairing ablation held against the same grounding (heuristic pairing)"
  python -m icce.eval.run_report_eval \
    --cache "${CACHE_DIR}/levir_cc_test" \
    --dataset levir_cc --llm "$LLM" --style caption \
    --modes llm_graphrag \
    --out "${RESULT_DIR}/levir_cc_caption_heuristic_pairing" --device "$DEVICE"
fi

# --- 6. full interpretation reports ---------------------------------------
if runs report; then
  # Not E5 -- this is the report-style rendering of the same pipeline, kept
  # because the service emits reports rather than captions. E5 is the pairing
  # swap in the caption stage above.
  log "full urban interpretation reports (report style)"
  python -m icce.eval.run_report_eval \
    --cache "${CACHE_DIR}/levir_cc_test" --checkpoint "$HEAD" \
    --dataset levir_cc --llm "$LLM" --style report \
    --out "${RESULT_DIR}/levir_cc_report" --device "$DEVICE"
fi

# --- 6b. external published outputs ---------------------------------------
# Scores a published model's own released captions with Change-Fact-Score.
# Without this, our factuality table has no outside reference point and a
# reviewer can fairly say we invented a metric and only measured ourselves.
# Populate icce/eval/baselines.json -> _external_outputs.slots[].path first.
if runs external; then
  python - <<'PYEOF'
import json, subprocess, sys
from pathlib import Path
slots = json.loads(Path("icce/eval/baselines.json").read_text())["_external_outputs"]["slots"]
todo = [s for s in slots if s.get("path")]
if not todo:
    print("no external prediction files configured; see baselines.json "
          "-> _external_outputs.slots[].path")
    sys.exit(0)
for s in todo:
    subprocess.run([sys.executable, "-m", "icce.eval.score_external",
                    "--predictions", s["path"], "--name", s["name"],
                    "--cache", "data/cache/levir_cc_test",
                    "--out", "results/external"], check=True)
PYEOF
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

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Rebuild the main (detection) environment after a pod restart.
#
# On this pod /workspace is a RunPod network volume and survives a stop; the
# container filesystem (/, /usr/local/lib/python3.12/dist-packages) does not.
# Everything big already lives on /workspace:
#
#   /workspace/sam3                      SAM3 source checkout
#   /workspace/.venvs/vllm               isolated vLLM environment
#   /workspace/.cache/huggingface        SAM3 weights + LEVIR datasets
#   /workspace/multi-source-...          this repo
#
# What is lost is only the pip metadata on the container disk. This script puts
# it back. It does not re-download a single model or dataset.
#
#   bash scripts/setup_env.sh
# ---------------------------------------------------------------------------
set -euo pipefail

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"

log "core dependencies"
# numpy<2 is not optional: sam3 requires it, and pip will happily install
# numpy 2.x as a transitive upgrade and break SAM3 at import time.
# transformers<5 is the pin requirements.txt asks for.
pip install -q \
  "numpy<2" \
  "transformers>=4.56.0,<5" \
  sqlalchemy scipy networkx tqdm Pillow accelerate sentencepiece protobuf \
  huggingface_hub hf_transfer pytest

log "SAM3 runtime deps missing from its own pyproject.toml"
# sam3 imports these but does not declare them (iopath), or declares them only
# in pyproject.toml, which --no-deps below deliberately ignores (ftfy, timm).
pip install -q einops pycocotools iopath "ftfy==6.1.1" "timm>=1.0.17"

log "SAM3 (editable, source stays on the network volume)"
if [ -d /workspace/sam3 ]; then
  pip install -q --no-deps -e /workspace/sam3
else
  echo "!! /workspace/sam3 missing -- re-clone:"
  echo "   git clone https://github.com/facebookresearch/sam3.git /workspace/sam3"
  exit 1
fi

log "verifying"
python - <<'PY'
import numpy, torch, transformers
print(f"  numpy        {numpy.__version__}   (must be <2 for sam3)")
print(f"  torch        {torch.__version__}   cuda={torch.cuda.is_available()}")
print(f"  transformers {transformers.__version__}   (must be <5)")
assert numpy.__version__ < "2", "numpy 2.x will break SAM3"
from sam3.model_builder import build_sam3_image_model      # noqa: F401
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
print("  sam3         imports OK")
PY

if [ -f .env ]; then
  echo "  .env present: $(grep -c SAM3_CHECKPOINT .env) SAM3_CHECKPOINT entry"
else
  echo "!! .env missing -- recreate it with the checkpoint path:"
  echo "   cp .env.example .env"
  echo "   echo 'SAM3_CHECKPOINT=\$(ls -d /workspace/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt)' >> .env"
fi

log "done. vLLM lives in its own venv and needs no reinstall:"
echo "  /workspace/.venvs/vllm/bin/vllm --version"

#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/BIG/semantic-code-reuse}"
ENV_PREFIX="${SEMANTIC_CODE_REUSE_ENV:-$HOME/envs/semantic-code-reuse}"
export HF_HOME=${HF_HOME:-$PROJECT_ROOT/outputs/hf_cache}
if [ -n "${CONDA_EXE:-}" ] && [ -f "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh" ]; then
  source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
  conda activate "$ENV_PREFIX"
elif [ -f "$HOME/.local/opt/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/.local/opt/miniconda3/etc/profile.d/conda.sh"
  conda activate "$ENV_PREFIX"
else
  export PATH="$ENV_PREFIX/bin:$PATH"
fi
cd "$PROJECT_ROOT"

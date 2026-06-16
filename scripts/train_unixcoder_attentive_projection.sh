#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
python -m src.train.train_dual_encoder --config configs/unixcoder_attentive_projection.yaml "$@"

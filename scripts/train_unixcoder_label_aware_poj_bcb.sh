#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
python -m data.scripts.build_poj_bcb_positive_pairs
python -m src.train.train_dual_encoder --config configs/unixcoder_label_aware_paper_poj.yaml "$@"
python -m src.train.train_dual_encoder --config configs/unixcoder_label_aware_poj_bcb.yaml "$@"
python -m src.eval.eval_bigclonebench_siamese --config configs/unixcoder_label_aware_poj_bcb.yaml "$@"

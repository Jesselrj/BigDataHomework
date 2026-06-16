#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
source "$(dirname "$0")/_common.sh"
bash scripts/run_tfidf.sh "$@"
bash scripts/train_unixcoder.sh "$@"
bash scripts/train_unixcoder_label_aware.sh "$@"
bash scripts/train_unixcoder_supcon_ce_k2.sh "$@"
python -m src.run_pipeline --write-final-table

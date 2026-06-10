#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
source "$(dirname "$0")/_common.sh"
python -m src.models.tfidf_baseline --config configs/tfidf.yaml "$@"

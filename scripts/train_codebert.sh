#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
source "$(dirname "$0")/_common.sh"
python -m src.train.train_cross_encoder --config configs/codebert_cls.yaml "$@"

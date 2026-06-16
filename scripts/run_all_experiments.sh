#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
source "$(dirname "$0")/_common.sh"
bash scripts/run_tfidf.sh "$@"
bash scripts/train_codebert.sh "$@"
bash scripts/train_graphcodebert.sh "$@"
bash scripts/train_unixcoder.sh "$@"
bash scripts/train_unixcoder_label_aware.sh "$@"
bash scripts/train_unixcoder_supcon_ce_k2.sh "$@"
bash scripts/run_hybrid_rerank.sh "$@"
bash scripts/train_graphcodebert_hard_negatives.sh "$@"
bash scripts/run_hybrid_rerank_hard.sh "$@"
python -m src.eval.error_analysis --config configs/hybrid_rerank.yaml
python -m src.run_pipeline --write-ablation --write-final-table

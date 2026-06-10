# Semantic Code Reuse Detection

Research-style implementation for semantic code reuse detection on POJ-104. It covers semantic retrieval and pairwise clone classification with TF-IDF, CodeBERT, GraphCodeBERT, UniXcoder, hybrid reranking, and hard-negative ablation.

## Tasks

- Retrieval: given a query program, rank candidate programs that solve the same problem. Primary metric: MAP@R.
- Classification: given two snippets, predict semantic equivalence. Primary metric: F1.

## Environment

All experiments are intended to run on `h100` under `/home/lrj/BIG/semantic-code-reuse`. Use physical GPU `cuda3` only; after `CUDA_VISIBLE_DEVICES=3`, PyTorch sees it as `cuda:0`. Hugging Face downloads use `https://huggingface.co`; the mirror endpoint was tested but is unreachable from h100.

```bash
ssh h100
cd ~/BIG/semantic-code-reuse
source /home/lrj/.local/opt/miniconda3/etc/profile.d/conda.sh
conda activate /home/lrj/envs/semantic-code-reuse
export CUDA_VISIBLE_DEVICES=3
export HF_ENDPOINT=https://huggingface.co
pip install -r requirements.txt
```

Check GPU visibility:

```bash
export CUDA_VISIBLE_DEVICES=3
python - <<PY
import torch
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(0))
PY
```

Expected: one visible GPU, device id `0`, corresponding to physical `cuda3`.

## Dataset Preparation

Download POJ-104 from Hugging Face and convert it to retrieval JSONL:

```bash
python -m data.scripts.download_poj104 --output-dir data/processed
python -m data.scripts.build_pairs --input-dir data/processed --output-dir data/processed
python -m data.scripts.build_hard_negatives --input data/processed/train.jsonl --output data/processed/hard_negatives.jsonl
```

For a local smoke test without the full dataset:

```bash
python -m src.run_pipeline --make-debug-data
bash scripts/run_tfidf.sh --debug
```

## Experiments

```bash
bash scripts/run_tfidf.sh
bash scripts/train_codebert.sh
bash scripts/train_graphcodebert.sh
bash scripts/train_unixcoder.sh
bash scripts/run_hybrid_rerank.sh
bash scripts/run_all_experiments.sh
```

Every script sets `CUDA_VISIBLE_DEVICES=3`, `HF_ENDPOINT=https://huggingface.co`, activates `/home/lrj/envs/semantic-code-reuse`, and writes outputs under `outputs/`.

## Metrics

Retrieval uses Recall@1, Recall@5, Recall@10, MRR, and MAP@R. MAP@R excludes the query itself and averages precision over the top R retrieved items, where R is the number of relevant candidates for that query. Classification uses accuracy, precision, recall, F1, and AUC.

## Results

Final results are written to `outputs/results/final_results.md`. JSON metrics are written to `outputs/results/*_results.json`. Error analysis is written to `outputs/results/error_analysis.md`.

| Method | Task | MAP@R | Recall@1 | Recall@5 | Recall@10 | MRR | F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TF-IDF | Retrieval | 0.5388 | 0.8077 | 0.9140 | 0.9487 | 0.8561 | - | Lexical baseline |
| CodeBERT | Classification | - | - | - | - | - | 0.9117 | Pair classifier |
| GraphCodeBERT | Classification | - | - | - | - | - | 0.9170 | Pair classifier |
| UniXcoder | Retrieval | 0.9811 | 0.9977 | 0.9988 | 0.9992 | 0.9982 | - | Dual encoder |
| UniXcoder + GraphCodeBERT | Retrieval + Rerank | 0.9816 | 0.9979 | 0.9989 | 0.9992 | 0.9984 | - | Hybrid method |
| Hybrid + Hard Negatives | Retrieval + Rerank | 0.9794 | 0.9978 | 0.9983 | 0.9986 | 0.9981 | - | Ablation |

## Error Analysis Summary

`outputs/results/error_analysis.md` includes concrete examples for high lexical similarity with different semantics, low lexical similarity with the same semantics, long snippets truncated by `max_length`, low-overlap same-problem implementations, GraphCodeBERT false negatives on lexically close positive pairs, and hybrid successes where TF-IDF top-1 retrieval fails.

## Limitations

GraphCodeBERT data-flow edges are not explicitly constructed; the implementation uses the pretrained GraphCodeBERT sequence encoder as a cross-encoder over paired code. Full training requires the POJ-104 processed files and the `h100` remote environment. Future work should add explicit data-flow extraction, larger hyperparameter sweeps, and robustness tests on additional clone-detection datasets.

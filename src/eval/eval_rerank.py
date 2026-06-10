from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.models.cross_encoder import CrossEncoder, load_pair_tokenizer, predict_pair_scores
from src.models.dual_encoder import DualEncoder, encode_rows, load_code_tokenizer
from src.utils.io import configure_hf_environment, load_yaml, read_jsonl, write_json
from src.utils.logging import log_environment, setup_logger
from src.utils.metrics import retrieval_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hybrid_rerank.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    configure_hf_environment(cfg)
    logger = setup_logger("hybrid_rerank", cfg["paths"]["logs_dir"])
    log_environment(logger)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = read_jsonl(cfg["paths"]["test_retrieval"])
    if args.debug:
        rows = rows[:128]
    dual_tokenizer = load_code_tokenizer(cfg["dual_encoder_checkpoint"])
    dual = DualEncoder.from_checkpoint(cfg["dual_encoder_checkpoint"]).to(device)
    vectors = encode_rows(dual, dual_tokenizer, rows, device, cfg["max_length"], cfg["eval_batch_size"])
    retrieval_scores = (vectors @ vectors.T).numpy()
    cross_tokenizer = load_pair_tokenizer(cfg["cross_encoder_checkpoint"])
    cross = CrossEncoder.from_checkpoint(cfg["cross_encoder_checkpoint"]).to(device)
    final_scores = retrieval_scores.copy()
    top_k = min(int(cfg["top_k"]), len(rows))
    top_indices = np.argsort(-retrieval_scores, axis=1)[:, :top_k]
    flat_positions = [(qi, int(ci)) for qi in range(len(rows)) for ci in top_indices[qi]]
    batch_size = int(cfg["eval_batch_size"])
    Path(cfg["prediction_file"]).parent.mkdir(parents=True, exist_ok=True)
    with Path(cfg["prediction_file"]).open("w", encoding="utf-8") as out:
        for start in tqdm(range(0, len(flat_positions), batch_size), desc="cross-rerank"):
            positions = flat_positions[start:start + batch_size]
            pairs = [{"code1": rows[qi]["code"], "code2": rows[ci]["code"]} for qi, ci in positions]
            cross_scores = predict_pair_scores(cross, cross_tokenizer, pairs, device, cfg["max_length"], batch_size)
            for (qi, ci), cross_score in zip(positions, cross_scores):
                final_score = float(cfg["alpha"]) * float(retrieval_scores[qi, ci]) + float(cfg["beta"]) * float(cross_score)
                final_scores[qi, ci] = final_score
                out.write(json.dumps({
                    "query_id": rows[qi]["id"],
                    "candidate_id": rows[ci]["id"],
                    "retrieval_score": float(retrieval_scores[qi, ci]),
                    "cross_encoder_score": float(cross_score),
                    "final_score": final_score,
                }, ensure_ascii=False) + "\n")
    metrics = retrieval_metrics(final_scores, [r["id"] for r in rows], [r["problem_id"] for r in rows], [r["id"] for r in rows], [r["problem_id"] for r in rows])
    metrics.update({"method": "UniXcoder + GraphCodeBERT", "alpha": cfg["alpha"], "beta": cfg["beta"], "top_k": top_k})
    write_json(cfg["result_file"], metrics)


if __name__ == "__main__":
    main()

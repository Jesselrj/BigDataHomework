from __future__ import annotations

import argparse
import torch

from src.models.dual_encoder import DualEncoder, encode_rows, load_code_tokenizer
from src.utils.io import configure_hf_environment, load_yaml, read_jsonl, write_json
from src.utils.logging import log_environment, setup_logger
from src.utils.metrics import retrieval_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unixcoder_retrieval.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    configure_hf_environment(cfg)
    logger = setup_logger("unixcoder_eval", cfg["paths"]["logs_dir"])
    log_environment(logger)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = load_code_tokenizer(cfg["checkpoint_dir"])
    model = DualEncoder.from_checkpoint(cfg["checkpoint_dir"], cfg.get("pooling", "mean"), float(cfg.get("temperature", 0.05))).to(device)
    rows = read_jsonl(cfg["paths"]["test_retrieval"])
    if args.debug:
        rows = rows[:128]
    vectors = encode_rows(model, tokenizer, rows, device, cfg["max_length"], cfg["eval_batch_size"])
    scores = (vectors @ vectors.T).numpy()
    metrics = retrieval_metrics(scores, [r["id"] for r in rows], [r["problem_id"] for r in rows], [r["id"] for r in rows], [r["problem_id"] for r in rows])
    write_json(cfg["result_file"], metrics)


if __name__ == "__main__":
    main()

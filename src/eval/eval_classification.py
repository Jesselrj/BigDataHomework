from __future__ import annotations

import argparse
import torch

from src.models.cross_encoder import CrossEncoder, load_pair_tokenizer, predict_pair_scores
from src.utils.io import configure_hf_environment, load_yaml, read_jsonl, write_json, write_jsonl
from src.utils.logging import log_environment, setup_logger
from src.utils.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    configure_hf_environment(cfg)
    logger = setup_logger(f"{cfg['experiment_name']}_eval", cfg["paths"]["logs_dir"])
    log_environment(logger)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = cfg["checkpoint_dir"]
    tokenizer = load_pair_tokenizer(checkpoint)
    model = CrossEncoder.from_checkpoint(checkpoint).to(device)
    rows = read_jsonl(cfg["paths"]["test_pairs"])
    if args.debug:
        rows = rows[:256]
    scores = predict_pair_scores(model, tokenizer, rows, device, cfg["max_length"], cfg["eval_batch_size"])
    metrics = classification_metrics([int(r["label"]) for r in rows], scores)
    write_json(cfg["result_file"], metrics)
    write_jsonl(cfg["prediction_file"], [{**r, "score": s, "prediction": int(s >= 0.5)} for r, s in zip(rows, scores)])


if __name__ == "__main__":
    main()

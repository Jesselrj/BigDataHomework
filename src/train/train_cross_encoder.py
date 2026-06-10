from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data.collators import PairCollator
from src.data.dataset import PairDataset
from src.models.cross_encoder import CrossEncoder, load_pair_tokenizer, predict_pair_scores
from src.utils.io import configure_hf_environment, ensure_dir, load_yaml, read_jsonl, write_json, write_jsonl
from src.utils.logging import log_environment, setup_logger
from src.utils.metrics import classification_metrics
from src.utils.seed import set_seed


def evaluate(model, tokenizer, path: str, cfg: dict, device: torch.device, debug: bool) -> tuple[dict, list[dict]]:
    rows = read_jsonl(path)
    if debug:
        rows = rows[:256]
    scores = predict_pair_scores(model, tokenizer, rows, device, cfg["max_length"], cfg["eval_batch_size"])
    labels = [int(row["label"]) for row in rows]
    metrics = classification_metrics(labels, scores)
    preds = [{**row, "score": score, "prediction": int(score >= 0.5)} for row, score in zip(rows, scores)]
    return metrics, preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--include-hard-negatives", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    configure_hf_environment(cfg)
    set_seed(int(cfg["seed"]))
    logger = setup_logger(cfg["experiment_name"], cfg["paths"]["logs_dir"])
    env = log_environment(logger)
    if env.get("torch_cuda_device_count") not in (None, 1):
        raise RuntimeError("Expected exactly one visible GPU after CUDA_VISIBLE_DEVICES=3")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = load_pair_tokenizer(cfg["model_name"])
    model = CrossEncoder(cfg["model_name"]).to(device)
    train_rows = read_jsonl(cfg["paths"]["train_pairs"])
    if args.include_hard_negatives and Path(cfg["paths"]["hard_negatives"]).exists():
        train_rows.extend(read_jsonl(cfg["paths"]["hard_negatives"]))
    if args.debug:
        train_rows = train_rows[:256]
    tmp_train = Path("outputs/predictions") / f"{cfg['experiment_name']}_train_cache.jsonl"
    write_jsonl(tmp_train, train_rows)
    train_ds = PairDataset(tmp_train)
    train_batch_size = min(int(cfg["cross_encoder_batch_size"]), 8) if args.debug else int(cfg["cross_encoder_batch_size"])
    eval_batch_size = min(int(cfg["eval_batch_size"]), 16) if args.debug else int(cfg["eval_batch_size"])
    loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, collate_fn=PairCollator(tokenizer, cfg["max_length"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    total_steps = max(1, len(loader) * (1 if args.debug else int(cfg["num_epochs"])))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * float(cfg["warmup_ratio"])), total_steps)
    scaler_enabled = cfg.get("precision") == "bf16" and device.type == "cuda"
    start_time = time.time()
    step = 0
    model.train()
    epochs = 1 if args.debug else int(cfg["num_epochs"])
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=scaler_enabled):
                loss = model(**batch).loss / int(cfg["gradient_accumulation_steps"])
            loss.backward()
            if (step + 1) % int(cfg["gradient_accumulation_steps"]) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            step += 1
            if args.debug and step >= 2:
                break
        logger.info("epoch=%s train_loss=%.6f", epoch, sum(losses) / max(len(losses), 1))
    checkpoint_dir = ensure_dir(cfg["checkpoint_dir"])
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    cfg = {**cfg, "eval_batch_size": eval_batch_size}
    metrics, preds = evaluate(model, tokenizer, cfg["paths"]["test_pairs"], cfg, device, args.debug)
    metrics.update({"method": cfg["experiment_name"], "checkpoint": str(checkpoint_dir), "runtime_seconds": time.time() - start_time})
    write_json(cfg["result_file"], metrics)
    write_jsonl(cfg["prediction_file"], preds)
    logger.info("test_metrics=%s", metrics)


if __name__ == "__main__":
    main()

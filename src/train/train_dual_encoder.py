from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data.collators import DualEncoderCollator
from src.data.dataset import DualEncoderPairDataset
from src.models.dual_encoder import DualEncoder, encode_rows, load_code_tokenizer
from src.utils.io import configure_hf_environment, ensure_dir, load_yaml, read_jsonl, write_json
from src.utils.logging import log_environment, setup_logger
from src.utils.metrics import retrieval_metrics
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unixcoder_retrieval.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    configure_hf_environment(cfg)
    set_seed(int(cfg["seed"]))
    logger = setup_logger(cfg["experiment_name"], cfg["paths"]["logs_dir"])
    env = log_environment(logger)
    if env.get("torch_cuda_device_count") not in (None, 1):
        raise RuntimeError("Expected exactly one visible GPU after setting CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = load_code_tokenizer(cfg["model_name"])
    model = DualEncoder(cfg["model_name"], cfg.get("pooling", "mean"), float(cfg.get("temperature", 0.05))).to(device)
    train_ds = DualEncoderPairDataset(cfg["paths"]["train_retrieval"], int(cfg["seed"]), args.debug)
    train_batch_size = min(int(cfg["dual_encoder_batch_size"]), 8) if args.debug else int(cfg["dual_encoder_batch_size"])
    eval_batch_size = min(int(cfg["eval_batch_size"]), 16) if args.debug else int(cfg["eval_batch_size"])
    loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, collate_fn=DualEncoderCollator(tokenizer, cfg["max_length"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    total_steps = max(1, len(loader) * (1 if args.debug else int(cfg["num_epochs"])))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * float(cfg["warmup_ratio"])), total_steps)
    start_time = time.time()
    step = 0
    epochs = 1 if args.debug else int(cfg["num_epochs"])
    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            batch = {outer: {k: v.to(device) for k, v in inner.items()} for outer, inner in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.get("precision") == "bf16" and device.type == "cuda"):
                loss = model(batch["query"], batch["positive"]) / int(cfg["gradient_accumulation_steps"])
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
    rows = read_jsonl(cfg["paths"]["test_retrieval"])
    if args.debug:
        rows = rows[:128]
    vectors = encode_rows(model, tokenizer, rows, device, cfg["max_length"], eval_batch_size)
    scores = (vectors @ vectors.T).numpy()
    metrics = retrieval_metrics(scores, [r["id"] for r in rows], [r["problem_id"] for r in rows], [r["id"] for r in rows], [r["problem_id"] for r in rows])
    metrics.update({"method": "UniXcoder", "checkpoint": str(checkpoint_dir), "runtime_seconds": time.time() - start_time})
    np.savez(cfg["embedding_file"], ids=[r["id"] for r in rows], labels=[r["problem_id"] for r in rows], vectors=vectors.numpy())
    write_json(cfg["result_file"], metrics)
    logger.info("test_metrics=%s", metrics)


if __name__ == "__main__":
    main()

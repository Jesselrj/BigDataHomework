from __future__ import annotations

import argparse
import math
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data.collators import DualEncoderCollator, RetrievalCollator
from src.data.dataset import DualEncoderPairDataset, RetrievalDataset
from src.models.dual_encoder import DualEncoder, encode_rows, load_code_tokenizer
from src.utils.io import configure_hf_environment, ensure_dir, load_yaml, read_jsonl, write_json
from src.utils.logging import log_environment, setup_logger
from src.utils.metrics import retrieval_metrics
from src.utils.seed import set_seed


class BalancedLabelBatchSampler:
    def __init__(self, items: list[dict], batch_size: int, samples_per_label: int, seed: int, label_key: str = "label_id"):
        if batch_size % samples_per_label != 0:
            raise ValueError("batch_size must be divisible by samples_per_label")
        self.batch_size = batch_size
        self.samples_per_label = samples_per_label
        self.labels_per_batch = batch_size // samples_per_label
        self.rng = random.Random(seed)
        self.by_label: dict[str, list[int]] = defaultdict(list)
        for idx, item in enumerate(items):
            self.by_label[str(item[label_key])].append(idx)
        self.labels = sorted(self.by_label)
        self.steps = max(1, math.ceil(len(items) / batch_size))

    def __iter__(self):
        for _ in range(self.steps):
            if len(self.labels) >= self.labels_per_batch:
                labels = self.rng.sample(self.labels, self.labels_per_batch)
            else:
                labels = self.rng.choices(self.labels, k=self.labels_per_batch)
            batch = []
            for label in labels:
                indices = self.by_label[label]
                if len(indices) >= self.samples_per_label:
                    batch.extend(self.rng.sample(indices, self.samples_per_label))
                else:
                    batch.extend(self.rng.choices(indices, k=self.samples_per_label))
            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.steps


@torch.no_grad()
def encode_rows_with_classifier(model, classifier, tokenizer, rows: list[dict], device: torch.device, max_length: int, batch_size: int) -> torch.Tensor:
    model.eval()
    classifier.eval()
    vectors = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch = tokenizer(
            [r["code"] for r in batch_rows],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = classifier(model.encode_batch(batch))
        vectors.append(F.normalize(logits.float(), p=2, dim=-1).cpu())
    return torch.cat(vectors, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unixcoder_retrieval.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    configure_hf_environment(cfg)
    set_seed(int(cfg["seed"]))
    label_aware_loss = bool(cfg.get("label_aware_loss", False))
    instance_contrastive_loss = bool(cfg.get("instance_contrastive_loss", False))
    classification_loss_weight = float(cfg.get("classification_loss_weight", 0.0))
    logger = setup_logger(cfg["experiment_name"], cfg["paths"]["logs_dir"])
    env = log_environment(logger)
    if env.get("torch_cuda_device_count") not in (None, 1):
        raise RuntimeError("Expected exactly one visible GPU after setting CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = load_code_tokenizer(cfg["model_name"])
    model = DualEncoder(cfg["model_name"], cfg.get("pooling", "mean"), float(cfg.get("temperature", 0.05)), bool(cfg.get("projection_head", False)), float(cfg.get("projection_dropout", 0.1))).to(device)
    train_batch_size = min(int(cfg["dual_encoder_batch_size"]), 8) if args.debug else int(cfg["dual_encoder_batch_size"])
    eval_batch_size = min(int(cfg["eval_batch_size"]), 16) if args.debug else int(cfg["eval_batch_size"])
    classifier = None
    if instance_contrastive_loss:
        train_ds = RetrievalDataset(cfg["paths"]["train_retrieval"], args.debug)
        label_to_id = {label: idx for idx, label in enumerate(sorted({row["problem_id"] for row in train_ds.rows}))}
        for row in train_ds.rows:
            row["label_id"] = label_to_id[row["problem_id"]]
        collator = RetrievalCollator(tokenizer, cfg["max_length"])
        if bool(cfg.get("balanced_batch", False)):
            sampler = BalancedLabelBatchSampler(train_ds.rows, train_batch_size, int(cfg.get("samples_per_label", 4)), int(cfg["seed"]))
            loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collator)
        else:
            loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, collate_fn=collator)
        if classification_loss_weight > 0:
            classifier = torch.nn.Linear(int(model.encoder.config.hidden_size), len(label_to_id)).to(device)
    else:
        train_ds = DualEncoderPairDataset(cfg["paths"]["train_retrieval"], int(cfg["seed"]), args.debug)
        collator = DualEncoderCollator(tokenizer, cfg["max_length"])
        if bool(cfg.get("balanced_batch", False)):
            sampler = BalancedLabelBatchSampler(train_ds.pairs, train_batch_size, int(cfg.get("samples_per_label", 4)), int(cfg["seed"]))
            loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collator)
        else:
            loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, collate_fn=collator)
    optimizer_params = list(model.parameters()) + ([] if classifier is None else list(classifier.parameters()))
    optimizer = torch.optim.AdamW(optimizer_params, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    total_steps = max(1, len(loader) * (1 if args.debug else int(cfg["num_epochs"])))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * float(cfg["warmup_ratio"])), total_steps)
    start_time = time.time()
    step = 0
    epochs = 1 if args.debug else int(cfg["num_epochs"])
    model.train()
    if classifier is not None:
        classifier.train()
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            if instance_contrastive_loss:
                labels = batch.pop("labels").to(device)
                inputs = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.get("precision") == "bf16" and device.type == "cuda"):
                    loss, embeddings = model.supervised_contrastive_loss(inputs, labels)
                    if classifier is not None:
                        loss = loss + classification_loss_weight * F.cross_entropy(classifier(embeddings), labels)
                    loss = loss / int(cfg["gradient_accumulation_steps"])
            else:
                batch = {
                    "query": {k: v.to(device) for k, v in batch["query"].items()},
                    "positive": {k: v.to(device) for k, v in batch["positive"].items()},
                    "labels": batch["labels"].to(device),
                }
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.get("precision") == "bf16" and device.type == "cuda"):
                    labels = batch["labels"] if label_aware_loss else None
                    loss = model(batch["query"], batch["positive"], labels) / int(cfg["gradient_accumulation_steps"])
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
    if classifier is not None:
        torch.save(classifier.state_dict(), checkpoint_dir / "classifier_head.pt")
    tokenizer.save_pretrained(str(checkpoint_dir))
    rows = read_jsonl(cfg["paths"]["test_retrieval"])
    if args.debug:
        rows = rows[:128]
    if bool(cfg.get("eval_with_classifier_logits", False)) and classifier is not None:
        vectors = encode_rows_with_classifier(model, classifier, tokenizer, rows, device, cfg["max_length"], eval_batch_size)
    else:
        vectors = encode_rows(model, tokenizer, rows, device, cfg["max_length"], eval_batch_size)
    scores = (vectors @ vectors.T).numpy()
    metrics = retrieval_metrics(scores, [r["id"] for r in rows], [r["problem_id"] for r in rows], [r["id"] for r in rows], [r["problem_id"] for r in rows])
    metrics.update({"method": cfg["experiment_name"], "checkpoint": str(checkpoint_dir), "runtime_seconds": time.time() - start_time})
    np.savez(cfg["embedding_file"], ids=[r["id"] for r in rows], labels=[r["problem_id"] for r in rows], vectors=vectors.numpy())
    write_json(cfg["result_file"], metrics)
    logger.info("test_metrics=%s", metrics)


if __name__ == "__main__":
    main()

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


class SiameseEncoder(nn.Module):
    def __init__(self, model_name: str, pooling: str = "mean", temperature: float = 0.05):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.temperature = temperature

    @classmethod
    def from_checkpoint(cls, checkpoint: str, pooling: str = "mean", temperature: float = 0.05):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.encoder = AutoModel.from_pretrained(checkpoint)
        obj.pooling = pooling
        obj.temperature = temperature
        return obj

    def encode_batch(self, batch: dict) -> torch.Tensor:
        outputs = self.encoder(**batch)
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled, p=2, dim=-1)

    def forward(self, left: dict, right: dict, labels: torch.Tensor | None = None) -> torch.Tensor:
        left_vec = self.encode_batch(left)
        right_vec = self.encode_batch(right)
        cosine = (left_vec * right_vec).sum(dim=-1)
        logits = cosine / self.temperature
        if labels is None:
            return logits
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    def save_pretrained(self, path: str) -> None:
        self.encoder.save_pretrained(path)


def load_code_tokenizer(model_or_checkpoint: str):
    return AutoTokenizer.from_pretrained(model_or_checkpoint)


@torch.no_grad()
def predict_pair_scores(model: SiameseEncoder, tokenizer, pairs: list[dict], device: torch.device, max_length: int, batch_size: int) -> list[float]:
    model.eval()
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch_rows = pairs[start:start + batch_size]
        left = tokenizer(
            [p["code1"] for p in batch_rows],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        right = tokenizer(
            [p["code2"] for p in batch_rows],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        left = {k: v.to(device) for k, v in left.items()}
        right = {k: v.to(device) for k, v in right.items()}
        logits = model(left, right)
        cosine = (logits * model.temperature).clamp(-1.0, 1.0)
        scores.extend(((cosine + 1.0) / 2.0).detach().cpu().tolist())
    return scores

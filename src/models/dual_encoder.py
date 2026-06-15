from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


class DualEncoder(nn.Module):
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

    def forward(self, query: dict, positive: dict, labels: torch.Tensor | None = None) -> torch.Tensor:
        q = self.encode_batch(query)
        p = self.encode_batch(positive)
        logits = q @ p.T / self.temperature
        if labels is None:
            targets = torch.arange(logits.size(0), device=logits.device)
            return F.cross_entropy(logits, targets)
        positive_mask = labels[:, None].eq(labels[None, :])
        numerator = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1)
        denominator = torch.logsumexp(logits, dim=1)
        return -(numerator - denominator).mean()

    def save_pretrained(self, path: str) -> None:
        self.encoder.save_pretrained(path)


def load_code_tokenizer(model_or_checkpoint: str):
    return AutoTokenizer.from_pretrained(model_or_checkpoint)


@torch.no_grad()
def encode_rows(model: DualEncoder, tokenizer, rows: list[dict], device: torch.device, max_length: int, batch_size: int) -> torch.Tensor:
    model.eval()
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
        vectors.append(model.encode_batch(batch).cpu())
    return torch.cat(vectors, dim=0)

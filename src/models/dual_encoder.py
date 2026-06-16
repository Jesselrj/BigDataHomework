from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


class DualEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        temperature: float = 0.05,
        projection_head: bool = False,
        projection_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.temperature = temperature
        self.projection_head_enabled = projection_head
        self.projection_dropout = projection_dropout
        hidden_size = int(self.encoder.config.hidden_size)
        self.attention_pool = nn.Linear(hidden_size, 1) if pooling == "attention" else None
        self.projection_head = (
            nn.Sequential(
                nn.Dropout(projection_dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
            if projection_head
            else None
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        pooling: str = "mean",
        temperature: float = 0.05,
        projection_head: bool = False,
        projection_dropout: float = 0.1,
    ):
        checkpoint_path = Path(checkpoint)
        metadata_path = checkpoint_path / "dual_encoder_config.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            pooling = metadata.get("pooling", pooling)
            temperature = float(metadata.get("temperature", temperature))
            projection_head = bool(metadata.get("projection_head", projection_head))
            projection_dropout = float(metadata.get("projection_dropout", projection_dropout))
        obj = cls(str(checkpoint_path), pooling, temperature, projection_head, projection_dropout)
        attention_path = checkpoint_path / "attention_pool.pt"
        if obj.attention_pool is not None and attention_path.exists():
            obj.attention_pool.load_state_dict(torch.load(attention_path, map_location="cpu"))
        projection_path = checkpoint_path / "projection_head.pt"
        if obj.projection_head is not None and projection_path.exists():
            obj.projection_head.load_state_dict(torch.load(projection_path, map_location="cpu"))
        return obj

    def encode_batch(self, batch: dict) -> torch.Tensor:
        outputs = self.encoder(**batch)
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        elif self.pooling == "attention":
            if self.attention_pool is None:
                raise RuntimeError("attention pooling requested but attention_pool is not initialized")
            mask = batch["attention_mask"].bool()
            scores = self.attention_pool(hidden).squeeze(-1).masked_fill(~mask, -torch.inf)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1)
        else:
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        if self.projection_head is not None:
            pooled = self.projection_head(pooled)
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

    def supervised_contrastive_loss(self, inputs: dict, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode_batch(inputs)
        logits = embeddings @ embeddings.T / self.temperature
        self_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        valid = positive_mask.any(dim=1)
        logits = logits.masked_fill(self_mask, -torch.inf)
        if not bool(valid.any()):
            return embeddings.sum() * 0.0, embeddings
        numerator = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1)
        denominator = torch.logsumexp(logits, dim=1)
        return -(numerator[valid] - denominator[valid]).mean(), embeddings

    def save_pretrained(self, path: str) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(output)
        if self.attention_pool is not None:
            torch.save(self.attention_pool.state_dict(), output / "attention_pool.pt")
        if self.projection_head is not None:
            torch.save(self.projection_head.state_dict(), output / "projection_head.pt")
        (output / "dual_encoder_config.json").write_text(
            json.dumps(
                {
                    "pooling": self.pooling,
                    "temperature": self.temperature,
                    "projection_head": self.projection_head_enabled,
                    "projection_dropout": self.projection_dropout,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


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

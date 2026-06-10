from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoder(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    @classmethod
    def from_checkpoint(cls, checkpoint: str):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        return obj

    def forward(self, **batch):
        return self.model(**batch)

    def save_pretrained(self, path: str) -> None:
        self.model.save_pretrained(path)


def load_pair_tokenizer(model_or_checkpoint: str):
    return AutoTokenizer.from_pretrained(model_or_checkpoint)


@torch.no_grad()
def predict_pair_scores(model: CrossEncoder, tokenizer, pairs: list[dict], device: torch.device, max_length: int, batch_size: int) -> list[float]:
    model.eval()
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        enc = tokenizer(
            [p["code1"] for p in batch],
            [p["code2"] for p in batch],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
        scores.extend(probs.detach().cpu().tolist())
    return scores

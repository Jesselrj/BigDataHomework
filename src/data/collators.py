from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PairCollator:
    tokenizer: object
    max_length: int = 512

    def __call__(self, batch: list[dict]) -> dict:
        enc = self.tokenizer(
            [b["code1"] for b in batch],
            [b["code2"] for b in batch],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        import torch
        enc["labels"] = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
        return enc


@dataclass
class DualEncoderCollator:
    tokenizer: object
    max_length: int = 512

    def __call__(self, batch: list[dict]) -> dict:
        import torch
        query = self.tokenizer(
            [b["query_code"] for b in batch],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        positive = self.tokenizer(
            [b["positive_code"] for b in batch],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"query": query, "positive": positive, "labels": torch.tensor([int(b["label_id"]) for b in batch], dtype=torch.long)}

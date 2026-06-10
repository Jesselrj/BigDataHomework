from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from torch.utils.data import Dataset

from src.utils.io import read_jsonl


class RetrievalDataset(Dataset):
    def __init__(self, path: str | Path, debug: bool = False):
        self.rows = read_jsonl(path)
        if debug:
            self.rows = self.rows[:256]
        for row in self.rows:
            if not {"id", "code", "problem_id"} <= set(row):
                raise ValueError(f"Malformed retrieval row: {row}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class PairDataset(Dataset):
    def __init__(self, path: str | Path, debug: bool = False):
        self.rows = read_jsonl(path)
        if debug:
            self.rows = self.rows[:512]
        for row in self.rows:
            if not {"code1", "code2", "label"} <= set(row):
                raise ValueError(f"Malformed pair row: {row}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class DualEncoderPairDataset(Dataset):
    def __init__(self, path: str | Path, seed: int = 42, debug: bool = False):
        rows = read_jsonl(path)
        if debug:
            rows = rows[:256]
        by_problem: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_problem[row["problem_id"]].append(row)
        self.pairs = []
        rng = random.Random(seed)
        for items in by_problem.values():
            if len(items) < 2:
                continue
            for item in items:
                other = rng.choice([r for r in items if r["id"] != item["id"]])
                self.pairs.append({"query_code": item["code"], "positive_code": other["code"]})
        if not self.pairs:
            raise ValueError("Dual encoder training needs at least one positive pair")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        return self.pairs[idx]

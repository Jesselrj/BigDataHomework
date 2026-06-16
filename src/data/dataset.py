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
        self.pairs = []
        if rows and {"code1", "code2", "label"} <= set(rows[0]):
            label_to_id: dict[str, int] = {}
            for row in rows:
                if int(row["label"]) != 1:
                    continue
                if row.get("problem_id1") and row.get("problem_id1") == row.get("problem_id2"):
                    label_key = "poj:" + str(row["problem_id1"])
                else:
                    label_key = "pair:" + str(row.get("id", len(self.pairs)))
                label_id = label_to_id.setdefault(label_key, len(label_to_id))
                self.pairs.append({"query_code": row["code1"], "positive_code": row["code2"], "label_id": label_id})
            if not self.pairs:
                raise ValueError("Dual encoder pair training needs at least one positive pair")
            return

        by_problem: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_problem[row["problem_id"]].append(row)
        rng = random.Random(seed)
        label_to_id = {problem_id: idx for idx, problem_id in enumerate(sorted(by_problem))}
        for problem_id, items in by_problem.items():
            if len(items) < 2:
                continue
            for item in items:
                other = rng.choice([r for r in items if r["id"] != item["id"]])
                self.pairs.append({"query_code": item["code"], "positive_code": other["code"], "label_id": label_to_id[problem_id]})
        if not self.pairs:
            raise ValueError("Dual encoder training needs at least one positive pair")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        return self.pairs[idx]

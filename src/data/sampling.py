from __future__ import annotations

import random
from collections import defaultdict


def group_by_problem(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["problem_id"]].append(row)
    return grouped


def sample_different_problem(rows: list[dict], problem_id: str, rng: random.Random) -> dict:
    candidates = [row for row in rows if row["problem_id"] != problem_id]
    if not candidates:
        raise ValueError("No negative candidate from a different problem_id")
    return rng.choice(candidates)

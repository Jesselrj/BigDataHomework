from __future__ import annotations

import argparse
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from src.utils.io import read_jsonl, write_jsonl


def build_pairs(rows: list[dict], seed: int = 42, negatives_per_positive: int = 1, max_positive_pairs_per_problem: int = 200) -> list[dict]:
    rng = random.Random(seed)
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not {"id", "code", "problem_id"} <= set(row):
            raise ValueError(f"Malformed retrieval row: {row}")
        by_problem[row["problem_id"]].append(row)
    positives: list[tuple[dict, dict]] = []
    for items in by_problem.values():
        combos = list(combinations(items, 2))
        rng.shuffle(combos)
        positives.extend(combos[:max_positive_pairs_per_problem])
    all_rows = list(rows)
    pairs = []
    pair_id = 0
    for left, right in positives:
        pairs.append(make_pair(pair_id, left, right, 1, "positive"))
        pair_id += 1
        candidates = [row for row in all_rows if row["problem_id"] != left["problem_id"]]
        for _ in range(negatives_per_positive):
            if not candidates:
                continue
            neg = rng.choice(candidates)
            pairs.append(make_pair(pair_id, left, neg, 0, "random_negative"))
            pair_id += 1
    rng.shuffle(pairs)
    return pairs


def make_pair(pair_id: int, left: dict, right: dict, label: int, pair_type: str) -> dict:
    return {
        "id": f"pair_{pair_id:08d}",
        "code1": left["code"],
        "code2": right["code"],
        "problem_id1": left["problem_id"],
        "problem_id2": right["problem_id"],
        "label": int(label),
        "pair_type": pair_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--max-positive-pairs-per-problem", type=int, default=200)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    for split in ("train", "validation", "test"):
        path = Path(args.input_dir) / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        if args.debug:
            rows = rows[:256]
        pairs = build_pairs(rows, args.seed, args.negatives_per_positive, args.max_positive_pairs_per_problem)
        write_jsonl(Path(args.output_dir) / f"{split}_pairs.jsonl", pairs)
        print(split, len(pairs))


if __name__ == "__main__":
    main()

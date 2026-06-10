from __future__ import annotations

import argparse
import random
from bisect import bisect_left

from src.models.tfidf_baseline import code_tokenizer
from src.utils.io import read_jsonl, write_jsonl


def token_count(code: str) -> int:
    return len(code_tokenizer(code))


def lexical_hard_negatives(rows: list[dict], per_sample: int) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    corpus = [" ".join(code_tokenizer(r["code"])) for r in rows]
    matrix = TfidfVectorizer(min_df=1).fit_transform(corpus)
    n_neighbors = min(len(rows), max(per_sample * 20, 100))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix)
    pairs = []
    for i, row in enumerate(rows):
        picked = 0
        for distance, j in zip(distances[i], indices[i]):
            if i == j or row["problem_id"] == rows[j]["problem_id"]:
                continue
            pairs.append(make_negative(len(pairs), row, rows[j], "lexical_hard_negative", float(1.0 - distance)))
            picked += 1
            if picked >= per_sample:
                break
    return pairs


def length_hard_negatives(rows: list[dict], per_sample: int) -> list[dict]:
    enriched = sorted((token_count(r["code"]), i, r) for i, r in enumerate(rows))
    counts = [item[0] for item in enriched]
    pairs = []
    for count, _, row in enriched:
        center = bisect_left(counts, count)
        left = center - 1
        right = center + 1
        candidates = []
        while (left >= 0 or right < len(enriched)) and len(candidates) < per_sample:
            left_diff = abs(count - enriched[left][0]) if left >= 0 else float("inf")
            right_diff = abs(count - enriched[right][0]) if right < len(enriched) else float("inf")
            if left_diff <= right_diff:
                other_count, _, other = enriched[left]
                left -= 1
            else:
                other_count, _, other = enriched[right]
                right += 1
            if other["problem_id"] != row["problem_id"]:
                candidates.append((abs(count - other_count), other))
        for diff, other in candidates:
            pairs.append(make_negative(len(pairs), row, other, "length_structure_hard_negative", -float(diff)))
    return pairs


def random_negatives(rows: list[dict], per_sample: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_other_problem = {}
    pairs = []
    for row in rows:
        candidates = by_other_problem.get(row["problem_id"])
        if candidates is None:
            candidates = [other for other in rows if other["problem_id"] != row["problem_id"]]
            by_other_problem[row["problem_id"]] = candidates
        for _ in range(per_sample):
            other = rng.choice(candidates)
            pairs.append(make_negative(len(pairs), row, other, "random_negative", 0.0))
    return pairs


def make_negative(pair_id: int, left: dict, right: dict, pair_type: str, score: float) -> dict:
    return {
        "id": f"hard_pair_{pair_id:08d}",
        "code1": left["code"],
        "code2": right["code"],
        "problem_id1": left["problem_id"],
        "problem_id2": right["problem_id"],
        "label": 0,
        "pair_type": pair_type,
        "hard_negative_score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train.jsonl")
    parser.add_argument("--output", default="data/processed/hard_negatives.jsonl")
    parser.add_argument("--per-sample", type=int, default=2)
    parser.add_argument("--random-per-sample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    if args.debug:
        rows = rows[:256]
    if len({r["problem_id"] for r in rows}) < 2:
        raise ValueError("Hard negatives require at least two problem_id values")
    pairs = (
        random_negatives(rows, args.random_per_sample, args.seed)
        + lexical_hard_negatives(rows, args.per_sample)
        + length_hard_negatives(rows, args.per_sample)
    )
    write_jsonl(args.output, pairs)
    print(f"wrote {len(pairs)} hard negatives to {args.output}")


if __name__ == "__main__":
    main()

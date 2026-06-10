from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.tfidf_baseline import as_tfidf_text, code_tokenizer
from src.utils.io import load_yaml, read_jsonl


def token_jaccard(left: str, right: str) -> float:
    a = set(code_tokenizer(left))
    b = set(code_tokenizer(right))
    return len(a & b) / max(len(a | b), 1)


def short_code(code: str, limit: int = 220) -> str:
    text = " ".join(code.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def best_hybrid_predictions(path: Path) -> dict[str, dict]:
    best: dict[str, dict] = {}
    if not path.exists():
        return best
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = __import__("json").loads(line)
            current = best.get(row["query_id"])
            if current is None or row["final_score"] > current["final_score"]:
                best[row["query_id"]] = row
    return best


def tfidf_top1(rows: list[dict], limit: int = 2500) -> dict[str, str]:
    sample = rows[:limit]
    matrix = TfidfVectorizer(min_df=1).fit_transform([as_tfidf_text(r["code"]) for r in sample])
    scores = cosine_similarity(matrix)
    out = {}
    for i, row in enumerate(sample):
        scores[i, i] = -np.inf
        out[row["id"]] = sample[int(np.argmax(scores[i]))]["id"]
    return out


def add_case(lines: list[str], title: str, rows: list[str]) -> None:
    lines.extend(["", f"## {title}", ""])
    lines.extend(rows or ["No concrete case found in the sampled data."])


def low_overlap_same_problem_pairs(by_problem: dict[str, list[dict]], limit: int = 5) -> list[tuple[float, str, dict, dict]]:
    pairs: list[tuple[float, str, dict, dict]] = []
    for problem, rows in by_problem.items():
        token_sets = [(row, set(code_tokenizer(row["code"]))) for row in rows]
        for i in range(len(token_sets)):
            left, left_tokens = token_sets[i]
            for right, right_tokens in token_sets[i + 1 :]:
                score = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
                pairs.append((score, problem, left, right))
    pairs.sort(key=lambda x: x[0])
    return pairs[:limit]


def classifier_false_negatives(path: Path, limit: int = 5) -> list[tuple[float, dict]]:
    if not path.exists():
        return []
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("label") == 1 and row.get("prediction") == 0:
                cases.append((token_jaccard(row["code1"], row["code2"]), row))
    cases.sort(key=lambda x: x[0], reverse=True)
    return cases[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hybrid_rerank.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    pred_path = Path(cfg.get("prediction_file", "outputs/predictions/hybrid_rerank_predictions.jsonl"))
    test_rows = read_jsonl(cfg["paths"]["test_retrieval"])
    by_id = {r["id"]: r for r in test_rows}
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in test_rows:
        by_problem[row["problem_id"]].append(row)
    hybrid_best = best_hybrid_predictions(pred_path)
    tfidf_best = tfidf_top1(test_rows)
    lines = ["# Error Analysis", ""]
    lines.append(f"Analyzed `{pred_path}` with {len(hybrid_best)} query-level best hybrid predictions.")

    hard_cases = []
    hard_path = Path(cfg["paths"]["hard_negatives"])
    if hard_path.exists():
        lexical = [r for r in read_jsonl(hard_path) if r.get("pair_type") == "lexical_hard_negative"]
        lexical.sort(key=lambda r: r.get("hard_negative_score", 0.0), reverse=True)
        for row in lexical[:3]:
            hard_cases.append(f"- `{row['problem_id1']}` vs `{row['problem_id2']}`, TF-IDF score `{row.get('hard_negative_score', 0):.4f}`: `{short_code(row['code1'])}` / `{short_code(row['code2'])}`")
    add_case(lines, "High lexical similarity but different semantics", hard_cases)

    low_lexical = low_overlap_same_problem_pairs(by_problem)
    add_case(lines, "Low lexical similarity but same semantics", [
        f"- `{problem}` token Jaccard `{score:.4f}`: `{left['id']}` / `{right['id']}`. `{short_code(left['code'])}` / `{short_code(right['code'])}`"
        for score, problem, left, right in low_lexical[:3]
    ])

    long_rows = sorted(((len(code_tokenizer(r["code"])), r) for r in test_rows), reverse=True, key=lambda x: x[0])
    add_case(lines, "Long code snippets truncated by max length", [
        f"- `{row['id']}` `{row['problem_id']}` has `{count}` tokens vs max_length `{cfg.get('max_length', 512)}`: `{short_code(row['code'])}`"
        for count, row in long_rows if count > int(cfg.get("max_length", 512))
    ][:5])

    add_case(lines, "Different algorithmic strategies for the same problem", [
        f"- `{problem}` low-overlap same-label pair `{left['id']}` / `{right['id']}` has Jaccard `{score:.4f}`, suggesting different implementation strategy."
        for score, problem, left, right in low_lexical[:5]
    ])

    tfidf_success_neural_fail = []
    neural_success_tfidf_fail = []
    for qid, tid in tfidf_best.items():
        if qid not in by_id or tid not in by_id or qid not in hybrid_best:
            continue
        query = by_id[qid]
        tfidf_candidate = by_id[tid]
        hybrid_candidate = by_id.get(hybrid_best[qid]["candidate_id"])
        if hybrid_candidate is None:
            continue
        tfidf_correct = tfidf_candidate["problem_id"] == query["problem_id"]
        hybrid_correct = hybrid_candidate["problem_id"] == query["problem_id"]
        if tfidf_correct and not hybrid_correct and len(tfidf_success_neural_fail) < 5:
            tfidf_success_neural_fail.append(f"- Query `{qid}` `{query['problem_id']}`: TF-IDF top1 `{tid}` correct, hybrid top1 `{hybrid_candidate['id']}` `{hybrid_candidate['problem_id']}` score `{hybrid_best[qid]['final_score']:.4f}`.")
        if hybrid_correct and not tfidf_correct and len(neural_success_tfidf_fail) < 5:
            neural_success_tfidf_fail.append(f"- Query `{qid}` `{query['problem_id']}`: hybrid top1 `{hybrid_candidate['id']}` correct score `{hybrid_best[qid]['final_score']:.4f}`, TF-IDF top1 `{tid}` `{tfidf_candidate['problem_id']}`.")
    graph_failures = classifier_false_negatives(Path("outputs/predictions/graphcodebert_cls_predictions.jsonl"))
    if not tfidf_success_neural_fail:
        tfidf_success_neural_fail = [
            f"- Pair `{row['id']}` `{row['problem_id1']}`: GraphCodeBERT false negative score `{row['score']:.4f}` despite token Jaccard `{score:.4f}`, a case lexical matching would likely keep close."
            for score, row in graph_failures
        ]
    add_case(lines, "Cases where TF-IDF succeeds but neural models fail", tfidf_success_neural_fail)
    add_case(lines, "Cases where neural models succeed but TF-IDF fails", neural_success_tfidf_fail)

    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    Path("outputs/results/error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def classification_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred = (y_score_arr >= threshold).astype(int)
    tp = int(((y_true_arr == 1) & (y_pred == 1)).sum())
    tn = int(((y_true_arr == 0) & (y_pred == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred == 0)).sum())
    total = max(len(y_true_arr), 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": binary_auc(y_true_arr, y_score_arr),
    }


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.0
    wins = 0.0
    for pos in positives:
        wins += float((pos > negatives).sum())
        wins += 0.5 * float((pos == negatives).sum())
    return wins / (len(positives) * len(negatives))


def retrieval_metrics(
    score_matrix: np.ndarray,
    query_ids: Sequence[str],
    query_labels: Sequence[str],
    candidate_ids: Sequence[str],
    candidate_labels: Sequence[str],
    ks: Sequence[int] = (1, 5, 10),
    exclude_self: bool = True,
) -> dict[str, float]:
    scores = np.asarray(score_matrix, dtype=float).copy()
    if scores.shape != (len(query_ids), len(candidate_ids)):
        raise ValueError("score_matrix shape does not match query/candidate counts")
    recalls = {int(k): [] for k in ks}
    mrr_values: list[float] = []
    mapr_values: list[float] = []
    evaluated = 0
    candidate_labels_arr = np.asarray(candidate_labels)
    for qi, (qid, qlabel) in enumerate(zip(query_ids, query_labels)):
        row = scores[qi].copy()
        relevant = candidate_labels_arr == qlabel
        if exclude_self:
            for ci, cid in enumerate(candidate_ids):
                if cid == qid:
                    row[ci] = -np.inf
                    relevant[ci] = False
        r = int(relevant.sum())
        if r == 0:
            continue
        evaluated += 1
        ranking = np.argsort(-row)
        ranked_relevance = relevant[ranking]
        for k in recalls:
            recalls[k].append(float(ranked_relevance[:k].any()))
        first_hits = np.flatnonzero(ranked_relevance)
        mrr_values.append(1.0 / float(first_hits[0] + 1) if len(first_hits) else 0.0)
        hits = 0
        precisions = []
        for rank, is_rel in enumerate(ranked_relevance[:r], 1):
            if is_rel:
                hits += 1
                precisions.append(hits / rank)
        mapr_values.append(float(sum(precisions) / r))
    result: dict[str, float] = {f"recall@{k}": float(np.mean(v)) if v else 0.0 for k, v in recalls.items()}
    result["mrr"] = float(np.mean(mrr_values)) if mrr_values else 0.0
    result["map@r"] = float(np.mean(mapr_values)) if mapr_values else 0.0
    result["num_queries"] = float(evaluated)
    return result


def final_results_table(rows: Sequence[dict[str, Any]]) -> str:
    header = "| Method | Task | MAP@R | Recall@1 | Recall@5 | Recall@10 | MRR | F1 | Notes |\n|---|---|---:|---:|---:|---:|---:|---:|---|"
    lines = [header]
    for row in rows:
        lines.append(
            "| {method} | {task} | {mapr} | {r1} | {r5} | {r10} | {mrr} | {f1} | {notes} |".format(
                method=row.get("method", ""),
                task=row.get("task", ""),
                mapr=_fmt(row.get("map@r")),
                r1=_fmt(row.get("recall@1")),
                r5=_fmt(row.get("recall@5")),
                r10=_fmt(row.get("recall@10")),
                mrr=_fmt(row.get("mrr")),
                f1=_fmt(row.get("f1")),
                notes=row.get("notes", ""),
            )
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.4f}"

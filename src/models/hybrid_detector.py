from __future__ import annotations

import numpy as np


def combine_scores(retrieval_scores: np.ndarray, cross_scores: np.ndarray, alpha: float = 0.4, beta: float = 0.6) -> np.ndarray:
    if retrieval_scores.shape != cross_scores.shape:
        raise ValueError("retrieval_scores and cross_scores must have the same shape")
    return alpha * retrieval_scores + beta * cross_scores

import numpy as np

from src.utils.metrics import classification_metrics, retrieval_metrics


def test_classification_metrics():
    m = classification_metrics([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.7])
    assert round(m["accuracy"], 2) == 0.50
    assert round(m["precision"], 2) == 0.50
    assert round(m["recall"], 2) == 0.50
    assert round(m["f1"], 2) == 0.50


def test_retrieval_metrics_excludes_self_and_computes_mapr():
    ids = ["a1", "a2", "b1", "b2"]
    labels = ["a", "a", "b", "b"]
    scores = np.array([
        [9.0, 8.0, 1.0, 0.0],
        [8.0, 9.0, 1.0, 0.0],
        [0.0, 1.0, 9.0, 8.0],
        [0.0, 1.0, 8.0, 9.0],
    ])
    m = retrieval_metrics(scores, ids, labels, ids, labels)
    assert m["recall@1"] == 1.0
    assert m["mrr"] == 1.0
    assert m["map@r"] == 1.0

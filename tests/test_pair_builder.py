from data.scripts.build_pairs import build_pairs


def test_build_pairs_labels_are_consistent():
    rows = [
        {"id": "a1", "code": "a", "problem_id": "p1"},
        {"id": "a2", "code": "aa", "problem_id": "p1"},
        {"id": "b1", "code": "b", "problem_id": "p2"},
        {"id": "b2", "code": "bb", "problem_id": "p2"},
    ]
    pairs = build_pairs(rows, negatives_per_positive=1)
    assert any(p["label"] == 1 and p["problem_id1"] == p["problem_id2"] for p in pairs)
    assert any(p["label"] == 0 and p["problem_id1"] != p["problem_id2"] for p in pairs)

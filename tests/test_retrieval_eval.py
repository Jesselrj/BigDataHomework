from src.models.tfidf_baseline import evaluate


def test_tfidf_eval_runs_on_debug_rows():
    rows = [
        {"id": "a1", "code": "int add(int a,int b){return a+b;}", "problem_id": "p1"},
        {"id": "a2", "code": "int sum(int x,int y){return x+y;}", "problem_id": "p1"},
        {"id": "b1", "code": "int mx(int a,int b){return a>b?a:b;}", "problem_id": "p2"},
        {"id": "b2", "code": "int max2(int x,int y){return x>y?x:y;}", "problem_id": "p2"},
    ]
    metrics = evaluate(rows, {"ngram_range": [1, 2]})
    assert "map@r" in metrics
    assert metrics["num_queries"] == 4.0

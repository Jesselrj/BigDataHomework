from __future__ import annotations

import argparse
from pathlib import Path

from data.scripts.build_pairs import build_pairs
from src.utils.io import read_json, write_jsonl
from src.utils.metrics import final_results_table


def make_debug_data() -> None:
    snippets = {
        "problem_add": ["int main(){int a,b;cin>>a>>b;cout<<a+b;}", "int main(){long x,y;scanf(\"%ld%ld\",&x,&y);printf(\"%ld\",x+y);}", "int main(){int s=0,x;while(cin>>x)s+=x;cout<<s;}"],
        "problem_max": ["int main(){int a,b;cin>>a>>b;cout<<max(a,b);}", "int main(){int x,y;scanf(\"%d%d\",&x,&y);printf(\"%d\",x>y?x:y);}", "int main(){vector<int> v(2);cin>>v[0]>>v[1];sort(v.begin(),v.end());cout<<v[1];}"],
        "problem_sort": ["int main(){int n;cin>>n;vector<int>a(n);for(int&i:a)cin>>i;sort(a.begin(),a.end());}", "int main(){int n;scanf(\"%d\",&n);int a[100];for(int i=0;i<n;i++)scanf(\"%d\",a+i);sort(a,a+n);}", "int main(){array<int,3>a;for(auto &x:a)cin>>x;sort(a.begin(),a.end());}"],
    }
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        rows = []
        idx = 0
        for problem, codes in snippets.items():
            for code in codes:
                rows.append({"id": f"{split}_{idx}", "code": code, "problem_id": problem, "language": "cpp"})
                idx += 1
        write_jsonl(f"data/processed/{split}.jsonl", rows)
        write_jsonl(f"data/processed/{split}_pairs.jsonl", build_pairs(rows, max_positive_pairs_per_problem=3))


def write_final_table() -> None:
    result_specs = [
        ("TF-IDF", "Retrieval", "outputs/results/tfidf_results.json", "Lexical baseline"),
        ("CodeBERT", "Classification", "outputs/results/codebert_cls_results.json", "Pair classifier"),
        ("GraphCodeBERT", "Classification", "outputs/results/graphcodebert_cls_results.json", "Pair classifier"),
        ("UniXcoder", "Retrieval", "outputs/results/unixcoder_retrieval_results.json", "Dual encoder"),
        ("UniXcoder + GraphCodeBERT", "Retrieval + Rerank", "outputs/results/hybrid_rerank_results.json", "Hybrid method"),
        ("Hybrid + Hard Negatives", "Retrieval + Rerank", "outputs/results/hybrid_rerank_hard_results.json", "Ablation"),
    ]
    rows = []
    for method, task, path, notes in result_specs:
        metrics = {}
        if Path(path).exists():
            metrics = read_json(path)
        rows.append({**metrics, "method": method, "task": task, "notes": notes})
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    Path("outputs/results/final_results.md").write_text(final_results_table(rows), encoding="utf-8")


def write_ablation() -> None:
    without = _read_result_if_exists("outputs/results/hybrid_rerank_results.json")
    with_hard = _read_result_if_exists("outputs/results/hybrid_rerank_hard_results.json")
    out = {
        "without_hard_negatives": without,
        "with_hard_negatives": with_hard,
        "delta_with_minus_without": {
            key: with_hard[key] - without[key]
            for key in ("map@r", "recall@1", "recall@5", "recall@10", "mrr")
            if isinstance(without.get(key), (int, float)) and isinstance(with_hard.get(key), (int, float))
        },
    }
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    Path("outputs/results/hard_negative_ablation.json").write_text(__import__("json").dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_result_if_exists(path: str) -> dict:
    if Path(path).exists():
        return read_json(path)
    return {"status": "missing", "path": path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-debug-data", action="store_true")
    parser.add_argument("--write-final-table", action="store_true")
    parser.add_argument("--write-ablation", action="store_true")
    args = parser.parse_args()
    if args.make_debug_data:
        make_debug_data()
    if args.write_final_table:
        write_final_table()
    if args.write_ablation:
        write_ablation()


if __name__ == "__main__":
    main()

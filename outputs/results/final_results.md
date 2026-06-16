| Method | Task | MAP@R | Recall@1 | Recall@5 | Recall@10 | MRR | F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TF-IDF | Retrieval | 0.2169 | 0.8077 | 0.9140 | 0.9487 | 0.8561 | - | Lexical baseline |
| UniXcoder | Retrieval | 0.9098 | 0.9976 | 0.9988 | 0.9992 | 0.9982 | - | Baseline dual encoder |
| UniXcoder + Label-aware Loss | Retrieval | 0.9117 | 0.9982 | 0.9989 | 0.9992 | 0.9986 | - | Ours: mask same-problem false negatives |
| UniXcoder + SupCon CE (k=2) | Retrieval | 0.9254 | 0.9981 | 0.9991 | 0.9996 | 0.9986 | - | Ours: supervised contrastive training with class proxy |

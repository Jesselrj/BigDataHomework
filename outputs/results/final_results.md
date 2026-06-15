| Method | Task | MAP@R | Recall@1 | Recall@5 | Recall@10 | MRR | F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TF-IDF | Retrieval | 0.2169 | 0.8077 | 0.9140 | 0.9487 | 0.8561 | - | Lexical baseline |
| CodeBERT | Classification | - | - | - | - | - | 0.9117 | Pair classifier |
| GraphCodeBERT | Classification | - | - | - | - | - | 0.9170 | Pair classifier |
| UniXcoder | Retrieval | 0.9117 | 0.9982 | 0.9989 | 0.9992 | 0.9986 | - | Label-aware dual encoder |
| UniXcoder + GraphCodeBERT | Retrieval + Rerank | 0.9053 | 0.9979 | 0.9989 | 0.9992 | 0.9984 | - | Hybrid method |
| Hybrid + Hard Negatives | Retrieval + Rerank | 0.8884 | 0.9978 | 0.9983 | 0.9986 | 0.9981 | - | Ablation |

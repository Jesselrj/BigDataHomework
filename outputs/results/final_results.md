| Method | Task | MAP@R | Recall@1 | Recall@5 | Recall@10 | MRR | F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TF-IDF | Retrieval | 0.2169 | 0.8077 | 0.9140 | 0.9487 | 0.8561 | - | Lexical baseline |
| CodeBERT | Classification | - | - | - | - | - | 0.9117 | Pair classifier |
| GraphCodeBERT | Classification | - | - | - | - | - | 0.9170 | Pair classifier |
| UniXcoder | Retrieval | 0.9098 | 0.9976 | 0.9988 | 0.9993 | 0.9982 | - | Local baseline dual encoder |
| UniXcoder + Label-aware Loss | Retrieval | 0.9117 | 0.9982 | 0.9989 | 0.9992 | 0.9986 | - | Ours: mask same-problem false negatives |
| UniXcoder + GraphCodeBERT | Retrieval + Rerank | 0.9053 | 0.9979 | 0.9989 | 0.9992 | 0.9984 | - | Hybrid method |
| Hybrid + Hard Negatives | Retrieval + Rerank | 0.8884 | 0.9978 | 0.9983 | 0.9986 | 0.9981 | - | Ablation |
| UniXcoder (paper) | POJ-104 / BCB | 0.9052 | - | - | - | - | 0.9520 | Paper metrics: BCB R=0.9290, P=0.9760 |
| UniXcoder + Label-aware Loss (paper POJ setting) | POJ-104 | 0.8931 | 0.9959 | 0.9984 | 0.9989 | 0.9969 | - | Ours, POJ train only |
| UniXcoder + Label-aware Loss (POJ+BCB train) | POJ-104 / BCB | 0.8925 | 0.9972 | 0.9985 | 0.9988 | 0.9978 | 0.7971 | Ours; BCB validation-calibrated threshold, fixed-threshold F1=0.2406 |

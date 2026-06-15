# 语义代码复用检测

本仓库是 POJ-104 数据集上的语义代码复用检测实验代码。

项目包含 TF-IDF、CodeBERT、GraphCodeBERT、UniXcoder、混合重排序和 hard negative 实验。

## 文件结构

```text
semantic-code-reuse/
├── configs/          # 实验配置
├── data/scripts/     # 数据处理脚本
├── scripts/          # 训练和评测脚本
├── src/              # 主要代码
├── tests/            # 单元测试
├── outputs/results/  # 实验结果
└── requirements.txt
```

## 环境配置

```bash
conda create -n semantic-code-reuse python=3.10 -y
conda activate semantic-code-reuse
pip install -r requirements.txt
```

如果使用 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0
```

进入项目目录：

```bash
cd /path/to/semantic-code-reuse
```

## 数据准备

下载并处理 POJ-104：

```bash
python -m data.scripts.download_poj104 --output-dir data/processed
python -m data.scripts.build_pairs --input-dir data/processed --output-dir data/processed
python -m data.scripts.build_hard_negatives \
  --input data/processed/train.jsonl \
  --output data/processed/hard_negatives.jsonl
```

处理后应包含：

```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl
data/processed/train_pairs.jsonl
data/processed/validation_pairs.jsonl
data/processed/test_pairs.jsonl
data/processed/hard_negatives.jsonl
```

## 运行实验

快速测试：

```bash
python -m src.run_pipeline --make-debug-data
bash scripts/run_tfidf.sh --debug
pytest -q
```

分别运行各个实验：

```bash
bash scripts/run_tfidf.sh
bash scripts/train_codebert.sh
bash scripts/train_graphcodebert.sh
bash scripts/train_unixcoder.sh
bash scripts/run_hybrid_rerank.sh
bash scripts/train_graphcodebert_hard_negatives.sh
bash scripts/run_hybrid_rerank_hard.sh
```

一键运行全部实验：

```bash
bash scripts/run_all_experiments.sh
```

输出文件保存在：

```text
outputs/checkpoints/
outputs/logs/
outputs/predictions/
outputs/results/
```

## 实验结果

| 方法 | 任务 | MAP@R | Recall@1 | Recall@5 | Recall@10 | MRR | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 检索 | 0.5388 | 0.8077 | 0.9140 | 0.9487 | 0.8561 | - |
| CodeBERT | 分类 | - | - | - | - | - | 0.9117 |
| GraphCodeBERT | 分类 | - | - | - | - | - | 0.9170 |
| UniXcoder | 检索 | 0.9811 | 0.9977 | 0.9988 | 0.9992 | 0.9982 | - |
| UniXcoder + GraphCodeBERT | 检索 + 重排序 | 0.9816 | 0.9979 | 0.9989 | 0.9992 | 0.9984 | - |
| Hybrid + Hard Negatives | 检索 + 重排序 | 0.9794 | 0.9978 | 0.9983 | 0.9986 | 0.9981 | - |

详细结果见：

```text
outputs/results/final_results.md
outputs/results/error_analysis.md
outputs/results/*_results.json
```

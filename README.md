# 语义代码复用检测

本仓库是 POJ-104 数据集上的语义代码复用检测实验代码。

项目包含 TF-IDF、CodeBERT、GraphCodeBERT、UniXcoder、混合重排序和 hard negative 实验。

## 我们的工作

本项目的主要工作包括：

- 将 POJ-104 整理为语义代码复用检测任务，支持检索和二分类两种实验设置。
- 实现 TF-IDF 词法检索基线、CodeBERT/GraphCodeBERT 代码对分类模型和 UniXcoder 双塔检索模型。
- 优化 UniXcoder 的 batch 内对比学习目标，避免同一 problem 的样本被错误当作负例。
- 设计 UniXcoder 召回 + GraphCodeBERT 重排序的混合方法。
- 构造 hard negative 样本，并分析其对分类和检索结果的影响。
- 完成统一的训练、评测、结果汇总和误差分析流程。

POJ-104 数据集和 CodeBERT、GraphCodeBERT、UniXcoder 等预训练模型来自公开资源，本仓库的贡献是围绕语义代码复用检测任务进行实验设计、方法实现和结果分析。

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
bash scripts/train_unixcoder_label_aware.sh
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

| 方法 | 角色 | 任务 | 主指标 | 说明 |
|---|---|---|---|---|
| TF-IDF | 对比方法 | 检索 | MAP@R 0.2169 | 词法基线 |
| CodeBERT | 对比方法 | 分类 | F1 0.9117 | 代码对分类 |
| GraphCodeBERT | 对比方法 | 分类 | F1 0.9170 | 代码对分类 |
| UniXcoder | 对比方法 | 检索 | MAP@R 0.9098 | 本地复现 baseline |
| Label-aware UniXcoder | 本文方法 | 检索 | MAP@R 0.9117 | 本项目主方法 |
| UniXcoder + GraphCodeBERT | 扩展实验 | 检索 + 重排序 | MAP@R 0.9053 | 两阶段重排序 |
| Hybrid + Hard Negatives | 消融实验 | 检索 + 重排序 | MAP@R 0.8884 | 困难负例消融 |

其中 `UniXcoder + Label-aware Loss` 是在 UniXcoder baseline 上做的训练目标优化：batch 内同一 problem 的样本不再作为负例，而是共同作为正例，从而减少假负例带来的错误惩罚。`UniXcoder + GraphCodeBERT` 是本文专门设计的混合重排序方法。`Hybrid + Hard Negatives` 用于观察 hard negative 训练对结果的影响，属于消融实验。

MAP@R 按 UniXcoder/CodeXGLUE 官方公式计算，即未进入 top-R 的相关样本按 0 计入。当前重排序方法提升了 Recall@1 和 MRR，但 MAP@R 低于 UniXcoder，说明现有融合权重更偏向提升首个正确结果，对 top-R 内整体相关样本排序仍需进一步优化。

## POJ-104 主实验结论

本课程作业的主结论聚焦 POJ-104 语义代码检索任务。相比 UniXcoder 原文报告的 MAP@R 0.9052，本项目的 Label-aware UniXcoder 达到 0.9117；相比本地复现的 UniXcoder baseline 0.9098，也有小幅提升。

| 方法 | MAP@R | 相对原文 | 相对本地 baseline | 说明 |
|---|---:|---:|---:|---|
| UniXcoder 原文 | 0.9052 | - | - | 论文报告结果 |
| UniXcoder baseline | 0.9098 | +0.0046 | - | 本地复现 |
| Label-aware UniXcoder | 0.9117 | +0.0065 | +0.0019 | 本项目方法 |

Label-aware UniXcoder 的核心改动是：在 batch 内对比学习时，同一 problem 的样本不再被当作负例，而是共同作为正例，从而减少假负例带来的错误惩罚。

## 本地展示前端

仓库提供了一个轻量级本地展示前端，可用于小组汇报时展示实验结果、指标对比、困难负例消融和错误分析。前端主体只依赖 Python 标准库，展示已有结果时不需要 GPU，也不需要本地模型权重。

```bash
python frontend/app.py --port 8501
```

启动后在浏览器打开：

```text
http://127.0.0.1:8501
```

也可以直接打开静态预览文件：

```text
frontend/preview.html
```

如果需要使用代码对推理演示，请将对应模型文件放在 `outputs/checkpoints/` 下。例如 GraphCodeBERT 分类模型应放在：

```text
outputs/checkpoints/graphcodebert_cls/
```

其中应包含 `config.json` 和 `model.safetensors` 等文件。

详细结果见：

```text
outputs/results/final_results.md
outputs/results/error_analysis.md
outputs/results/*_results.json
```

# 语义代码复用检测

本仓库是 POJ-104 数据集上的语义代码复用检测实验代码。

项目包含 TF-IDF、UniXcoder baseline、Label-aware UniXcoder 和 UniXcoder + SupCon CE 优化方法。

## 我们的工作

本项目的主要工作包括：

- 将 POJ-104 整理为语义代码复用检测任务，支持标准检索评测。
- 实现 TF-IDF 词法检索基线和 UniXcoder 双塔检索模型。
- 优化 UniXcoder 的 batch 内对比学习目标，避免同一 problem 的样本被错误当作负例。
- 在 UniXcoder 上实现监督对比学习 + CE 辅助约束，并使用 P-K balanced batch 采样增强同类正样本和跨类负样本。
- 完成统一的训练、评测、结果汇总和误差分析流程。

POJ-104 数据集和 UniXcoder 预训练模型来自公开资源，本仓库的贡献是围绕语义代码复用检测任务进行实验设计、方法实现和结果分析。

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
```

处理后应包含：

```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl
data/processed/train_pairs.jsonl
data/processed/validation_pairs.jsonl
data/processed/test_pairs.jsonl
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
bash scripts/train_unixcoder.sh
bash scripts/train_unixcoder_label_aware.sh
bash scripts/train_unixcoder_supcon_ce_k2.sh
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
| UniXcoder | 对比方法 | 检索 | MAP@R 0.9098 | 本地复现 baseline |
| Label-aware UniXcoder | 本文方法 | 检索 | MAP@R 0.9117 | 假负例修正 |
| UniXcoder + SupCon CE (k=2) | 本文方法 | 检索 | MAP@R 0.9254 | 最终主方法 |

其中 `UniXcoder + SupCon CE (k=2)` 是本项目最终主方法。它直接在 UniXcoder 上做训练目标和采样策略优化：每个 batch 中每个 `problem_id` 采样 2 个代码样本，同一 `problem_id` 的样本作为正例，不同 `problem_id` 的样本作为负例，并加入轻量 CE 辅助约束。`UniXcoder + Label-aware Loss` 是较早的假负例修正版本。

MAP@R 按 UniXcoder/CodeXGLUE 官方公式计算，即未进入 top-R 的相关样本按 0 计入。

## POJ-104 主实验结论

本课程作业的主结论聚焦 POJ-104 语义代码检索任务。相比 UniXcoder 原文报告的 MAP@R 0.9052，本项目最终方法 `UniXcoder + SupCon CE (k=2)` 达到 0.9254；相比本地复现的 UniXcoder baseline 0.9098，提升 0.0156。

| 方法 | MAP@R | 相对原文 | 相对本地 baseline | 说明 |
|---|---:|---:|---:|---|
| UniXcoder 原文 | 0.9052 | - | - | 论文报告结果 |
| UniXcoder baseline | 0.9098 | +0.0046 | - | 本地复现 |
| Label-aware UniXcoder | 0.9117 | +0.0065 | +0.0019 | 假负例修正 |
| UniXcoder + SupCon CE (k=2) | 0.9254 | +0.0202 | +0.0156 | 本项目最终方法 |

最终方法的核心改动是：在 batch 内采用 P-K balanced sampling，每个题目采样 2 个代码样本；训练目标使用监督对比学习，将同一 `problem_id` 的样本共同作为正例；同时加入 CE 辅助约束，让表示空间更明确地区分不同题目类别。该方法只改变训练过程，评测仍使用标准 MAP@R。

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

详细结果见：

```text
outputs/results/final_results.md
outputs/results/error_analysis.md
outputs/results/*_results.json
```

# goal.md

## Project Goal

Build a complete research-style system for **semantic code reuse detection**, also known as **semantic code clone detection** or **Type-4 code clone detection**.

The project should implement two related tasks:

1. **Code semantic retrieval**: given a query program, retrieve other programs that solve the same problem.
2. **Code-pair binary classification**: given two code snippets, predict whether they are semantically equivalent or reused.

The main dataset is **POJ-104** from CodeXGLUE. The main method is:

> **UniXcoder Dual Encoder for retrieval + GraphCodeBERT Cross-Encoder for reranking/classification**

The project must include the following experiments:

1. TF-IDF baseline.
2. CodeBERT binary classification.
3. GraphCodeBERT binary classification.
4. UniXcoder semantic retrieval.
5. UniXcoder retrieval + GraphCodeBERT reranking.
6. Ablation experiment with hard negatives.

The final deliverable should be a reproducible codebase with training scripts, evaluation scripts, metrics, logs, and a clear README.

------

## Research Task Definition

### Task 1: Semantic Code Retrieval

Input:

```text
query_code
candidate_code_pool
```

Output:

```text
Top-K semantically equivalent code snippets
```

In POJ-104, two programs are treated as semantically equivalent if they solve the same programming problem.

Evaluation metrics:

```text
MAP@R
Recall@1
Recall@5
Recall@10
MRR
```

Primary metric:

```text
MAP@R
```

------

### Task 2: Code-Pair Binary Classification

Input:

```text
code1
code2
```

Output:

```text
label = 1 if code1 and code2 are semantically equivalent
label = 0 otherwise
```

Evaluation metrics:

```text
Accuracy
Precision
Recall
F1
AUC
```

Primary metric:

```text
F1
```

------

## Required System Design

The final system must contain two stages.

### Stage 1: Candidate Retrieval

Implement fast retrieval methods:

1. TF-IDF baseline.
2. UniXcoder dual-encoder retrieval.

The dual encoder should independently encode two code snippets:

```text
code -> encoder -> vector
```

Similarity should be computed by cosine similarity or dot product.

The dual encoder should support contrastive training with in-batch negatives.

Recommended loss:

```text
InfoNCE / MultipleNegativesRankingLoss
```

------

### Stage 2: Cross-Encoder Reranking

Implement a GraphCodeBERT cross-encoder:

```text
[CLS] code1 [SEP] code2 [SEP] -> GraphCodeBERT -> clone / non-clone
```

The cross-encoder should be used in two ways:

1. As a standalone binary classifier.
2. As a reranker for the top-k candidates returned by the UniXcoder dual encoder.

Final reranking score:

```text
final_score = alpha * retrieval_score + beta * cross_encoder_score
```

Default:

```text
alpha = 0.4
beta = 0.6
```

Allow these values to be configurable.

------

## Dataset Requirements

Use POJ-104 as the main dataset.

Expected data logic:

```text
same problem_id  -> positive pair
different problem_id -> negative pair
```

The codebase should support the following processed data formats.

### Retrieval Format

```json
{
  "id": "sample_000001",
  "code": "int main() { ... }",
  "problem_id": "problem_001",
  "language": "cpp"
}
```

### Pair Classification Format

```json
{
  "id": "pair_000001",
  "code1": "int main() { ... }",
  "code2": "int main() { ... }",
  "problem_id1": "problem_001",
  "problem_id2": "problem_001",
  "label": 1,
  "pair_type": "positive"
}
```

------

## Hard Negative Construction

Hard negatives are required for the ablation experiment.

Construct at least three types of negative pairs:

### 1. Random Negatives

```text
problem_id1 != problem_id2
```

Randomly sample pairs from different problems.

### 2. Lexical Hard Negatives

Use TF-IDF similarity to find code snippets from different problems that look lexically similar.

Condition:

```text
problem_id1 != problem_id2
TF-IDF similarity is high
```

### 3. Length / Structure Hard Negatives

Pair code snippets from different problems with similar code length, token count, or structural patterns.

Condition:

```text
problem_id1 != problem_id2
abs(token_count1 - token_count2) is small
```

The hard-negative ablation should compare:

```text
without hard negatives
vs
with hard negatives
```

------

## Required Repository Structure

Create the following project structure:

```text
semantic-code-reuse/
├── README.md
├── goal.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── tfidf.yaml
│   ├── codebert_cls.yaml
│   ├── graphcodebert_cls.yaml
│   ├── unixcoder_retrieval.yaml
│   └── hybrid_rerank.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── scripts/
│       ├── preprocess_poj104.py
│       ├── build_pairs.py
│       └── build_hard_negatives.py
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── collators.py
│   │   └── sampling.py
│   ├── models/
│   │   ├── tfidf_baseline.py
│   │   ├── dual_encoder.py
│   │   ├── cross_encoder.py
│   │   └── hybrid_detector.py
│   ├── train/
│   │   ├── train_dual_encoder.py
│   │   └── train_cross_encoder.py
│   ├── eval/
│   │   ├── eval_retrieval.py
│   │   ├── eval_classification.py
│   │   └── eval_rerank.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── seed.py
│   │   ├── logging.py
│   │   └── io.py
│   └── run_pipeline.py
├── scripts/
│   ├── run_tfidf.sh
│   ├── train_codebert.sh
│   ├── train_graphcodebert.sh
│   ├── train_unixcoder.sh
│   ├── run_hybrid_rerank.sh
│   └── run_all_experiments.sh
├── outputs/
│   ├── logs/
│   ├── checkpoints/
│   ├── predictions/
│   └── results/
└── tests/
    ├── test_metrics.py
    ├── test_pair_builder.py
    └── test_retrieval_eval.py
```

------

## Model Requirements

### TF-IDF Baseline

Implement TF-IDF retrieval using tokenized code.

Requirements:

```text
- Tokenize code.
- Build TF-IDF vectors.
- Compute cosine similarity.
- Evaluate MAP@R and Recall@K.
```

Output file:

```text
outputs/results/tfidf_results.json
```

------

### CodeBERT Binary Classifier

Implement a binary classifier using:

```text
microsoft/codebert-base
```

Input format:

```text
code1 [SEP] code2
```

Training objective:

```text
CrossEntropyLoss
```

Metrics:

```text
Accuracy
Precision
Recall
F1
AUC
```

Output file:

```text
outputs/results/codebert_cls_results.json
```

------

### GraphCodeBERT Binary Classifier

Implement a binary classifier using:

```text
microsoft/graphcodebert-base
```

Input format:

```text
code1 [SEP] code2
```

Training objective:

```text
CrossEntropyLoss
```

Metrics:

```text
Accuracy
Precision
Recall
F1
AUC
```

Output file:

```text
outputs/results/graphcodebert_cls_results.json
```

------

### UniXcoder Dual Encoder

Implement a dual encoder using:

```text
microsoft/unixcoder-base
```

Architecture:

```text
shared encoder for query code and candidate code
mean pooling or CLS pooling
L2 normalization
dot-product similarity
```

Training objective:

```text
InfoNCE with in-batch negatives
```

Metrics:

```text
MAP@R
Recall@1
Recall@5
Recall@10
MRR
```

Output file:

```text
outputs/results/unixcoder_retrieval_results.json
```

------

### Hybrid Retrieval + Reranking

Implement a two-stage pipeline:

1. Use UniXcoder dual encoder to retrieve top-k candidates.
2. Use GraphCodeBERT cross-encoder to rerank the top-k candidates.

Default top-k:

```text
100
```

Final metrics:

```text
MAP@R
Recall@1
Recall@5
Recall@10
MRR
```

Output file:

```text
outputs/results/hybrid_rerank_results.json
```

------

## Training Configuration

Assume access to one NVIDIA A100 64GB GPU.

Default training settings:

```yaml
seed: 42
precision: bf16
max_length: 512
num_epochs: 3
learning_rate: 2e-5
weight_decay: 0.01
warmup_ratio: 0.1
gradient_accumulation_steps: 1
```

Recommended batch sizes:

```yaml
cross_encoder_batch_size: 32
dual_encoder_batch_size: 128
eval_batch_size: 128
```

The batch sizes should be configurable and should not be hard-coded.

------

## Execution Environment Requirements

All experiments must be executed on the remote machine named `h100`.

The project directory must be created under:

```bash
~/BIG
```

The final repository path should be:

```bash
~/BIG/semantic-code-reuse
```

Do not create the project under the local machine or under any other directory on the remote machine.

------

## SSH Requirement

Before running any experiment, connect to the remote server:

```bash
ssh h100
```

After connecting to `h100`, create the working directory:

```bash
mkdir -p ~/BIG
cd ~/BIG
```

Clone or create the project repository under:

```bash
~/BIG/semantic-code-reuse
```

Example:

```bash
cd ~/BIG
git clone <REPOSITORY_URL> semantic-code-reuse
cd semantic-code-reuse
```

If the repository is created from scratch, initialize it as:

```bash
cd ~/BIG
mkdir -p semantic-code-reuse
cd semantic-code-reuse
git init
```

------

## GPU Requirement

All neural model training and evaluation experiments must use only physical GPU `cuda3` on the `h100` server.

Use the following environment variable before running any Python training or evaluation command:

```bash
export CUDA_VISIBLE_DEVICES=3
```

After setting this variable, the selected physical GPU `cuda3` will appear inside PyTorch as:

```text
cuda:0
```

Therefore, inside the code, use:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

or:

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

Do not hard-code `cuda:3` inside Python scripts, because after `CUDA_VISIBLE_DEVICES=3` is set, the visible GPU is remapped to `cuda:0`.

------

## Required GPU Check

Before running experiments, verify that the correct GPU is visible:

```bash
export CUDA_VISIBLE_DEVICES=3
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(0))
PY
```

Expected result:

```text
CUDA available: True
Visible GPU count: 1
Current device: 0
GPU name: <H100 GPU name>
```

If `Visible GPU count` is not `1`, stop and check the environment before running experiments.

------

## Updated Training Configuration

The default hardware setting is:

```yaml
machine: h100
project_root: ~/BIG/semantic-code-reuse
gpu: cuda3
cuda_visible_devices: "3"
visible_device_inside_pytorch: "cuda:0"
precision: bf16
```

Assume access to one NVIDIA H100 GPU through physical device `cuda3`.

Default training settings:

```yaml
seed: 42
precision: bf16
max_length: 512
num_epochs: 3
learning_rate: 2e-5
weight_decay: 0.01
warmup_ratio: 0.1
gradient_accumulation_steps: 1
```

Recommended batch sizes for H100:

```yaml
cross_encoder_batch_size: 64
dual_encoder_batch_size: 256
eval_batch_size: 256
```

The batch sizes must remain configurable. If out-of-memory occurs, reduce batch sizes in this order:

```text
cross_encoder_batch_size: 64 -> 32 -> 16
dual_encoder_batch_size: 256 -> 128 -> 64
eval_batch_size: 256 -> 128
```

------

## Required Script Behavior

Every shell script under `scripts/` must explicitly set:

```bash
export CUDA_VISIBLE_DEVICES=3
```

Every training script must log the following environment information:

```text
hostname
current working directory
CUDA_VISIBLE_DEVICES
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
```

The logs should be saved under:

```bash
outputs/logs/
```

------

## Updated Reproducibility Command

The full experiment should be reproducible from the `h100` server as follows:

```bash
ssh h100

mkdir -p ~/BIG
cd ~/BIG/semantic-code-reuse

export CUDA_VISIBLE_DEVICES=3

pip install -r requirements.txt
bash scripts/run_all_experiments.sh
```

The project must not assume execution from any directory other than:

```bash
~/BIG/semantic-code-reuse
```

All outputs must be saved under:

```bash
~/BIG/semantic-code-reuse/outputs
```

------

## Updated Minimal Acceptance Criteria

In addition to the original acceptance criteria, the project is considered complete only if the following environment requirements are satisfied:

1. The repository is located at `~/BIG/semantic-code-reuse` on the `h100` server.
2. All experiments are launched after `ssh h100`.
3. All neural experiments use `CUDA_VISIBLE_DEVICES=3`.
4. The training logs confirm that only one GPU is visible to PyTorch.
5. The visible PyTorch device is `cuda:0`, corresponding to physical GPU `cuda3`.
6. No script hard-codes local paths outside `~/BIG/semantic-code-reuse`.
7. No script hard-codes `cuda:3` inside Python code.
8. All output files are saved under `~/BIG/semantic-code-reuse/outputs`.

## Metrics Implementation

Implement the following metrics from scratch or with reliable library support.

### Classification Metrics

```text
accuracy
precision
recall
f1
auc
```

### Retrieval Metrics

```text
Recall@K
MRR
MAP@R
```

MAP@R should be implemented carefully.

For each query:

```text
R = number of relevant candidates for this query
AP@R = average precision over the top R retrieved results
MAP@R = mean AP@R over all queries
```

Exclude the query itself from its candidate list if the query code appears in the retrieval pool.

------

## Experiment Plan

Run and save results for the following experiments.

### Experiment 1: TF-IDF Baseline

Command:

```bash
bash scripts/run_tfidf.sh
```

Expected output:

```text
outputs/results/tfidf_results.json
```

------

### Experiment 2: CodeBERT Binary Classification

Command:

```bash
bash scripts/train_codebert.sh
```

Expected output:

```text
outputs/results/codebert_cls_results.json
```

------

### Experiment 3: GraphCodeBERT Binary Classification

Command:

```bash
bash scripts/train_graphcodebert.sh
```

Expected output:

```text
outputs/results/graphcodebert_cls_results.json
```

------

### Experiment 4: UniXcoder Retrieval

Command:

```bash
bash scripts/train_unixcoder.sh
```

Expected output:

```text
outputs/results/unixcoder_retrieval_results.json
```

------

### Experiment 5: UniXcoder Retrieval + GraphCodeBERT Reranking

Command:

```bash
bash scripts/run_hybrid_rerank.sh
```

Expected output:

```text
outputs/results/hybrid_rerank_results.json
```

------

### Experiment 6: Hard Negative Ablation

Run two versions:

```text
without_hard_negatives
with_hard_negatives
```

Expected output:

```text
outputs/results/hard_negative_ablation.json
```

The result should clearly show whether hard negatives improve robustness.

------

## Final Results Table

Generate a final table in:

```text
outputs/results/final_results.md
```

The table should follow this format:

```markdown
| Method | Task | MAP@R | Recall@1 | Recall@5 | Recall@10 | MRR | F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TF-IDF | Retrieval | | | | | | - | Lexical baseline |
| CodeBERT | Classification | - | - | - | - | - | | Pair classifier |
| GraphCodeBERT | Classification | - | - | - | - | - | | Pair classifier |
| UniXcoder | Retrieval | | | | | | - | Dual encoder |
| UniXcoder + GraphCodeBERT | Retrieval + Rerank | | | | | | - | Hybrid method |
| Hybrid + Hard Negatives | Retrieval + Rerank | | | | | | | Ablation |
```

------

## Error Analysis

Implement an error analysis script:

```bash
python -m src.eval.error_analysis
```

The script should save:

```text
outputs/results/error_analysis.md
```

Analyze at least the following cases:

1. High lexical similarity but different semantics.
2. Low lexical similarity but same semantics.
3. Long code snippets truncated by max length.
4. Different algorithmic strategies for the same problem.
5. Cases where TF-IDF succeeds but neural models fail.
6. Cases where neural models succeed but TF-IDF fails.

------

## Reproducibility Requirements

The codebase must support:

```bash
pip install -r requirements.txt
bash scripts/run_all_experiments.sh
```

The project should use fixed random seeds.

The project should log:

```text
dataset statistics
training loss
validation metrics
test metrics
hyperparameters
checkpoint paths
runtime
GPU information
```

Use JSON or YAML for configuration.

Do not hard-code absolute local paths.

------

## README Requirements

The README must include:

1. Project introduction.
2. Task definition.
3. Dataset preparation.
4. Environment setup.
5. How to run each experiment.
6. How to reproduce final results.
7. Explanation of metrics.
8. Main results table.
9. Error analysis summary.
10. Limitations and future work.

------

## Minimal Acceptance Criteria

The project is considered complete only if all of the following are satisfied:

1. POJ-104 preprocessing works.
2. Pair construction works.
3. TF-IDF baseline runs end-to-end.
4. CodeBERT classifier trains and evaluates.
5. GraphCodeBERT classifier trains and evaluates.
6. UniXcoder dual encoder trains and evaluates.
7. Hybrid reranking pipeline runs.
8. Hard-negative ablation runs.
9. Final results are saved to `outputs/results/final_results.md`.
10. README explains how to reproduce the experiments.

------

## Preferred Implementation Details

Use:

```text
Python 3.10+
PyTorch
Transformers
Scikit-learn
NumPy
Pandas
tqdm
PyYAML
FAISS if available
```

Use Hugging Face Transformers for pretrained models.

Use FAISS for fast vector retrieval if possible. If FAISS is unavailable, fall back to matrix multiplication with PyTorch or NumPy.

------

## Important Constraints

1. Keep the implementation modular.
2. Avoid putting all code into one giant script.
3. Every experiment should be runnable independently.
4. Every metric should be saved to disk.
5. Every model checkpoint should be saved under `outputs/checkpoints/`.
6. The system should work on a small debug subset before full training.
7. Add a `--debug` flag to major scripts.
8. Add tests for metrics and pair construction.
9. Prefer clear, reliable code over over-engineered abstractions.
10. Do not silently ignore malformed data.

------

## Debug Mode

Every training and evaluation script should support:

```bash
--debug
```

In debug mode:

```text
use a small subset of data
run only a few batches
save temporary results
verify that the pipeline works end-to-end
```

Example:

```bash
python -m src.train.train_cross_encoder --config configs/graphcodebert_cls.yaml --debug
```

------

## Expected Final Deliverables

At the end of the project, the repository should contain:

```text
working source code
processed dataset files or preprocessing instructions
trained model checkpoints
experiment logs
result JSON files
final_results.md
error_analysis.md
README.md
```

The final method to emphasize in the report is:

> UniXcoder dual-encoder retrieval first retrieves semantically similar candidate programs, then GraphCodeBERT cross-encoder reranks the candidates and improves the precision of semantic code reuse detection. Hard negatives are used to reduce false positives caused by lexical or structural similarity.
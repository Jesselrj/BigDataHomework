from __future__ import annotations

import argparse
import json
import os
import re
import sys
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
REPORT_PATH = PROJECT_ROOT / "语义代码复用检测实验报告.md"
DEFAULT_CKPT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"
DEFAULT_TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
DEFAULT_MAX_CANDIDATES = 240

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METHODS = [
    {
        "id": "tfidf",
        "name": "TF-IDF",
        "task": "语义检索",
        "file": "tfidf_results.json",
        "notes": "词法相似度基线方法",
    },
    {
        "id": "unixcoder",
        "name": "UniXcoder",
        "task": "语义检索",
        "file": "unixcoder_retrieval_results.json",
        "notes": "双塔语义检索模型",
        "checkpoint": "unixcoder_retrieval",
    },
    {
        "id": "label_aware",
        "name": "Label-aware UniXcoder",
        "task": "语义检索",
        "file": "unixcoder_label_aware_results.json",
        "notes": "修正 batch 内同题样本被误作负例的问题",
        "checkpoint": "unixcoder_label_aware",
    },
    {
        "id": "supcon_ce",
        "name": "UniXcoder + SupCon CE (k=2)",
        "task": "语义检索",
        "file": "unixcoder_supcon_ce_k2_w02_results.json",
        "notes": "基于 P-K balanced batch 的监督对比学习与 CE 辅助约束",
        "checkpoint": "unixcoder_supcon_ce_k2_w02",
    },
]

INFERENCE_MODELS = [
    {"name": "UniXcoder", "checkpoint": "unixcoder_retrieval"},
    {"name": "Label-aware UniXcoder", "checkpoint": "unixcoder_label_aware"},
    {"name": "UniXcoder + SupCon CE (k=2)", "checkpoint": "unixcoder_supcon_ce_k2_w02"},
]

MODEL_CACHE: dict[str, tuple[object, object, object]] = {}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_text(path: Path, max_chars: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars] if max_chars else text


def read_report_section(start_heading: str, end_heading: str, fallback_path: Path, max_chars: int | None = None) -> str:
    report = read_text(REPORT_PATH)
    if report and start_heading in report:
        start = report.index(start_heading)
        end = report.find(end_heading, start + len(start_heading))
        section = report[start:end if end != -1 else None].strip()
        return section[:max_chars] if max_chars else section
    return read_text(fallback_path, max_chars=max_chars)


def local_checkpoint_root() -> Path:
    return Path(os.environ.get("SEMANTIC_REUSE_CKPT_ROOT", str(DEFAULT_CKPT_ROOT))).expanduser()


def checkpoint_file_state(path: Path) -> tuple[bool, bool]:
    if not path.exists() or not path.is_dir():
        return False, False
    files = {p.name for p in path.iterdir()}
    has_config = "config.json" in files
    has_weights = any(name in files for name in ("model.safetensors", "pytorch_model.bin"))
    return has_config, has_weights


def checkpoint_status() -> list[dict]:
    root = local_checkpoint_root()
    rows = []
    for method in INFERENCE_MODELS:
        checkpoint = method["checkpoint"]
        path = root / checkpoint
        files = sorted(p.name for p in path.iterdir()) if path.exists() and path.is_dir() else []
        has_config, has_weights = checkpoint_file_state(path)
        rows.append(
            {
                "name": method["name"],
                "checkpoint": checkpoint,
                "local_path": str(path),
                "exists": path.exists(),
                "looks_ready": has_config and has_weights,
                "files": files[:8],
            }
        )
    return rows


def demo_data_path() -> Path:
    return Path(os.environ.get("SEMANTIC_REUSE_TEST_FILE", str(DEFAULT_TEST_FILE))).expanduser()


@lru_cache(maxsize=1)
def demo_rows() -> tuple[dict, ...]:
    return tuple(read_jsonl(demo_data_path()))


def demo_examples(limit: int = 6) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in demo_rows():
        grouped.setdefault(str(row.get("problem_id", "")), []).append(row)

    examples = []
    for problem_id, rows in grouped.items():
        if len(rows) < 2:
            continue
        query, positive = rows[0], rows[1]
        examples.append(
            {
                "id": f"{query.get('id')}-{positive.get('id')}",
                "title": f"{problem_id} · {query.get('id')} vs candidate pool",
                "problem_id": problem_id,
                "query_id": query.get("id"),
                "positive_id": positive.get("id"),
                "query_code": query.get("code", ""),
                "positive_code": positive.get("code", ""),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def clamp_int(value: object, default: int, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(upper, parsed))


def build_candidate_rows(query_id: str | None, problem_id: str | None, max_candidates: int) -> list[dict]:
    positives = []
    negatives = []
    for row in demo_rows():
        if query_id and row.get("id") == query_id:
            continue
        if problem_id and row.get("problem_id") == problem_id:
            positives.append(row)
        else:
            negatives.append(row)
    pool = positives[: min(len(positives), max_candidates // 2)]
    pool.extend(negatives[: max_candidates - len(pool)])
    return pool[:max_candidates]


def compact_code(code: str, limit: int = 1400) -> str:
    text = str(code or "").strip()
    return text[:limit] + ("\n..." if len(text) > limit else "")


def code_tokens(code: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|&&|\|\||[{}()[\];,+\-*/%=<>]", str(code or "")))


def token_jaccard(left: str, right: str) -> float:
    a = code_tokens(left)
    b = code_tokens(right)
    return len(a & b) / max(len(a | b), 1)


@lru_cache(maxsize=1)
def demo_rows_by_id() -> dict[str, dict]:
    return {str(row.get("id")): row for row in demo_rows()}


def markdown_section(markdown: str, title: str) -> list[str]:
    marker = f"## {title}"
    start = markdown.find(marker)
    if start == -1:
        return []
    next_start = markdown.find("\n## ", start + len(marker))
    section = markdown[start: next_start if next_start != -1 else None]
    return [line.strip() for line in section.splitlines() if line.strip().startswith("- ")]


def parse_backtick_values(line: str) -> list[str]:
    return re.findall(r"`([^`]+)`", line)


def parse_markdown_pair(line: str) -> tuple[str, str]:
    pieces = re.findall(r": `([^`]+)` / `([^`]+)`", line)
    if pieces:
        return pieces[0]
    pieces = re.findall(r"\. `([^`]+)` / `([^`]+)`", line)
    if pieces:
        return pieces[0]
    return "", ""


def make_pair_case(title: str, subtitle: str, left_label: str, right_label: str, left_code: str, right_code: str, badge: str) -> dict:
    return {
        "title": title,
        "subtitle": subtitle,
        "badge": badge,
        "left_label": left_label,
        "right_label": right_label,
        "left_code": compact_code(left_code),
        "right_code": compact_code(right_code),
    }


def error_analysis_detail_markdown() -> str:
    return """## 与本任务的关系

本作业的任务不是判断固定两段代码是否相同，而是模拟代码查重中的候选召回：给定一段待查代码，在 POJ-104 候选库中检索解决同一问题的代码。评价指标 MAP@R 关注的是相关代码能否排在前 R 个结果中，因此错误分析也围绕“该召回的没召回”和“不该靠前的排太前”展开。

## 分类说明

- Label-aware 修正：普通 batch 内训练会把同一 problem_id 的其他代码当作负例推远。Label-aware loss 屏蔽这些假负例后，部分原本 top-1 误召回的 query 能回到同题代码。
- SupCon CE 修正：最终方法显式把同题样本拉近、把不同题类别边界拉开，因此能修正普通 UniXcoder 对模板相似不同题的误召回。
- 排名提升：有些 query 在 baseline 下虽然同题代码存在，但第一个同题结果排得很靠后。SupCon CE 把正例提前到 top-1，直接提升 MAP@R。
- 长度限制：UniXcoder 输入长度为 512，超长代码会被截断。若核心判断或输出逻辑在后半段，向量表示会缺失关键信息，进而影响检索排序。
- 二分类漏判：二分类模型在部分正例代码对上给出低分，说明“把两个代码拼成一对判断”并不总是稳定。我们的主方法选择检索式双塔模型，是因为它更贴合 POJ-104 的候选召回设定。

## 我们方法的对应改进

UniXcoder baseline 已经能显著优于词法基线，但普通 batch 内对比学习会把同一 problem_id 的其他代码误当负例。Label-aware loss 首先避免这类假负例继续破坏表示空间；最终方法 UniXcoder + SupCon CE 进一步使用 P-K balanced batch，让每个 batch 中同一题目至少有多个样本，并把同 problem_id 的代码作为正例拉近。这直接针对“同题不同写法”的漏召回问题。

CE 辅助约束则帮助模型把不同 problem_id 的表示边界拉开，降低“模板相似但语义不同”的误召回风险。最终 MAP@R 从 UniXcoder baseline 的 0.9098 提升到 0.9254，说明这些训练目标改动确实改善了检索式查重的排序质量。

## 汇报时的结论

这些样例不只是说明 baseline 会错，更重要的是说明改进方法修正了哪些错：Label-aware 处理 batch 假负例，SupCon 拉近同题实现，CE 拉开不同题边界。本项目的价值在于把代码查重转化为语义检索问题，并让“同题不同写法”的代码在向量空间更接近，让“不同题但模板相似”的代码更容易分开。
"""


def false_negative_cases(limit: int = 3) -> list[dict]:
    path = PROJECT_ROOT / "outputs" / "predictions" / "graphcodebert_cls_predictions.jsonl"
    cases: list[tuple[float, dict]] = []
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("label") == 1 and row.get("prediction") == 0:
                cases.append((token_jaccard(row.get("code1", ""), row.get("code2", "")), row))
    cases.sort(key=lambda item: item[0], reverse=True)
    out = []
    for overlap, row in cases[:limit]:
        out.append(
            make_pair_case(
                f"{row.get('id')} · {row.get('problem_id1')}",
                f"GraphCodeBERT false negative，score {float(row.get('score', 0.0)):.4f}，token Jaccard {overlap:.4f}",
                "代码 A",
                "代码 B",
                row.get("code1", ""),
                row.get("code2", ""),
                "漏判",
            )
        )
    return out


def retrieval_repair_cases(specs: list[dict]) -> list[dict]:
    rows_by_id = demo_rows_by_id()
    cases = []
    for spec in specs:
        query = rows_by_id.get(spec["query"], {})
        wrong = rows_by_id.get(spec["wrong"], {})
        fixed = rows_by_id.get(spec["fixed"], {})
        wrong_problem = wrong.get("problem_id", spec.get("wrong_problem", "不同题"))
        fixed_problem = fixed.get("problem_id", spec.get("problem", "同题"))
        if spec.get("mode") == "positive":
            cases.append(
                make_pair_case(
                    f"{spec['query']} · {query.get('problem_id', spec.get('problem', ''))}",
                    f"{spec['method']} 将第一个同题结果从 rank {spec['before']} 提前到 rank {spec['after']}；baseline top-1 是 {spec['wrong']}（{wrong_problem}）",
                    f"Query {spec['query']}",
                    f"改进后同题候选 {spec['fixed']}（{fixed_problem}）",
                    query.get("code", ""),
                    fixed.get("code", ""),
                    "排序提升",
                )
            )
        else:
            cases.append(
                make_pair_case(
                    f"{spec['query']} · {query.get('problem_id', spec.get('problem', ''))}",
                    f"{spec['method']} 修正 top-1：baseline 误召回 {spec['wrong']}（{wrong_problem}），改进后命中 {spec['fixed']}（{fixed_problem}）；同题 rank {spec['before']} → {spec['after']}",
                    f"Query {spec['query']}",
                    f"baseline 错误候选 {spec['wrong']}",
                    query.get("code", ""),
                    wrong.get("code", ""),
                    "已修正",
                )
            )
    return cases


def build_error_case_groups() -> list[dict]:
    markdown = read_text(RESULTS_DIR / "error_analysis.md")
    rows_by_id = demo_rows_by_id()
    groups: list[dict] = []

    groups.append(
        {
            "kind": "label_fix",
            "kind_label": "Label-aware 修正",
            "title": "屏蔽 batch 假负例后的修正样例",
            "summary": "这些 query 在普通 UniXcoder 下 top-1 召回到不同题；Label-aware loss 避免同题样本被当负例后，top-1 回到同题代码。",
            "cases": retrieval_repair_cases(
                [
                    {"query": "test_1395", "wrong": "test_167", "fixed": "test_1208", "method": "Label-aware", "before": 7, "after": 1},
                    {"query": "test_6245", "wrong": "test_8124", "fixed": "test_6138", "method": "Label-aware", "before": 6, "after": 1},
                    {"query": "test_2688", "wrong": "test_10361", "fixed": "test_2628", "method": "Label-aware", "before": 5, "after": 1},
                ]
            ),
        }
    )

    groups.append(
        {
            "kind": "supcon_fix",
            "kind_label": "SupCon CE 修正",
            "title": "监督对比学习后的 top-1 修正样例",
            "summary": "这些错误来自普通 UniXcoder 的语义空间混淆。SupCon CE 把同题代码拉近，并通过 CE 辅助约束拉开不同题边界。",
            "cases": retrieval_repair_cases(
                [
                    {"query": "test_7943", "wrong": "test_5720", "fixed": "test_7600", "method": "SupCon CE", "before": 1083, "after": 1},
                    {"query": "test_11110", "wrong": "test_3656", "fixed": "test_11136", "method": "SupCon CE", "before": 66, "after": 1},
                    {"query": "test_9821", "wrong": "test_11034", "fixed": "test_9805", "method": "SupCon CE", "before": 19, "after": 1},
                ]
            ),
        }
    )

    groups.append(
        {
            "kind": "rank_fix",
            "kind_label": "正例提前",
            "title": "同题候选被提前到 top-1",
            "summary": "这类样例直接对应 MAP@R 的提升：第一个同题结果原本排在较后位置，改进模型把它提前到首位。",
            "cases": retrieval_repair_cases(
                [
                    {"query": "test_7600", "wrong": "test_11928", "fixed": "test_7943", "method": "SupCon CE", "before": 19, "after": 1, "mode": "positive"},
                    {"query": "test_11511", "wrong": "test_4169", "fixed": "test_11729", "method": "SupCon CE", "before": 11, "after": 1, "mode": "positive"},
                    {"query": "test_8932", "wrong": "test_3630", "fixed": "test_8626", "method": "SupCon CE", "before": 8, "after": 1, "mode": "positive"},
                ]
            ),
        }
    )

    long_cases = []
    for line in markdown_section(markdown, "Long code snippets truncated by max length")[:3]:
        values = parse_backtick_values(line)
        sample_id = values[0] if values else ""
        problem = values[1] if len(values) > 1 else ""
        tokens = values[2] if len(values) > 2 else "-"
        row = rows_by_id.get(sample_id, {})
        snippet = parse_backtick_values(line)[-1] if parse_backtick_values(line) else ""
        long_cases.append(
            {
                "title": f"{sample_id} · {problem}",
                "subtitle": f"{tokens} tokens，超过 max_length 512，后半段逻辑可能被截断",
                "badge": "截断风险",
                "single_label": sample_id,
                "single_code": compact_code(row.get("code") or snippet),
            }
        )
    groups.append(
        {
            "kind": "truncation",
            "kind_label": "长度限制",
            "title": "长代码截断风险",
            "summary": "模型最大长度为 512，超长代码的关键逻辑如果出现在后半部分，会影响向量表示。",
            "cases": long_cases,
        }
    )

    groups.append(
        {
            "kind": "model_miss",
            "kind_label": "模型漏判",
            "title": "神经模型漏判样例",
            "summary": "以下正例代码对被 GraphCodeBERT 判为不相似，展示了二分类模型在复杂实现上的漏判风险。",
            "cases": false_negative_cases(),
        }
    )

    return groups


def build_summary() -> dict:
    methods = []
    for method in METHODS:
        metrics = read_json(RESULTS_DIR / method["file"])
        methods.append({**method, "metrics": metrics})

    best_retrieval = max(
        (m for m in methods if isinstance(m["metrics"].get("map@r"), (int, float))),
        key=lambda item: item["metrics"]["map@r"],
        default=None,
    )
    best_classifier = max(
        (m for m in methods if isinstance(m["metrics"].get("f1"), (int, float))),
        key=lambda item: item["metrics"]["f1"],
        default=None,
    )
    error_analysis = error_analysis_detail_markdown()
    final_table = read_text(RESULTS_DIR / "final_results.md")
    return {
        "project_root": str(PROJECT_ROOT),
        "results_dir": str(RESULTS_DIR),
        "checkpoint_root": str(local_checkpoint_root()),
        "methods": methods,
        "best_retrieval": best_retrieval,
        "best_classifier": best_classifier,
        "ablation": {},
        "error_analysis": error_analysis,
        "error_case_groups": build_error_case_groups(),
        "final_table": final_table,
        "checkpoints": checkpoint_status(),
        "inference_models": INFERENCE_MODELS,
        "examples": demo_examples(),
    }


def static_preview_html() -> str:
    data = json.dumps(build_summary(), ensure_ascii=False).replace("</", "<\\/")
    return INDEX_HTML.replace('<script src="/static/app.js"></script>', f"<script>window.__SUMMARY__ = {data};</script>\n  <script>{APP_JS}</script>").replace(
        '<link rel="stylesheet" href="/static/app.css">',
        f"<style>{APP_CSS}</style>",
    )


def load_dual_encoder(checkpoint: Path):
    key = str(checkpoint.resolve())
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    if not checkpoint.exists():
        raise FileNotFoundError(f"未找到模型目录：{checkpoint}")
    has_config, has_weights = checkpoint_file_state(checkpoint)
    if not has_config or not has_weights:
        raise FileNotFoundError(
            f"模型目录不完整：{checkpoint}。需要至少包含 config.json 和 model.safetensors 或 pytorch_model.bin。"
        )

    try:
        import torch
        from src.models.dual_encoder import DualEncoder, load_code_tokenizer
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("当前环境缺少 torch 或 transformers，请先安装 requirements.txt 后再进行本地推理。") from exc

    requested_device = os.environ.get("SEMANTIC_REUSE_DEVICE", "auto")
    if requested_device != "auto":
        device = torch.device(requested_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = load_code_tokenizer(str(checkpoint))
    model = DualEncoder.from_checkpoint(str(checkpoint)).to(device)
    model.eval()
    MODEL_CACHE[key] = (tokenizer, model, device)
    return MODEL_CACHE[key]


def find_example(example_id: str) -> dict | None:
    for example in demo_examples():
        if example["id"] == example_id:
            return example
    return None


def infer_retrieval(payload: dict) -> dict:
    example_id = str(payload.get("example_id", ""))
    example = find_example(example_id) if example_id else None
    code = str(payload.get("code", "") or (example or {}).get("query_code", "")).strip()
    if not code:
        return {"ok": False, "error": "请先输入或选择一段查询代码。"}

    model_id = str(payload.get("model", ""))
    allowed = {item["checkpoint"] for item in INFERENCE_MODELS}
    if model_id not in allowed:
        return {"ok": False, "error": f"当前不支持该模型：{model_id}"}

    import torch
    from src.models.dual_encoder import encode_rows

    max_candidates = clamp_int(payload.get("max_candidates"), DEFAULT_MAX_CANDIDATES, 20, 600)
    top_k = clamp_int(payload.get("top_k"), 5, 1, 12)
    query_id = str((example or {}).get("query_id") or payload.get("query_id") or "") or None
    problem_id = str((example or {}).get("problem_id") or payload.get("problem_id") or "") or None
    candidates = build_candidate_rows(query_id, problem_id, max_candidates)
    if not candidates:
        return {"ok": False, "error": f"未找到可检索的测试集样本：{demo_data_path()}"}

    checkpoint = local_checkpoint_root() / model_id
    tokenizer, model, device = load_dual_encoder(checkpoint)
    max_length = clamp_int(payload.get("max_length"), 512, 64, 1024)
    batch_size = clamp_int(payload.get("batch_size"), 64, 1, 128)

    query_vec = encode_rows(model, tokenizer, [{"code": code}], device, max_length, 1)
    candidate_vecs = encode_rows(model, tokenizer, candidates, device, max_length, batch_size)
    scores = (query_vec @ candidate_vecs.T).squeeze(0)
    ranked = torch.argsort(scores, descending=True).tolist()[:top_k]

    results = []
    for rank, idx in enumerate(ranked, start=1):
        row = candidates[idx]
        candidate_problem = str(row.get("problem_id", ""))
        results.append(
            {
                "rank": rank,
                "id": row.get("id"),
                "problem_id": candidate_problem,
                "score": float(scores[idx]),
                "is_same_problem": bool(problem_id and candidate_problem == problem_id),
                "code": str(row.get("code", ""))[:1800],
            }
        )

    return {
        "ok": True,
        "model": model_id,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "query_id": query_id,
        "problem_id": problem_id,
        "candidate_count": len(candidates),
        "top_k": top_k,
        "results": results,
    }


class AppHandler(BaseHTTPRequestHandler):
    server_version = "SemanticReuseDemo/0.1"

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.respond_headers("text/html; charset=utf-8", len(INDEX_HTML.encode("utf-8")))
            return
        if parsed.path == "/static/app.css":
            self.respond_headers("text/css; charset=utf-8", len(APP_CSS.encode("utf-8")))
            return
        if parsed.path == "/static/app.js":
            self.respond_headers("application/javascript; charset=utf-8", len(APP_JS.encode("utf-8")))
            return
        if parsed.path in {"/api/summary", "/api/health"}:
            self.respond_headers("application/json; charset=utf-8", 0)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.respond(static_preview_html(), "text/html; charset=utf-8")
            return
        if parsed.path == "/static/app.css":
            self.respond(APP_CSS, "text/css; charset=utf-8")
            return
        if parsed.path == "/static/app.js":
            self.respond(APP_JS, "application/javascript; charset=utf-8")
            return
        if parsed.path == "/api/summary":
            self.respond_json(build_summary())
            return
        if parsed.path == "/api/health":
            self.respond_json({"ok": True, "project_root": str(PROJECT_ROOT)})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/infer":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            result = infer_retrieval(payload)
            status = HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST
            self.respond_json(result, status=status)
        except FileNotFoundError as exc:
            self.respond_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "hint": f"请将对应模型文件同步到本地 {local_checkpoint_root()} 后再进行推理。",
                },
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        except Exception as exc:
            self.respond_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[frontend] " + fmt % args + "\n")

    def respond(self, body: str, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = body.encode("utf-8")
        self.respond_headers(content_type, len(data), status=status)
        self.wfile.write(data)

    def respond_headers(self, content_type: str, content_length: int, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def respond_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.respond(json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8", status)


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>语义代码复用检测系统展示</title>
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <p class="eyebrow">POJ-104 · Type-4 代码克隆检测</p>
      <h1>语义代码复用检测系统展示</h1>
      <p class="subtitle">面向代码语义检索、代码对分类、混合重排序与困难负例分析的实验系统。</p>
    </div>
    <nav class="tabs" aria-label="Main navigation">
      <button class="tab is-active" data-view="overview">总览</button>
      <button class="tab" data-view="metrics">指标</button>
      <button class="tab" data-view="demo">示例推理</button>
      <button class="tab" data-view="errors">错误分析</button>
    </nav>
    <div id="headerStatus" class="header-status">加载中</div>
  </header>

  <main>
    <section id="overview" class="view is-active">
      <section class="hero-panel">
        <div class="hero-copy">
          <span class="hero-kicker">语义代码复用检测</span>
          <h2>识别不同实现形式下的语义等价代码。</h2>
          <p>本展示页汇总 POJ-104 数据集上的语义检索、代码对分类、混合重排序与困难负例消融结果，并在本地模型文件可用时提供代码对推理演示。</p>
        </div>
        <div class="summary-grid">
          <article class="summary-block accent-green">
            <span>最佳检索</span>
            <strong id="bestRetrieval">-</strong>
            <small id="bestRetrievalMeta">MAP@R</small>
          </article>
          <article class="summary-block accent-blue">
            <span>优化提升</span>
            <strong id="bestClassifier">-</strong>
            <small id="bestClassifierMeta">vs UniXcoder</small>
          </article>
          <article class="summary-block accent-red">
            <span>运行模式</span>
            <strong id="runMode">展示</strong>
            <small id="runModeMeta">本地模型状态检测中</small>
          </article>
        </div>
      </section>
      <section class="band pipeline-band">
        <div class="section-title">
          <h2>系统流程</h2>
          <p>系统首先将 POJ-104 处理为语义检索样本，再分别评估词法基线、UniXcoder baseline 和本文优化后的 UniXcoder 方法。</p>
        </div>
        <div class="pipeline">
          <div>POJ-104</div>
          <div>语义检索样本</div>
          <div>UniXcoder · SupCon CE</div>
          <div>SupCon CE 优化</div>
          <div>指标汇总与错误分析</div>
        </div>
      </section>
      <section class="band">
        <div class="section-title">
          <h2>方法概览</h2>
          <p>页面聚焦 POJ-104 语义检索任务，对比词法基线、UniXcoder baseline、假负例修正和监督对比学习优化。</p>
        </div>
        <div id="methodCards" class="method-cards"></div>
      </section>
      <section class="band">
        <div>
          <h2>本地模型状态</h2>
          <div id="checkpointList" class="checkpoint-list"></div>
        </div>
      </section>
    </section>

    <section id="metrics" class="view">
      <div class="section-title">
        <h2>实验指标</h2>
        <p>语义检索任务主要关注 MAP@R、Recall 和 MRR。</p>
      </div>
      <div id="metricHighlights" class="metric-highlights"></div>
      <section class="band visual-section">
        <div class="section-title">
          <h2>主要结果可视化</h2>
          <p>条形图突出不同方法在核心指标上的差异，便于汇报时快速说明模型提升来源。</p>
        </div>
        <div class="viz-grid">
          <article class="chart-panel">
            <div class="chart-heading">
              <h3>语义检索 MAP@R</h3>
              <span>越高越好</span>
            </div>
            <div id="retrievalChart" class="rank-chart"></div>
          </article>
          <article class="chart-panel">
            <div class="chart-heading">
              <h3>多指标热力图</h3>
              <span>颜色越深越好</span>
            </div>
            <div id="metricMatrix" class="metric-matrix"></div>
          </article>
          <article class="chart-panel chart-panel-wide">
            <div class="chart-heading">
              <h3>相对 UniXcoder 的 MAP@R 变化</h3>
              <span>展示改进幅度</span>
            </div>
            <div id="improvementChart" class="delta-chart"></div>
          </article>
        </div>
      </section>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>方法</th>
              <th>任务</th>
              <th>MAP@R</th>
              <th>R@1</th>
              <th>R@5</th>
              <th>R@10</th>
              <th>MRR</th>
              <th>F1</th>
              <th>说明</th>
            </tr>
          </thead>
          <tbody id="metricsRows"></tbody>
        </table>
      </div>
    </section>

    <section id="demo" class="view">
      <div class="section-title">
        <h2>示例推理</h2>
        <p>选择已保存的双塔模型权重，在测试集候选池中检索与查询代码语义最接近的实现。</p>
      </div>
      <section class="demo-layout">
        <div class="control-grid">
          <label>
            模型
            <select id="modelSelect"></select>
          </label>
          <label>
            示例
            <select id="exampleSelect"></select>
          </label>
          <label>
            Top-K
            <input id="topKInput" type="number" min="1" max="12" value="5">
          </label>
          <label>
            候选数
            <input id="candidateInput" type="number" min="20" max="600" value="240">
          </label>
        </div>
        <div class="demo-workspace">
          <article class="query-panel">
            <label>
              待查代码
              <textarea id="queryCode" spellcheck="false"></textarea>
            </label>
            <div class="code-preview">
              <div class="code-heading">
                <span>高亮预览</span>
                <small id="queryMeta">-</small>
              </div>
              <pre><code id="queryPreview"></code></pre>
            </div>
          </article>
          <article class="results-panel">
            <div class="demo-actions">
              <button id="runInfer" class="primary" type="button">运行检索</button>
              <div id="inferResult" class="result-panel">请选择示例后运行。</div>
            </div>
            <div id="retrievalResults" class="retrieval-results"></div>
          </article>
        </div>
      </section>
    </section>

    <section id="errors" class="view">
      <div class="section-title">
        <h2>错误分析</h2>
        <p>本页汇总语义检索模型在词法相似、实现风格差异和长代码样本上的典型表现。</p>
      </div>
      <div id="errorCards" class="error-cards"></div>
      <section class="band visual-section">
        <div class="section-title">
          <h2>错误类型可视化</h2>
          <p>把改进后可避免的错误、残余风险和二分类漏判放在同一张图里，便于汇报时说明分析重点。</p>
        </div>
        <div id="errorDistribution" class="error-visual"></div>
      </section>
      <div id="errorCategoryTabs" class="error-category-tabs"></div>
      <div id="errorCaseGroups" class="error-case-groups"></div>
      <h2 class="detail-heading">详细说明</h2>
      <div id="errorAnalysis" class="markdown-panel"></div>
    </section>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
"""


APP_CSS = """
:root {
  color-scheme: light;
  --ink: #18212f;
  --muted: #657184;
  --soft: #8793a3;
  --line: #dce4ec;
  --line-strong: #c5d1dc;
  --bg: #f4f7f9;
  --panel: #ffffff;
  --panel-tint: #f9fbfc;
  --green: #147765;
  --blue: #285fae;
  --red: #b64e43;
  --gold: #a66f12;
  --violet: #6554b5;
  --shadow: 0 18px 48px rgb(26 38 55 / 0.10);
  --shadow-soft: 0 8px 22px rgb(26 38 55 / 0.08);
}

* { box-sizing: border-box; }
html { min-width: 320px; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background:
    linear-gradient(180deg, #eef4f3 0, rgba(238, 244, 243, 0) 360px),
    var(--bg);
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 10;
  display: grid;
  grid-template-columns: minmax(260px, 1fr) auto auto;
  align-items: center;
  gap: 18px;
  padding: 18px clamp(18px, 4vw, 48px);
  border-bottom: 1px solid rgba(197, 209, 220, 0.9);
  background: rgba(255, 255, 255, 0.88);
  backdrop-filter: blur(16px);
}

.eyebrow {
  margin: 0 0 6px;
  font-size: 12px;
  font-weight: 700;
  color: var(--green);
  letter-spacing: 0;
  text-transform: uppercase;
}

h1, h2, p { margin-top: 0; }
h1 { margin-bottom: 4px; font-size: 28px; line-height: 1.14; letter-spacing: 0; }
h2 { margin-bottom: 8px; font-size: 18px; letter-spacing: 0; }
code { font-size: 0.93em; }
.subtitle { margin: 0; color: var(--muted); line-height: 1.45; }

.tabs {
  display: flex;
  gap: 6px;
  padding: 5px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #edf2f5;
}

.tab, .primary {
  border: 0;
  font: inherit;
  cursor: pointer;
}

.tab {
  min-width: 78px;
  padding: 10px 12px;
  border-radius: 6px;
  color: var(--muted);
  background: transparent;
  transition: background 160ms ease, color 160ms ease, box-shadow 160ms ease;
}

.tab.is-active {
  color: var(--ink);
  background: #fff;
  box-shadow: 0 2px 8px rgb(30 44 58 / 0.10);
}

.header-status {
  justify-self: end;
  min-width: 88px;
  padding: 9px 12px;
  border: 1px solid var(--line);
  border-radius: 999px;
  color: var(--green);
  background: #eef8f5;
  text-align: center;
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}

main {
  width: min(1440px, 100%);
  margin: 0 auto;
  padding: 24px clamp(18px, 4vw, 48px) 44px;
}

.view { display: none; }
.view.is-active { display: block; }

.hero-panel {
  display: grid;
  grid-template-columns: 0.88fr 1.12fr;
  gap: 20px;
  align-items: stretch;
  padding: 22px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background:
    linear-gradient(135deg, rgba(20, 119, 101, 0.10), rgba(40, 95, 174, 0.08)),
    #fff;
  box-shadow: var(--shadow-soft);
}

.hero-copy {
  display: grid;
  align-content: center;
  min-height: 220px;
  padding: 8px 8px 8px 0;
}

.hero-kicker {
  width: fit-content;
  margin-bottom: 14px;
  padding: 6px 10px;
  border: 1px solid rgba(20, 119, 101, 0.24);
  border-radius: 999px;
  color: var(--green);
  background: rgba(255, 255, 255, 0.72);
  font-size: 12px;
  font-weight: 800;
}

.hero-copy h2 {
  max-width: 620px;
  margin-bottom: 12px;
  font-size: clamp(28px, 4vw, 48px);
  line-height: 1.06;
}

.hero-copy p {
  max-width: 620px;
  margin-bottom: 0;
  color: #526173;
  font-size: 16px;
  line-height: 1.7;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  align-content: stretch;
}

.summary-block, .band, .markdown-panel, .result-panel, .table-wrap {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}

.summary-block {
  position: relative;
  overflow: hidden;
  display: grid;
  align-content: space-between;
  min-height: 160px;
  padding: 18px;
  border-top: 5px solid var(--green);
  box-shadow: 0 10px 28px rgb(31 44 56 / 0.06);
}

.summary-block::after {
  content: "";
  position: absolute;
  right: -34px;
  bottom: -42px;
  width: 118px;
  height: 118px;
  border: 18px solid rgba(20, 119, 101, 0.10);
  border-radius: 50%;
}

.accent-blue::after { border-color: rgba(40, 95, 174, 0.12); }
.accent-red::after { border-color: rgba(182, 78, 67, 0.12); }

.summary-block span, .summary-block small {
  position: relative;
  z-index: 1;
  display: block;
  color: var(--muted);
}

.summary-block strong {
  position: relative;
  z-index: 1;
  display: block;
  margin: 16px 0 8px;
  max-width: 100%;
  font-size: clamp(18px, 2vw, 31px);
  line-height: 1.08;
  letter-spacing: -0.02em;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.accent-blue { border-top-color: var(--blue); }
.accent-red { border-top-color: var(--red); }
.accent-green { border-top-color: var(--green); }

.band {
  margin-top: 18px;
  padding: 22px;
  box-shadow: 0 8px 22px rgb(31 44 56 / 0.05);
}

.section-title {
  max-width: 900px;
  margin: 0 0 16px;
}

.section-title p, .body-copy {
  color: var(--muted);
  line-height: 1.65;
}

.pipeline-band {
  background:
    linear-gradient(90deg, rgba(255, 255, 255, 0.94), rgba(249, 251, 252, 0.96)),
    #fff;
}

.pipeline {
  display: grid;
  grid-template-columns: repeat(5, minmax(120px, 1fr));
  gap: 10px;
  counter-reset: step;
}

.pipeline div {
  position: relative;
  min-height: 82px;
  display: grid;
  align-content: center;
  padding: 10px;
  padding-left: 46px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel-tint);
  color: #2d3947;
  font-weight: 700;
}

.pipeline div::before {
  counter-increment: step;
  content: counter(step);
  position: absolute;
  left: 12px;
  top: 50%;
  display: grid;
  width: 24px;
  height: 24px;
  place-items: center;
  border-radius: 50%;
  color: #fff;
  background: var(--green);
  font-size: 12px;
  transform: translateY(-50%);
}

.method-cards, .metric-highlights {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.metric-highlights {
  margin-bottom: 16px;
}

.method-card, .metric-card {
  min-height: 132px;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel-tint);
}

.method-card {
  display: grid;
  gap: 10px;
}

.method-card header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.method-card h3, .metric-card h3 {
  margin: 0;
  font-size: 15px;
}

.metric-card {
  display: grid;
  gap: 9px;
  position: relative;
  overflow: hidden;
  background:
    linear-gradient(135deg, rgba(20, 119, 101, 0.08), rgba(40, 95, 174, 0.06)),
    #fff;
}

.metric-card strong {
  display: block;
  font-size: 30px;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}

.metric-card span {
  color: var(--muted);
}

.method-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.55;
}

.pill {
  width: fit-content;
  padding: 5px 8px;
  border-radius: 999px;
  color: var(--blue);
  background: #edf4ff;
  font-size: 11px;
  font-weight: 800;
}

.score-row {
  display: grid;
  grid-template-columns: 68px 1fr 56px;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 12px;
}

.bar-track {
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  background: #e7edf2;
}

.bar-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--green), var(--blue));
}

.visual-section {
  margin-top: 0;
}

.viz-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

.chart-panel-wide {
  grid-column: 1 / -1;
}

.chart-panel {
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel-tint);
}

.chart-heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 14px;
}

.chart-heading h3 {
  margin: 0;
  font-size: 15px;
}

.chart-heading span {
  color: var(--muted);
  font-size: 12px;
}

.rank-chart {
  display: grid;
  gap: 12px;
}

.rank-row {
  display: grid;
  grid-template-columns: minmax(130px, 190px) 1fr 64px;
  align-items: center;
  gap: 10px;
}

.rank-name {
  min-width: 0;
  overflow: hidden;
  color: #2f3b49;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.rank-value {
  color: var(--ink);
  font-variant-numeric: tabular-nums;
  font-weight: 800;
  text-align: right;
}

.rank-track {
  height: 14px;
  overflow: hidden;
  border: 1px solid #dbe5ed;
  border-radius: 999px;
  background: #edf2f5;
}

.rank-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--green), var(--blue));
}

.rank-row.is-baseline .rank-fill {
  background: linear-gradient(90deg, #9a6b16, #cf9a2c);
}

.rank-row.is-best .rank-name {
  color: var(--green);
}

.metric-matrix {
  overflow-x: auto;
}

.matrix-grid {
  display: grid;
  grid-template-columns: minmax(150px, 1.2fr) repeat(4, minmax(82px, 1fr));
  min-width: 560px;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  background: #fff;
}

.matrix-cell {
  min-height: 44px;
  display: grid;
  align-content: center;
  padding: 9px 10px;
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  color: #273345;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.matrix-cell:nth-child(5n) {
  border-right: 0;
}

.matrix-cell.is-head {
  background: #edf3f6;
  color: #344254;
  font-weight: 800;
}

.matrix-cell.is-method {
  font-weight: 800;
}

.matrix-cell.is-score {
  font-weight: 800;
  text-align: right;
}

.delta-chart {
  display: grid;
  gap: 12px;
}

.delta-row {
  display: grid;
  grid-template-columns: minmax(150px, 220px) 1fr 72px;
  align-items: center;
  gap: 10px;
}

.delta-label {
  min-width: 0;
  overflow: hidden;
  color: #2f3b49;
  font-size: 13px;
  font-weight: 800;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.delta-track {
  position: relative;
  height: 18px;
  overflow: hidden;
  border: 1px solid #dbe5ed;
  border-radius: 999px;
  background: linear-gradient(90deg, #f4ebe8 0 50%, #edf5f2 50% 100%);
}

.delta-track::before {
  content: "";
  position: absolute;
  left: 50%;
  top: 0;
  width: 1px;
  height: 100%;
  background: #aebbc7;
}

.delta-fill {
  position: absolute;
  top: 3px;
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--green), var(--blue));
}

.delta-fill.is-negative {
  background: linear-gradient(90deg, #cf9a2c, var(--red));
}

.delta-value {
  font-variant-numeric: tabular-nums;
  font-weight: 800;
  text-align: right;
}

.two-col {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 24px;
}

.checkpoint-list {
  display: grid;
  gap: 8px;
}

.checkpoint-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel-tint);
}

.checkpoint-item small {
  color: var(--muted);
}

.status-ok { color: var(--green); font-weight: 700; }
.status-miss { color: var(--red); font-weight: 700; }

.command {
  overflow: auto;
  padding: 14px;
  border: 1px solid #253244;
  border-radius: 8px;
  background: #111820;
  color: #edf4f8;
  line-height: 1.55;
}

.table-wrap {
  overflow-x: auto;
  background: #fff;
  box-shadow: var(--shadow-soft);
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 860px;
}

th, td {
  padding: 14px 13px;
  text-align: left;
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
}

th {
  font-size: 12px;
  color: var(--muted);
  background: #f2f6f8;
  text-transform: uppercase;
}

td.metric {
  font-variant-numeric: tabular-nums;
  font-weight: 700;
}

tbody tr:hover {
  background: #f9fbfc;
}

.delta-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(120px, 1fr));
  gap: 10px;
  margin-top: 12px;
}

.delta {
  padding: 15px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel-tint);
}

.delta span {
  display: block;
  color: var(--muted);
  font-size: 12px;
}

.delta strong {
  display: block;
  margin-top: 6px;
  font-variant-numeric: tabular-nums;
}

.ablation-viz {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
}

.ablation-card {
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
}

.ablation-card h3 {
  margin: 0 0 12px;
  font-size: 13px;
}

.paired-bars {
  display: grid;
  gap: 8px;
}

.paired-row {
  display: grid;
  grid-template-columns: 34px 1fr 58px;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 12px;
}

.paired-track {
  height: 9px;
  overflow: hidden;
  border-radius: 999px;
  background: #e8eef3;
}

.paired-fill {
  height: 100%;
  border-radius: inherit;
  background: var(--blue);
}

.paired-row.after .paired-fill {
  background: var(--red);
}

.markdown-panel {
  margin-top: 14px;
  padding: 22px;
  line-height: 1.65;
  box-shadow: var(--shadow-soft);
}

.detail-heading {
  margin: 22px 0 -2px;
}

.markdown-panel h1 { font-size: 22px; }
.markdown-panel h2 { margin-top: 24px; padding-top: 14px; border-top: 1px solid var(--line); }
.markdown-panel li { margin: 8px 0; }
.markdown-panel code {
  padding: 2px 5px;
  border-radius: 5px;
  background: #eef2f5;
}

.error-cards {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.error-card {
  min-height: 150px;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 8px 22px rgb(31 44 56 / 0.05);
}

.error-card span {
  display: inline-grid;
  width: 28px;
  height: 28px;
  margin-bottom: 12px;
  place-items: center;
  border-radius: 50%;
  color: #fff;
  background: var(--blue);
  font-size: 12px;
  font-weight: 800;
}

.error-card h3 {
  margin: 0 0 8px;
  font-size: 15px;
}

.error-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}

.error-visual {
  display: grid;
  grid-template-columns: 220px 1fr;
  align-items: center;
  gap: 20px;
}

.donut-wrap {
  position: relative;
  display: grid;
  justify-items: center;
  gap: 10px;
}

.donut {
  width: 168px;
  aspect-ratio: 1;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: conic-gradient(var(--green) 0 20%, var(--blue) 20% 40%, var(--violet) 40% 60%, var(--gold) 60% 80%, var(--red) 80% 100%);
}

.donut::before {
  content: "";
  width: 104px;
  aspect-ratio: 1;
  border-radius: 50%;
  background: #fff;
  box-shadow: inset 0 0 0 1px var(--line);
}

.donut-total {
  position: absolute;
  left: 50%;
  top: 84px;
  transform: translate(-50%, -50%);
  display: grid;
  justify-items: center;
  pointer-events: none;
  font-variant-numeric: tabular-nums;
}

.donut-total strong {
  font-size: 28px;
  line-height: 1;
}

.donut-total span {
  color: var(--muted);
  font-size: 12px;
}

.error-bars {
  display: grid;
  gap: 10px;
}

.error-bar-row {
  display: grid;
  grid-template-columns: minmax(110px, 150px) 1fr 40px;
  align-items: center;
  gap: 10px;
}

.error-bar-label {
  color: #2f3b49;
  font-size: 13px;
  font-weight: 800;
}

.error-bar-track {
  height: 12px;
  overflow: hidden;
  border-radius: 999px;
  background: #edf2f5;
}

.error-bar-fill {
  height: 100%;
  border-radius: inherit;
}

.error-bar-count {
  color: var(--ink);
  font-variant-numeric: tabular-nums;
  font-weight: 800;
  text-align: right;
}

.error-case-groups {
  display: grid;
  gap: 16px;
  margin-top: 18px;
}

.error-category-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 18px;
}

.error-filter {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 8px 12px;
  color: var(--muted);
  background: #fff;
  font: inherit;
  font-size: 13px;
  font-weight: 800;
  cursor: pointer;
}

.error-filter.is-active {
  color: #fff;
  border-color: var(--green);
  background: var(--green);
}

.error-case-group {
  display: grid;
  gap: 12px;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  box-shadow: var(--shadow-soft);
}

.error-case-group > header {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  align-items: start;
}

.error-case-group h3 {
  margin: 0;
  font-size: 17px;
}

.error-case-group p {
  margin: 0;
  color: var(--muted);
  line-height: 1.55;
}

.group-kind {
  flex: 0 0 auto;
  padding: 6px 10px;
  border: 1px solid rgba(40, 95, 174, 0.18);
  border-radius: 999px;
  color: var(--blue);
  background: #eef4fb;
  font-size: 12px;
  font-weight: 900;
}

.case-list {
  display: grid;
  gap: 12px;
}

.case-item {
  display: grid;
  gap: 10px;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel-tint);
}

.case-item > header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: start;
}

.case-item h4 {
  margin: 0 0 5px;
  font-size: 14px;
}

.case-item small {
  color: var(--muted);
  line-height: 1.45;
}

.case-badge {
  flex: 0 0 auto;
  padding: 5px 9px;
  border-radius: 999px;
  color: #fff;
  background: var(--red);
  font-size: 12px;
  font-weight: 800;
}

.case-code-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.case-code {
  min-width: 0;
}

.case-code > span {
  display: block;
  margin-bottom: 6px;
  color: #344254;
  font-size: 12px;
  font-weight: 800;
}

.case-code pre {
  max-height: 250px;
  margin: 0;
  overflow: auto;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  font-size: 12px;
  line-height: 1.45;
  tab-size: 2;
}

.demo-layout {
  display: grid;
  gap: 16px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  box-shadow: var(--shadow-soft);
}

select, textarea, input {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--ink);
  font: inherit;
  transition: border-color 160ms ease, box-shadow 160ms ease;
}

select, input {
  max-width: 340px;
  padding: 10px;
}

select:focus, textarea:focus, input:focus {
  outline: none;
  border-color: rgba(40, 95, 174, 0.62);
  box-shadow: 0 0 0 4px rgba(40, 95, 174, 0.10);
}

label {
  display: grid;
  gap: 7px;
  color: var(--muted);
  font-size: 13px;
}

.editor-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

.control-grid {
  display: grid;
  grid-template-columns: minmax(220px, 1.1fr) minmax(260px, 1.4fr) minmax(90px, 0.4fr) minmax(110px, 0.5fr);
  gap: 14px;
  align-items: end;
}

.demo-workspace {
  display: grid;
  grid-template-columns: minmax(360px, 0.9fr) minmax(420px, 1.1fr);
  gap: 16px;
  align-items: start;
}

.query-panel, .results-panel {
  display: grid;
  gap: 14px;
  min-width: 0;
}

textarea {
  min-height: 360px;
  resize: vertical;
  padding: 12px;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 13px;
  line-height: 1.5;
  color: var(--ink);
}

.primary {
  width: fit-content;
  padding: 11px 18px;
  border-radius: 8px;
  background: #166f61;
  color: #fff;
  font-weight: 800;
  box-shadow: 0 10px 24px rgb(20 119 101 / 0.22);
  transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease;
}

.primary:hover {
  background: #0f6457;
  transform: translateY(-1px);
  box-shadow: 0 14px 30px rgb(20 119 101 / 0.26);
}

.primary:disabled {
  opacity: 0.55;
  cursor: wait;
  transform: none;
}

.result-panel {
  min-height: 68px;
  padding: 14px;
  white-space: pre-wrap;
  color: #2b3746;
  background: var(--panel-tint);
}

.demo-actions {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 14px;
  align-items: stretch;
}

.retrieval-results {
  display: grid;
  gap: 12px;
}

.retrieval-item {
  display: grid;
  gap: 10px;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel-tint);
}

.retrieval-item header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
}

.retrieval-item h3 {
  margin: 0;
  font-size: 15px;
}

.code-heading {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 9px 11px;
  border: 1px solid var(--line);
  border-bottom: 0;
  border-radius: 8px 8px 0 0;
  background: #edf3f6;
  color: #344254;
  font-size: 12px;
  font-weight: 800;
}

.code-heading small {
  color: var(--muted);
  font-weight: 700;
}

.code-preview pre, .retrieval-item pre {
  max-height: 360px;
  margin: 0;
  overflow: auto;
  padding: 12px;
  border: 1px solid var(--line);
  background: #fbfcfd;
  font-size: 12px;
  line-height: 1.45;
  tab-size: 2;
}

.code-preview pre {
  border-radius: 0 0 8px 8px;
}

.retrieval-item pre {
  border-radius: 8px;
}

.code-preview code, .retrieval-item code {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  color: #273345;
}

.tok-comment { color: #7a8492; font-style: italic; }
.tok-string { color: #996414; }
.tok-number { color: #9a4d98; }
.tok-preproc { color: #286c8f; font-weight: 700; }
.tok-keyword { color: #285fae; font-weight: 800; }
.tok-type { color: #147765; font-weight: 800; }
.tok-op { color: #b64e43; }

.result-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  color: var(--muted);
  font-size: 12px;
}

.match-badge {
  flex: 0 0 auto;
  padding: 5px 9px;
  border-radius: 999px;
  color: #fff;
  background: var(--green);
  font-size: 12px;
  font-weight: 800;
}

.match-badge.is-negative {
  background: var(--soft);
}

@media (max-width: 1120px) {
  .topbar {
    grid-template-columns: 1fr;
    align-items: start;
  }
  .header-status {
    justify-self: start;
  }
  .hero-panel, .two-col, .control-grid, .demo-workspace {
    grid-template-columns: 1fr;
  }
  .method-cards, .metric-highlights {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .viz-grid, .error-cards, .ablation-viz, .error-visual {
    grid-template-columns: 1fr;
  }
  .case-code-grid {
    grid-template-columns: 1fr;
  }
  .error-case-group > header {
    display: grid;
  }
}

@media (max-width: 860px) {
  .editor-grid, .demo-actions {
    grid-template-columns: 1fr;
  }
  .tabs {
    width: 100%;
    overflow-x: auto;
  }
  .summary-grid, .pipeline, .delta-grid, .method-cards, .metric-highlights {
    grid-template-columns: 1fr;
  }
  .rank-row {
    grid-template-columns: 1fr;
    gap: 6px;
  }
  .rank-value {
    text-align: left;
  }
  .hero-copy {
    min-height: 0;
  }
}
"""


APP_JS = """
const state = { summary: null };

const fmt = (value) => {
  if (value === undefined || value === null) return "-";
  if (typeof value === "number") return value.toFixed(4);
  return String(value);
};

const metricValue = (method) => {
  const metrics = method.metrics || {};
  if (typeof metrics["map@r"] === "number") return { label: "MAP@R", value: metrics["map@r"] };
  if (typeof metrics.f1 === "number") return { label: "F1", value: metrics.f1 };
  return { label: "指标", value: null };
};

const hasLiveBackend = () => window.location.protocol === "http:" || window.location.protocol === "https:";

const backendBootHint = "请使用 `python frontend\\\\app.py --port 8501` 启动本地服务，并在浏览器打开 `http://127.0.0.1:8501`。";

const checkpointById = (modelId) => {
  if (!state.summary || !state.summary.checkpoints) return null;
  return state.summary.checkpoints.find((item) => item.checkpoint === modelId) || null;
};

const exampleById = (exampleId) => {
  if (!state.summary || !state.summary.examples) return null;
  return state.summary.examples.find((item) => item.id === exampleId) || null;
};

const inferErrorText = (response, data) => {
  if (data && (data.error || data.hint)) {
    return `${data.error || "推理失败。"}\n${data.hint || ""}`.trim();
  }
  if (!response.ok) {
    return `推理接口返回错误（HTTP ${response.status}）。`;
  }
  return "推理失败，请稍后重试。";
};

const ERROR_SUMMARY = [
  ["Label-aware 修正", "屏蔽 batch 内同题假负例后，部分普通 UniXcoder 的 top-1 误召回可以被修正。"],
  ["SupCon CE 修正", "监督对比学习把同题实现拉近，CE 辅助约束把不同题边界拉开。"],
  ["正例排名提前", "改进模型把原本排在后面的同题候选提前到 top-1，直接改善 MAP@R。"],
  ["二分类漏判", "固定代码对二分类会漏掉部分正例，检索式双塔更贴合候选召回任务。"],
  ["长度限制", "超长代码仍可能被 512 token 截断，是改进训练目标之外还需要说明的残余风险。"],
];

const ERROR_KIND_ORDER = [
  ["all", "全部"],
  ["label_fix", "Label-aware 修正"],
  ["supcon_fix", "SupCon CE 修正"],
  ["rank_fix", "正例提前"],
  ["truncation", "长度限制"],
  ["model_miss", "模型漏判"],
];

const escapeHtml = (text) => String(text)
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

const CPP_KEYWORDS = new Set([
  "alignas", "alignof", "asm", "auto", "break", "case", "catch", "class", "const", "constexpr",
  "continue", "default", "delete", "do", "else", "enum", "explicit", "extern", "for", "friend",
  "goto", "if", "inline", "namespace", "new", "operator", "private", "protected", "public",
  "return", "sizeof", "static", "struct", "switch", "template", "this", "throw", "try", "typedef",
  "typename", "using", "virtual", "while"
]);

const CPP_TYPES = new Set([
  "bool", "char", "double", "float", "int", "long", "short", "signed", "unsigned", "void",
  "wchar_t", "size_t", "string", "vector", "array", "map", "set", "queue", "stack", "cin", "cout",
  "scanf", "printf", "gets", "strlen", "sort", "max", "min"
]);

const highlightCode = (code) => {
  const pattern = new RegExp("(^\\\\s*#.*$)|//[^\\\\n]*|/[*][\\\\s\\\\S]*?[*]/|\\\"(?:\\\\\\\\.|[^\\\"\\\\\\\\])*\\\"|'(?:\\\\\\\\.|[^'\\\\\\\\])*'|\\\\b\\\\d+(?:[.]\\\\d+)?\\\\b|\\\\b[A-Za-z_]\\\\w*\\\\b|[{}()[\\\\];,.+\\\\-*\\\\/%=!<>&|?:]+|\\\\s+|.", "gm");
  return String(code || "").replace(pattern, (token, preproc) => {
    const safe = escapeHtml(token);
    if (/^\s+$/.test(token)) return safe;
    if (preproc) return `<span class="tok-preproc">${safe}</span>`;
    if (token.startsWith("//") || token.startsWith("/*")) return `<span class="tok-comment">${safe}</span>`;
    if (token.startsWith('"') || token.startsWith("'")) return `<span class="tok-string">${safe}</span>`;
    if (/^\d/.test(token)) return `<span class="tok-number">${safe}</span>`;
    if (/^[{}()[\];,.+\-*\/%=!<>&|?:]+$/.test(token)) return `<span class="tok-op">${safe}</span>`;
    if (CPP_KEYWORDS.has(token)) return `<span class="tok-keyword">${safe}</span>`;
    if (CPP_TYPES.has(token)) return `<span class="tok-type">${safe}</span>`;
    return safe;
  });
};

const renderQueryPreview = () => {
  const code = document.getElementById("queryCode").value;
  document.getElementById("queryPreview").innerHTML = highlightCode(code);
  const lineCount = code ? code.split(String.fromCharCode(10)).length : 0;
  document.getElementById("queryMeta").textContent = `${lineCount} lines · ${code.length} chars`;
};

const markdownLite = (text) => {
  const lines = String(text || "").split("\\n");
  let html = "";
  let inList = false;
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) {
      if (inList) { html += "</ul>"; inList = false; }
      continue;
    }
    if (line.startsWith("# ")) {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<h1>${escapeHtml(line.slice(2))}</h1>`;
    } else if (line.startsWith("## ")) {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<h2>${escapeHtml(line.slice(3))}</h2>`;
    } else if (line.startsWith("- ")) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${escapeHtml(line.slice(2))}</li>`;
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<p>${escapeHtml(line)}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html;
};

const setView = (view) => {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.view === view));
  document.querySelectorAll(".view").forEach((panel) => panel.classList.toggle("is-active", panel.id === view));
};

const renderRankChart = (targetId, methods, metricKey) => {
  const rows = methods
    .filter((method) => typeof (method.metrics || {})[metricKey] === "number")
    .sort((a, b) => b.metrics[metricKey] - a.metrics[metricKey]);
  const best = rows[0] ? rows[0].metrics[metricKey] : 1;
  document.getElementById(targetId).innerHTML = rows.map((method, index) => {
    const value = method.metrics[metricKey];
    const width = Math.max(3, Math.min(100, value / best * 100));
    const classes = [
      "rank-row",
      index === 0 ? "is-best" : "",
      method.id === "tfidf" ? "is-baseline" : "",
    ].filter(Boolean).join(" ");
    return `
      <div class="${classes}">
        <div class="rank-name" title="${escapeHtml(method.name)}">${escapeHtml(method.name)}</div>
        <div class="rank-track"><div class="rank-fill" style="width:${width}%"></div></div>
        <div class="rank-value">${fmt(value)}</div>
      </div>
    `;
  }).join("");
};

const renderMetricMatrix = (targetId, methods) => {
  const metrics = [
    ["map@r", "MAP@R"],
    ["recall@1", "R@1"],
    ["recall@5", "R@5"],
    ["mrr", "MRR"],
  ];
  const rows = (methods || []).filter((method) => method.metrics && typeof method.metrics["map@r"] === "number");
  const ranges = {};
  metrics.forEach(([key]) => {
    const values = rows.map((method) => method.metrics[key]).filter((value) => typeof value === "number");
    ranges[key] = {
      min: values.length ? Math.min.apply(null, values) : 0,
      max: values.length ? Math.max.apply(null, values) : 1,
    };
  });
  const head = `<div class="matrix-cell is-head">方法</div>${metrics.map(([, label]) => `<div class="matrix-cell is-head">${label}</div>`).join("")}`;
  const body = rows.map((method) => {
    const cells = metrics.map(([key]) => {
      const value = method.metrics[key];
      const range = ranges[key];
      const ratio = typeof value === "number" && range.max > range.min ? (value - range.min) / (range.max - range.min) : 0;
      const alpha = 0.10 + ratio * 0.34;
      return `<div class="matrix-cell is-score" style="background: rgba(20, 119, 101, ${alpha.toFixed(3)})">${fmt(value)}</div>`;
    }).join("");
    return `<div class="matrix-cell is-method">${escapeHtml(method.name)}</div>${cells}`;
  }).join("");
  document.getElementById(targetId).innerHTML = `<div class="matrix-grid">${head}${body}</div>`;
};

const renderImprovementChart = (targetId, methods) => {
  const baseline = (methods || []).find((method) => method.id === "unixcoder");
  const baseValue = baseline && baseline.metrics ? baseline.metrics["map@r"] : null;
  const rows = (methods || [])
    .filter((method) => method.metrics && typeof method.metrics["map@r"] === "number")
    .filter((method) => method.id !== "unixcoder");
  if (typeof baseValue !== "number" || !rows.length) {
    document.getElementById(targetId).innerHTML = "";
    return;
  }
  const maxAbs = Math.max.apply(null, rows.map((method) => Math.abs(method.metrics["map@r"] - baseValue)).concat([0.001]));
  document.getElementById(targetId).innerHTML = rows.map((method) => {
    const delta = method.metrics["map@r"] - baseValue;
    const width = Math.max(2, Math.min(50, Math.abs(delta) / maxAbs * 50));
    const left = delta >= 0 ? 50 : 50 - width;
    const cls = delta >= 0 ? "" : "is-negative";
    const sign = delta >= 0 ? "+" : "";
    return `
      <div class="delta-row">
        <div class="delta-label" title="${escapeHtml(method.name)}">${escapeHtml(method.name)}</div>
        <div class="delta-track">
          <div class="delta-fill ${cls}" style="left:${left}%; width:${width}%"></div>
        </div>
        <div class="delta-value">${sign}${fmt(delta)}</div>
      </div>
    `;
  }).join("");
};

const renderErrorDistribution = (groups) => {
  const target = document.getElementById("errorDistribution");
  const colors = ["#147765", "#285fae", "#6554b5", "#a66f12", "#b64e43"];
  const rows = (groups || [])
    .filter((group) => group.cases && group.cases.length)
    .map((group, index) => ({
      label: group.kind_label || group.title || "错误类型",
      count: group.cases.length,
      color: colors[index % colors.length],
    }));
  const total = rows.reduce((sum, row) => sum + row.count, 0);
  const maxCount = Math.max.apply(null, rows.map((row) => row.count).concat([1]));
  let cursor = 0;
  const stops = rows.map((row) => {
    const start = total ? cursor / total * 100 : 0;
    cursor += row.count;
    const end = total ? cursor / total * 100 : 0;
    return `${row.color} ${start.toFixed(2)}% ${end.toFixed(2)}%`;
  }).join(", ");
  target.innerHTML = `
    <div class="donut-wrap">
      <div class="donut" style="background: conic-gradient(${stops})"></div>
      <div class="donut-total"><strong>${total}</strong><span>个展示样例</span></div>
    </div>
    <div class="error-bars">
      ${rows.map((row) => `
        <div class="error-bar-row">
          <div class="error-bar-label">${escapeHtml(row.label)}</div>
          <div class="error-bar-track"><div class="error-bar-fill" style="width:${Math.max(4, row.count / maxCount * 100)}%; background:${row.color}"></div></div>
          <div class="error-bar-count">${row.count}</div>
        </div>
      `).join("")}
    </div>
  `;
};

const renderErrorCards = () => {
  document.getElementById("errorCards").innerHTML = ERROR_SUMMARY.map(([title, body], index) => `
    <article class="error-card">
      <span>${index + 1}</span>
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(body)}</p>
    </article>
  `).join("");
};

const renderCaseCode = (label, code) => `
  <div class="case-code">
    <span>${escapeHtml(label || "代码")}</span>
    <pre><code>${highlightCode(code || "")}</code></pre>
  </div>
`;

const renderCaseItem = (item) => {
  const title = item.title || "案例";
  const subtitle = item.subtitle || "";
  const badge = item.badge || "案例";
  const codeHtml = item.single_code
    ? `<div class="case-code-grid">${renderCaseCode(item.single_label || "代码", item.single_code)}</div>`
    : `<div class="case-code-grid">${renderCaseCode(item.left_label, item.left_code)}${renderCaseCode(item.right_label, item.right_code)}</div>`;
  return `
    <article class="case-item">
      <header>
        <div>
          <h4>${escapeHtml(title)}</h4>
          <small>${escapeHtml(subtitle)}</small>
        </div>
        <span class="case-badge">${escapeHtml(badge)}</span>
      </header>
      ${codeHtml}
    </article>
  `;
};

const renderErrorCaseGroups = (groups) => {
  const target = document.getElementById("errorCaseGroups");
  const tabs = document.getElementById("errorCategoryTabs");
  const visibleGroups = (groups || []).filter((group) => group.cases && group.cases.length);
  const counts = {};
  visibleGroups.forEach((group) => {
    counts[group.kind] = (counts[group.kind] || 0) + group.cases.length;
  });
  const total = visibleGroups.reduce((sum, group) => sum + group.cases.length, 0);
  tabs.innerHTML = ERROR_KIND_ORDER.map(([kind, label], index) => {
    const count = kind === "all" ? total : (counts[kind] || 0);
    if (!count) return "";
    return `<button class="error-filter ${index === 0 ? "is-active" : ""}" type="button" data-kind="${escapeHtml(kind)}">${escapeHtml(label)} · ${count}</button>`;
  }).join("");
  target.innerHTML = visibleGroups.map((group) => `
    <section class="error-case-group" data-kind="${escapeHtml(group.kind || "")}">
      <header>
        <div>
          <h3>${escapeHtml(group.title)}</h3>
          <p>${escapeHtml(group.summary || "")}</p>
        </div>
        <span class="group-kind">${escapeHtml(group.kind_label || "错误类型")} · ${group.cases.length}</span>
      </header>
      <div class="case-list">
        ${group.cases.map(renderCaseItem).join("")}
      </div>
    </section>
  `).join("");
  tabs.querySelectorAll(".error-filter").forEach((button) => {
    button.addEventListener("click", () => {
      const kind = button.dataset.kind;
      tabs.querySelectorAll(".error-filter").forEach((item) => item.classList.toggle("is-active", item === button));
      target.querySelectorAll(".error-case-group").forEach((group) => {
        group.style.display = kind === "all" || group.dataset.kind === kind ? "grid" : "none";
      });
    });
  });
};

const renderSummary = (summary) => {
  const bestRetrieval = summary.best_retrieval;
  const baseline = summary.methods.find((method) => method.id === "unixcoder");
  document.getElementById("bestRetrieval").textContent = bestRetrieval ? bestRetrieval.name : "-";
  document.getElementById("bestRetrievalMeta").textContent = bestRetrieval ? `MAP@R ${fmt(bestRetrieval.metrics["map@r"])}` : "MAP@R";
  const delta = bestRetrieval && baseline ? bestRetrieval.metrics["map@r"] - baseline.metrics["map@r"] : null;
  document.getElementById("bestClassifier").textContent = typeof delta === "number" ? `+${fmt(delta)}` : "-";
  document.getElementById("bestClassifierMeta").textContent = "vs UniXcoder MAP@R";

  const ready = summary.checkpoints.some((item) => item.looks_ready);
  document.getElementById("runMode").textContent = ready ? "可推理" : "展示";
  document.getElementById("runModeMeta").textContent = ready ? "已检测到本地模型文件" : "未检测到本地模型文件";
  document.getElementById("headerStatus").textContent = ready ? "推理可用" : "结果展示";

  const checkpoints = document.getElementById("checkpointList");
  checkpoints.innerHTML = summary.checkpoints.map((item) => `
    <div class="checkpoint-item">
      <div>
        <strong>${escapeHtml(item.name)}</strong><br>
        <small>${item.looks_ready ? "已检测到配置文件与权重文件" : "尚未检测到完整权重文件"}</small>
      </div>
      <span class="${item.looks_ready ? "status-ok" : "status-miss"}">${item.looks_ready ? "已就绪" : "未同步"}</span>
    </div>
  `).join("");

  document.getElementById("methodCards").innerHTML = summary.methods.map((method) => {
    const score = metricValue(method);
    const width = typeof score.value === "number" ? Math.max(0, Math.min(100, score.value * 100)) : 0;
    return `
      <article class="method-card">
        <header>
          <h3>${escapeHtml(method.name)}</h3>
          <span class="pill">${escapeHtml(method.task)}</span>
        </header>
        <p>${escapeHtml(method.notes)}</p>
        <div class="score-row">
          <span>${escapeHtml(score.label)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <strong>${fmt(score.value)}</strong>
        </div>
      </article>
    `;
  }).join("");
};

const renderMetrics = (summary) => {
  const highlights = [
    summary.best_retrieval,
    summary.methods.find((method) => method.id === "label_aware"),
    summary.methods.find((method) => method.id === "tfidf"),
  ].filter(Boolean);
  document.getElementById("metricHighlights").innerHTML = highlights.map((method) => {
    const score = metricValue(method);
    return `
      <article class="metric-card">
        <h3>${escapeHtml(method.name)}</h3>
        <p class="pill">${escapeHtml(score.label)}</p>
        <strong>${fmt(score.value)}</strong>
        <span>${escapeHtml(method.task)}</span>
      </article>
    `;
  }).join("");

  renderRankChart("retrievalChart", summary.methods, "map@r");
  renderMetricMatrix("metricMatrix", summary.methods);
  renderImprovementChart("improvementChart", summary.methods);

  document.getElementById("metricsRows").innerHTML = summary.methods.map((method) => {
    const m = method.metrics || {};
    return `<tr>
      <td>${escapeHtml(method.name)}</td>
      <td>${escapeHtml(method.task)}</td>
      <td class="metric">${fmt(m["map@r"])}</td>
      <td class="metric">${fmt(m["recall@1"])}</td>
      <td class="metric">${fmt(m["recall@5"])}</td>
      <td class="metric">${fmt(m["recall@10"])}</td>
      <td class="metric">${fmt(m.mrr)}</td>
      <td class="metric">${fmt(m.f1)}</td>
      <td>${escapeHtml(method.notes)}</td>
    </tr>`;
  }).join("");

};

const renderErrors = (summary) => {
  renderErrorCards();
  renderErrorDistribution(summary.error_case_groups);
  renderErrorCaseGroups(summary.error_case_groups);
  document.getElementById("errorAnalysis").innerHTML = markdownLite(summary.error_analysis);
};

const updateExampleCode = () => {
  const select = document.getElementById("exampleSelect");
  const example = exampleById(select.value);
  if (!example) return;
  document.getElementById("queryCode").value = example.query_code || "";
  document.getElementById("retrievalResults").innerHTML = "";
  document.getElementById("inferResult").textContent = `当前示例：${example.problem_id} · query ${example.query_id}`;
  renderQueryPreview();
};

const renderDemo = (summary) => {
  const modelSelect = document.getElementById("modelSelect");
  const exampleSelect = document.getElementById("exampleSelect");
  modelSelect.innerHTML = (summary.inference_models || []).map((item) => `
    <option value="${escapeHtml(item.checkpoint)}">${escapeHtml(item.name)}</option>
  `).join("");
  exampleSelect.innerHTML = (summary.examples || []).map((item) => `
    <option value="${escapeHtml(item.id)}">${escapeHtml(item.title)}</option>
  `).join("");
  updateExampleCode();
};

const renderRetrievalResults = (data) => {
  const target = document.getElementById("retrievalResults");
  target.innerHTML = (data.results || []).map((item) => `
    <article class="retrieval-item">
      <header>
        <div>
          <h3>#${item.rank} · ${escapeHtml(item.id)}</h3>
          <div class="result-meta">
            <span>${escapeHtml(item.problem_id)}</span>
            <span>score ${fmt(item.score)}</span>
          </div>
        </div>
        <span class="match-badge ${item.is_same_problem ? "" : "is-negative"}">${item.is_same_problem ? "同题命中" : "非同题"}</span>
      </header>
      <pre><code>${highlightCode(item.code)}</code></pre>
    </article>
  `).join("");
};

const runInfer = async () => {
  const button = document.getElementById("runInfer");
  const result = document.getElementById("inferResult");
  const modelId = document.getElementById("modelSelect").value;
  const checkpoint = checkpointById(modelId);
  if (!hasLiveBackend()) {
    result.textContent = `当前打开的是静态预览页面，无法调用本地推理接口。\n${backendBootHint}`;
    return;
  }
  if (checkpoint && !checkpoint.looks_ready) {
    result.textContent = `当前模型尚未同步到本地：${checkpoint.local_path}\n请补齐 config.json 和模型权重文件（model.safetensors 或 pytorch_model.bin）后再运行推理。`;
    return;
  }
  button.disabled = true;
  document.getElementById("retrievalResults").innerHTML = "";
  result.textContent = "模型加载和编码中，首次运行会慢一些。";
  try {
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: modelId,
        example_id: document.getElementById("exampleSelect").value,
        code: document.getElementById("queryCode").value,
        top_k: document.getElementById("topKInput").value,
        max_candidates: document.getElementById("candidateInput").value
      })
    });
    let data = null;
    try {
      data = await response.json();
    } catch (error) {
      data = null;
    }
    if (!response.ok || !data || !data.ok) {
      result.textContent = inferErrorText(response, data);
      return;
    }
    const hits = (data.results || []).filter((item) => item.is_same_problem).length;
    result.textContent = `检索完成：Top-${data.top_k} 中同题命中 ${hits} 个\\n候选数量：${data.candidate_count}\\n运行设备：${data.device}\\n模型路径：${data.checkpoint}`;
    renderRetrievalResults(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    result.textContent = `无法连接推理接口。\n${backendBootHint}\n原始错误：${message}`;
  } finally {
    button.disabled = false;
  }
};

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => setView(tab.dataset.view)));

const showBootError = (error) => {
    const message = error instanceof Error ? error.stack || error.message : String(error);
    const status = document.getElementById("headerStatus");
    if (status) status.textContent = "加载失败";
    const main = document.querySelector("main");
    if (main) {
      main.insertAdjacentHTML(
        "afterbegin",
        `<div class="markdown-panel"><h2>页面初始化失败</h2><pre>${escapeHtml(message)}</pre></div>`
      );
    }
};

const boot = (summary) => {
    state.summary = summary;
    renderSummary(summary);
    renderMetrics(summary);
    renderErrors(summary);
    renderDemo(summary);
    document.getElementById("exampleSelect").addEventListener("change", updateExampleCode);
    document.getElementById("queryCode").addEventListener("input", renderQueryPreview);
    document.getElementById("runInfer").addEventListener("click", runInfer);
};

const safeBoot = (summary) => {
  try {
    boot(summary);
  } catch (error) {
    showBootError(error);
  }
};

const fetchSummary = async () => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch("/api/summary", { signal: controller.signal, cache: "no-store" });
    if (!response.ok) throw new Error(`summary 接口返回 HTTP ${response.status}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
};

if (window.__SUMMARY__) {
  safeBoot(window.__SUMMARY__);
} else {
  fetchSummary().then(safeBoot).catch(showBootError);
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Local frontend for semantic code reuse detection results.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--export-static", help="Write a no-server HTML preview snapshot and exit.")
    args = parser.parse_args()

    if args.export_static:
        out = Path(args.export_static)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(static_preview_html(), encoding="utf-8")
        print(f"Wrote static preview to {out}")
        return

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Semantic reuse frontend running at {url}")
    print(f"Checkpoint root: {local_checkpoint_root()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()

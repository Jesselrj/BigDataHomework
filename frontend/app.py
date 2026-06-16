from __future__ import annotations

import argparse
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
REPORT_PATH = PROJECT_ROOT / "语义代码复用检测实验报告.md"
DEFAULT_CKPT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"

METHODS = [
    {
        "id": "tfidf",
        "name": "TF-IDF",
        "task": "语义检索",
        "file": "tfidf_results.json",
        "notes": "词法相似度基线方法",
    },
    {
        "id": "codebert",
        "name": "CodeBERT",
        "task": "代码对分类",
        "file": "codebert_cls_results.json",
        "notes": "预训练代码模型二分类器",
        "checkpoint": "codebert_cls",
    },
    {
        "id": "graphcodebert",
        "name": "GraphCodeBERT",
        "task": "代码对分类",
        "file": "graphcodebert_cls_results.json",
        "notes": "代码对分类器，并作为混合方法中的重排序模型",
        "checkpoint": "graphcodebert_cls",
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
        "id": "hybrid",
        "name": "UniXcoder + GraphCodeBERT",
        "task": "检索与重排序",
        "file": "hybrid_rerank_results.json",
        "notes": "先召回候选，再进行代码对级别重排序",
    },
    {
        "id": "hybrid_hard",
        "name": "Hybrid + Hard Negatives",
        "task": "困难负例消融",
        "file": "hybrid_rerank_hard_results.json",
        "notes": "加入随机、词法相似、长度结构相似负例后的消融结果",
    },
]

INFERENCE_MODELS = [
    {"name": "CodeBERT", "checkpoint": "codebert_cls"},
    {"name": "GraphCodeBERT", "checkpoint": "graphcodebert_cls"},
    {"name": "GraphCodeBERT + Hard Negatives", "checkpoint": "graphcodebert_hard_negatives"},
    {"name": "UniXcoder", "checkpoint": "unixcoder_retrieval"},
]

MODEL_CACHE: dict[str, tuple[object, object, object]] = {}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


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
    ablation = read_json(RESULTS_DIR / "hard_negative_ablation.json")
    error_analysis = read_report_section("## 7. 错误分析", "## 8. 可复现性", RESULTS_DIR / "error_analysis.md", max_chars=9000)
    final_table = read_text(RESULTS_DIR / "final_results.md")
    return {
        "project_root": str(PROJECT_ROOT),
        "results_dir": str(RESULTS_DIR),
        "checkpoint_root": str(local_checkpoint_root()),
        "methods": methods,
        "best_retrieval": best_retrieval,
        "best_classifier": best_classifier,
        "ablation": ablation,
        "error_analysis": error_analysis,
        "final_table": final_table,
        "checkpoints": checkpoint_status(),
    }


def static_preview_html() -> str:
    data = json.dumps(build_summary(), ensure_ascii=False).replace("</", "<\\/")
    return INDEX_HTML.replace('<script src="/static/app.js"></script>', f"<script>window.__SUMMARY__ = {data};</script>\n  <script>{APP_JS}</script>").replace(
        '<link rel="stylesheet" href="/static/app.css">',
        f"<style>{APP_CSS}</style>",
    )


def load_cross_encoder(checkpoint: Path):
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
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("当前环境缺少 torch 或 transformers，请先安装 requirements.txt 后再进行本地推理。") from exc

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).to(device)
    model.eval()
    MODEL_CACHE[key] = (tokenizer, model, device)
    return MODEL_CACHE[key]


def infer_pair(payload: dict) -> dict:
    code1 = str(payload.get("code1", "")).strip()
    code2 = str(payload.get("code2", "")).strip()
    if not code1 or not code2:
        return {"ok": False, "error": "请分别输入两段待比较的代码。"}

    model_id = str(payload.get("model", "graphcodebert_cls"))
    allowed = {"graphcodebert_cls", "codebert_cls", "graphcodebert_hard_negatives"}
    if model_id not in allowed:
        return {"ok": False, "error": f"当前不支持该模型：{model_id}"}

    root = local_checkpoint_root()
    checkpoint = root / model_id
    tokenizer, model, device = load_cross_encoder(checkpoint)

    import torch

    max_length = int(payload.get("max_length", 512))
    enc = tokenizer(
        [code1],
        [code2],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {key: value.to(device) for key, value in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
    score = float(probs[1])
    return {
        "ok": True,
        "model": model_id,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "semantic_equivalence_score": score,
        "prediction": int(score >= 0.5),
        "label": "语义等价" if score >= 0.5 else "语义不等价",
    }


class AppHandler(BaseHTTPRequestHandler):
    server_version = "SemanticReuseDemo/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.respond(INDEX_HTML, "text/html; charset=utf-8")
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
            result = infer_pair(payload)
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
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

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
      <button class="tab" data-view="errors">错误分析</button>
      <button class="tab" data-view="demo">推理演示</button>
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
            <span>最佳分类</span>
            <strong id="bestClassifier">-</strong>
            <small id="bestClassifierMeta">F1</small>
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
          <p>系统首先将 POJ-104 处理为检索样本和代码对，再分别评估词法基线、双塔检索模型、Cross-Encoder 分类模型与混合重排序方法。</p>
        </div>
        <div class="pipeline">
          <div>POJ-104</div>
          <div>检索样本与代码对</div>
          <div>TF-IDF · CodeBERT · GraphCodeBERT · UniXcoder</div>
          <div>混合重排序</div>
          <div>指标汇总与错误分析</div>
        </div>
      </section>
      <section class="band">
        <div class="section-title">
          <h2>方法概览</h2>
          <p>检索模型负责快速召回候选代码，Cross-Encoder 负责代码对级别的精细判断，混合方法进一步融合两类模型信号。</p>
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
        <p>语义检索任务主要关注 MAP@R、Recall 和 MRR；代码对分类任务主要关注 F1。</p>
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
              <h3>代码对分类 F1</h3>
              <span>越高越好</span>
            </div>
            <div id="classificationChart" class="rank-chart"></div>
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
      <section class="band">
        <h2>困难负例消融</h2>
        <div id="ablationViz" class="ablation-viz"></div>
        <div id="ablationGrid" class="delta-grid"></div>
      </section>
    </section>

    <section id="errors" class="view">
      <div class="section-title">
        <h2>错误分析</h2>
        <p>本页汇总模型在高词法相似负例、低词法相似正例、长代码截断和边界样本上的典型表现。</p>
      </div>
      <div id="errorCards" class="error-cards"></div>
      <h2 class="detail-heading">详细说明</h2>
      <div id="errorAnalysis" class="markdown-panel"></div>
    </section>

    <section id="demo" class="view">
      <div class="section-title">
        <h2>代码对推理演示</h2>
        <p>推理演示默认读取本地 <code>outputs/checkpoints</code>。若尚未同步模型文件，系统会保留结果展示功能，并在推理时提示需要同步的目录。</p>
      </div>
      <div class="demo-layout">
        <label>
          模型
          <select id="modelSelect">
            <option value="graphcodebert_cls">GraphCodeBERT</option>
            <option value="codebert_cls">CodeBERT</option>
            <option value="graphcodebert_hard_negatives">GraphCodeBERT + Hard Negatives</option>
          </select>
        </label>
        <div class="editor-grid">
          <label>代码片段 1<textarea id="code1" spellcheck="false">int main(){int a,b;cin&gt;&gt;a&gt;&gt;b;cout&lt;&lt;a+b;}</textarea></label>
          <label>代码片段 2<textarea id="code2" spellcheck="false">int main(){long x,y;scanf("%ld%ld",&amp;x,&amp;y);printf("%ld",x+y);}</textarea></label>
        </div>
        <button id="runInfer" class="primary">运行推理</button>
        <div id="inferResult" class="result-panel">输入两段代码后可进行语义等价判断。</div>
      </div>
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

.demo-layout {
  display: grid;
  gap: 14px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  box-shadow: var(--shadow-soft);
}

select, textarea {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: var(--ink);
  font: inherit;
  transition: border-color 160ms ease, box-shadow 160ms ease;
}

select {
  max-width: 340px;
  padding: 10px;
}

select:focus, textarea:focus {
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

textarea {
  min-height: 260px;
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
  min-height: 58px;
  padding: 14px;
  white-space: pre-wrap;
  color: #2b3746;
  background: var(--panel-tint);
}

@media (max-width: 1120px) {
  .topbar {
    grid-template-columns: 1fr;
    align-items: start;
  }
  .header-status {
    justify-self: start;
  }
  .hero-panel, .two-col {
    grid-template-columns: 1fr;
  }
  .method-cards, .metric-highlights {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .viz-grid, .error-cards, .ablation-viz {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 860px) {
  .editor-grid {
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
  if (!state.summary?.checkpoints) return null;
  return state.summary.checkpoints.find((item) => item.checkpoint === modelId) || null;
};

const inferErrorText = (response, data) => {
  if (data?.error || data?.hint) {
    return `${data.error || "推理失败。"}\n${data.hint || ""}`.trim();
  }
  if (!response.ok) {
    return `推理接口返回错误（HTTP ${response.status}）。`;
  }
  return "推理失败，请稍后重试。";
};

const ERROR_SUMMARY = [
  ["高词法相似负例", "不同题目的程序可能共享高度相似的循环、数组和输入输出模板，词法方法容易误判。"],
  ["低词法相似正例", "同一题目的代码可以使用完全不同的变量组织、分支结构和实现风格。"],
  ["长代码截断", "部分样本 token 数远超最大长度 512，模型可能无法观察完整逻辑。"],
  ["不同算法策略", "同一问题下可能存在通用算法、特殊样例处理或不同复杂度实现。"],
  ["神经模型失误", "部分高 token 重合正例仍被 GraphCodeBERT 给出较低分数。"],
  ["语义模型优势", "混合模型能在 TF-IDF top-1 失败时召回同题目代码，体现语义表示价值。"],
];

const escapeHtml = (text) => String(text)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

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
  const best = rows[0]?.metrics[metricKey] || 1;
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

const renderAblationViz = (summary) => {
  const before = summary.ablation.without_hard_negatives || {};
  const after = summary.ablation.with_hard_negatives || {};
  const keys = ["map@r", "recall@1", "recall@5", "recall@10", "mrr"];
  document.getElementById("ablationViz").innerHTML = keys.map((key) => {
    const beforeValue = typeof before[key] === "number" ? before[key] : 0;
    const afterValue = typeof after[key] === "number" ? after[key] : 0;
    const maxValue = Math.max(beforeValue, afterValue, 1);
    return `
      <article class="ablation-card">
        <h3>${escapeHtml(key)}</h3>
        <div class="paired-bars">
          <div class="paired-row">
            <span>原始</span>
            <div class="paired-track"><div class="paired-fill" style="width:${Math.max(3, beforeValue / maxValue * 100)}%"></div></div>
            <strong>${fmt(beforeValue)}</strong>
          </div>
          <div class="paired-row after">
            <span>困难</span>
            <div class="paired-track"><div class="paired-fill" style="width:${Math.max(3, afterValue / maxValue * 100)}%"></div></div>
            <strong>${fmt(afterValue)}</strong>
          </div>
        </div>
      </article>
    `;
  }).join("");
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

const renderSummary = (summary) => {
  const bestRetrieval = summary.best_retrieval;
  const bestClassifier = summary.best_classifier;
  document.getElementById("bestRetrieval").textContent = bestRetrieval ? bestRetrieval.name : "-";
  document.getElementById("bestRetrievalMeta").textContent = bestRetrieval ? `MAP@R ${fmt(bestRetrieval.metrics["map@r"])}` : "MAP@R";
  document.getElementById("bestClassifier").textContent = bestClassifier ? bestClassifier.name : "-";
  document.getElementById("bestClassifierMeta").textContent = bestClassifier ? `F1 ${fmt(bestClassifier.metrics.f1)}` : "F1";

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
    summary.best_classifier,
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
  renderRankChart("classificationChart", summary.methods, "f1");
  renderAblationViz(summary);

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

  const delta = summary.ablation.delta_with_minus_without || {};
  const keys = ["map@r", "recall@1", "recall@5", "recall@10", "mrr"];
  document.getElementById("ablationGrid").innerHTML = keys.map((key) => `
    <div class="delta">
      <span>${escapeHtml(key)}</span>
      <strong>${fmt(delta[key])}</strong>
    </div>
  `).join("");
};

const renderErrors = (summary) => {
  renderErrorCards();
  document.getElementById("errorAnalysis").innerHTML = markdownLite(summary.error_analysis);
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
  result.textContent = "模型加载中，首次推理可能会慢一些。";
  try {
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: modelId,
        code1: document.getElementById("code1").value,
        code2: document.getElementById("code2").value
      })
    });
    let data = null;
    try {
      data = await response.json();
    } catch (error) {
      data = null;
    }
    if (!response.ok || !data?.ok) {
      result.textContent = inferErrorText(response, data);
      return;
    }
    result.textContent = `预测结果：${data.label}\\n语义等价概率：${data.semantic_equivalence_score.toFixed(4)}\\n运行设备：${data.device}\\n模型路径：${data.checkpoint}`;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    result.textContent = `无法连接推理接口。\n${backendBootHint}\n原始错误：${message}`;
  } finally {
    button.disabled = false;
  }
};

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => setView(tab.dataset.view)));
document.getElementById("runInfer").addEventListener("click", runInfer);

const boot = (summary) => {
    state.summary = summary;
    renderSummary(summary);
    renderMetrics(summary);
    renderErrors(summary);
};

if (window.__SUMMARY__) {
  boot(window.__SUMMARY__);
} else {
  fetch("/api/summary")
    .then((response) => response.json())
    .then(boot)
    .catch((error) => {
      document.body.insertAdjacentHTML("afterbegin", `<pre>${escapeHtml(error)}</pre>`);
    });
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

from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"

COLORS = {
    "ink": "#18212f",
    "muted": "#657184",
    "line": "#dce4ec",
    "grid": "#edf2f5",
    "green": "#147765",
    "blue": "#285fae",
    "red": "#b64e43",
    "gold": "#a66f12",
    "violet": "#6554b5",
    "panel": "#f8fafb",
}

METHODS = [
    ("TF-IDF", "tfidf_results.json", COLORS["gold"]),
    ("UniXcoder", "unixcoder_retrieval_results.json", COLORS["blue"]),
    ("Label-aware", "unixcoder_label_aware_results.json", COLORS["green"]),
    ("SupCon CE", "unixcoder_supcon_ce_k2_w02_results.json", COLORS["violet"]),
]

METRICS = [
    ("map@r", "MAP@R"),
    ("recall@1", "R@1"),
    ("recall@5", "R@5"),
    ("recall@10", "R@10"),
    ("mrr", "MRR"),
]


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def method_rows() -> list[dict]:
    rows = []
    for label, filename, color in METHODS:
        metrics = load_json(filename)
        rows.append({"label": label, "color": color, "metrics": metrics})
    return rows


def tag(name: str, attrs: dict | None = None, body: str | None = None) -> str:
    attrs = attrs or {}
    attr = " ".join(f'{key}="{escape(str(value))}"' for key, value in attrs.items() if value is not None)
    if body is None:
        return f"<{name} {attr}/>" if attr else f"<{name}/>"
    return f"<{name} {attr}>{body}</{name}>" if attr else f"<{name}>{body}</{name}>"


def text(x: float, y: float, value: str, size: int = 16, weight: int = 400, fill: str = COLORS["ink"], anchor: str = "start") -> str:
    return tag(
        "text",
        {
            "x": round(x, 2),
            "y": round(y, 2),
            "font-size": size,
            "font-weight": weight,
            "fill": fill,
            "text-anchor": anchor,
            "font-family": "Inter, Arial, sans-serif",
        },
        escape(value),
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str | None = None, radius: int = 0, opacity: float | None = None) -> str:
    return tag(
        "rect",
        {
            "x": round(x, 2),
            "y": round(y, 2),
            "width": round(w, 2),
            "height": round(h, 2),
            "rx": radius,
            "fill": fill,
            "stroke": stroke,
            "opacity": opacity,
        },
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = COLORS["line"], width: float = 1, dash: str | None = None) -> str:
    return tag(
        "line",
        {
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
            "stroke": stroke,
            "stroke-width": width,
            "stroke-dasharray": dash,
        },
    )


def path(d: str, fill: str = "none", stroke: str = COLORS["ink"], width: float = 2) -> str:
    return tag("path", {"d": d, "fill": fill, "stroke": stroke, "stroke-width": width, "stroke-linecap": "round", "stroke-linejoin": "round"})


def svg(width: int, height: int, body: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            tag("rect", {"width": width, "height": height, "fill": "#ffffff"}),
            body,
            "</svg>",
            "",
        ]
    )


def write_svg(name: str, width: int, height: int, body: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / name).write_text(svg(width, height, body), encoding="utf-8")


def title_block(title: str, subtitle: str) -> str:
    return text(52, 52, title, 28, 800) + text(52, 82, subtitle, 14, 500, COLORS["muted"])


def figure_metric_bars(rows: list[dict]) -> None:
    width, height = 1240, 720
    left, right, top, bottom = 120, 50, 145, 105
    chart_w = width - left - right
    chart_h = height - top - bottom
    parts = [title_block("Retrieval performance on POJ-104", "MAP@R is the primary retrieval metric; all bars use the same 0-1 scale.")]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + chart_h * (1 - tick)
        parts.append(line(left, y, width - right, y, COLORS["grid"]))
        parts.append(text(left - 14, y + 5, f"{tick:.2f}", 12, 500, COLORS["muted"], "end"))
    group_w = chart_w / len(METRICS)
    bar_w = 34
    for m_idx, (key, label) in enumerate(METRICS):
        gx = left + m_idx * group_w
        parts.append(text(gx + group_w / 2, height - 64, label, 15, 800, COLORS["ink"], "middle"))
        for r_idx, row in enumerate(rows):
            value = row["metrics"][key]
            x = gx + group_w / 2 - (len(rows) * bar_w + (len(rows) - 1) * 8) / 2 + r_idx * (bar_w + 8)
            h = value * chart_h
            y = top + chart_h - h
            parts.append(rect(x, y, bar_w, h, row["color"], radius=5))
            if key == "map@r":
                parts.append(text(x + bar_w / 2, y - 8, f"{value:.3f}", 11, 700, COLORS["muted"], "middle"))
    legend_x = width - right - 520
    for idx, row in enumerate(rows):
        x = legend_x + idx * 130
        parts.append(rect(x, 105, 14, 14, row["color"], radius=3))
        parts.append(text(x + 22, 117, row["label"], 13, 700))
    parts.append(line(left, top + chart_h, width - right, top + chart_h, COLORS["line"], 1.4))
    write_svg("fig1_retrieval_metrics.svg", width, height, "\n".join(parts))


def figure_metric_matrix(rows: list[dict]) -> None:
    width, height = 1160, 620
    x0, y0 = 78, 145
    cell_w, label_w, cell_h = 152, 220, 72
    parts = [title_block("Metric matrix", "Darker cells indicate stronger relative performance within each metric column.")]
    max_by_key = {key: max(row["metrics"][key] for row in rows) for key, _ in METRICS}
    min_by_key = {key: min(row["metrics"][key] for row in rows) for key, _ in METRICS}
    parts.append(rect(x0, y0, label_w + cell_w * len(METRICS), cell_h * (len(rows) + 1), "#ffffff", COLORS["line"], 10))
    parts.append(rect(x0, y0, label_w + cell_w * len(METRICS), cell_h, COLORS["panel"], radius=10))
    parts.append(text(x0 + 20, y0 + 45, "Method", 14, 800))
    for idx, (_, label) in enumerate(METRICS):
        parts.append(text(x0 + label_w + idx * cell_w + cell_w / 2, y0 + 45, label, 14, 800, COLORS["ink"], "middle"))
    for r_idx, row in enumerate(rows):
        y = y0 + cell_h * (r_idx + 1)
        parts.append(line(x0, y, x0 + label_w + cell_w * len(METRICS), y, COLORS["line"]))
        parts.append(text(x0 + 20, y + 45, row["label"], 14, 800))
        for c_idx, (key, _) in enumerate(METRICS):
            value = row["metrics"][key]
            denom = max(max_by_key[key] - min_by_key[key], 1e-9)
            ratio = (value - min_by_key[key]) / denom
            alpha = 0.10 + ratio * 0.36
            x = x0 + label_w + c_idx * cell_w
            parts.append(rect(x, y, cell_w, cell_h, COLORS["green"], opacity=alpha))
            parts.append(text(x + cell_w / 2, y + 45, f"{value:.4f}", 14, 800, COLORS["ink"], "middle"))
    for idx in range(len(METRICS) + 1):
        x = x0 + label_w + idx * cell_w
        parts.append(line(x, y0, x, y0 + cell_h * (len(rows) + 1), COLORS["line"]))
    write_svg("fig2_metric_matrix.svg", width, height, "\n".join(parts))


def figure_mapr_improvement(rows: list[dict]) -> None:
    width, height = 1080, 600
    baseline = next(row for row in rows if row["label"] == "UniXcoder")
    base = baseline["metrics"]["map@r"]
    deltas = [(row["label"], row["metrics"]["map@r"] - base, row["color"]) for row in rows if row["label"] != "UniXcoder"]
    max_abs = max(abs(delta) for _, delta, _ in deltas)
    parts = [title_block("MAP@R change relative to UniXcoder", f"Baseline UniXcoder MAP@R = {base:.4f}; positive values indicate ranking improvements.")]
    x0, y0, track_w, row_h = 260, 170, 650, 82
    zero = x0 + track_w / 2
    for idx, (label, delta, color) in enumerate(deltas):
        y = y0 + idx * row_h
        parts.append(text(74, y + 30, label, 15, 800))
        parts.append(rect(x0, y + 12, track_w, 20, "#f0f3f5", COLORS["line"], 10))
        parts.append(line(zero, y + 8, zero, y + 38, "#aebbc7", 1.5))
        bar_w = abs(delta) / max_abs * (track_w / 2)
        x = zero if delta >= 0 else zero - bar_w
        parts.append(rect(x, y + 16, bar_w, 12, color if delta >= 0 else COLORS["red"], radius=6))
        sign = "+" if delta >= 0 else ""
        parts.append(text(940, y + 31, f"{sign}{delta:.4f}", 16, 800, color if delta >= 0 else COLORS["red"], "end"))
    parts.append(text(zero, 480, "0", 12, 700, COLORS["muted"], "middle"))
    parts.append(text(x0, 480, f"-{max_abs:.3f}", 12, 700, COLORS["muted"], "middle"))
    parts.append(text(x0 + track_w, 480, f"+{max_abs:.3f}", 12, 700, COLORS["muted"], "middle"))
    write_svg("fig3_mapr_improvement.svg", width, height, "\n".join(parts))


def rounded_box(x: float, y: float, w: float, h: float, title: str, body: str, fill: str, stroke: str) -> str:
    return "\n".join(
        [
            rect(x, y, w, h, fill, stroke, 14),
            text(x + 20, y + 34, title, 17, 800),
            text(x + 20, y + 63, body, 13, 600, COLORS["muted"]),
        ]
    )


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = COLORS["ink"]) -> str:
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 10
    p1 = (x2 - head * math.cos(angle - 0.5), y2 - head * math.sin(angle - 0.5))
    p2 = (x2 - head * math.cos(angle + 0.5), y2 - head * math.sin(angle + 0.5))
    return line(x1, y1, x2, y2, color, 2) + path(f"M {x2} {y2} L {p1[0]} {p1[1]} L {p2[0]} {p2[1]} Z", fill=color, stroke=color, width=1)


def figure_method_framework() -> None:
    width, height = 1280, 690
    parts = [title_block("Semantic code reuse retrieval framework", "The final model optimizes same-problem retrieval rather than fixed pair classification.")]
    y = 170
    boxes = [
        (70, y, 210, 112, "POJ-104 code", "query + candidate pool", "#f9fbfc", COLORS["line"]),
        (340, y, 220, 112, "UniXcoder encoder", "shared dual-tower weights", "#edf4ff", COLORS["blue"]),
        (620, y, 230, 112, "Embedding space", "cosine similarity ranking", "#edf5f2", COLORS["green"]),
        (910, y, 250, 112, "Top-K retrieval", "same problem should rank high", "#f7f0fb", COLORS["violet"]),
    ]
    for box in boxes:
        parts.append(rounded_box(*box))
    for idx in range(len(boxes) - 1):
        x1 = boxes[idx][0] + boxes[idx][2]
        x2 = boxes[idx + 1][0]
        parts.append(arrow(x1 + 12, y + 56, x2 - 12, y + 56, COLORS["muted"]))
    parts.append(rounded_box(318, 395, 290, 112, "Label-aware loss", "mask same-problem false negatives", "#fff8ec", COLORS["gold"]))
    parts.append(rounded_box(672, 395, 318, 112, "SupCon + CE objective", "pull positives close; separate classes", "#eef6f4", COLORS["green"]))
    parts.append(arrow(462, 395, 462, 292, COLORS["gold"]))
    parts.append(arrow(831, 395, 744, 292, COLORS["green"]))
    parts.append(text(640, 602, "Outcome: MAP@R improves from 0.9098 to 0.9254", 20, 800, COLORS["ink"], "middle"))
    write_svg("fig4_method_framework.svg", width, height, "\n".join(parts))


def figure_error_repair_case() -> None:
    width, height = 1180, 650
    parts = [title_block("Representative retrieval error repaired by SupCon CE", "Query test_7943: the first same-problem candidate moves from rank 1083 to rank 1.")]
    y = 178
    parts.append(rounded_box(70, y, 250, 100, "Query", "test_7943 / problem_96", "#f9fbfc", COLORS["line"]))
    parts.append(rounded_box(445, 122, 300, 130, "UniXcoder baseline", "top-1: test_5720 / problem_92", "#fff7f4", COLORS["red"]))
    parts.append(rounded_box(445, 354, 300, 130, "SupCon CE", "top-1: test_7600 / problem_96", "#eef6f4", COLORS["green"]))
    parts.append(arrow(330, y + 48, 430, 184, COLORS["red"]))
    parts.append(arrow(330, y + 72, 430, 418, COLORS["green"]))
    parts.append(text(810, 174, "wrong problem", 18, 800, COLORS["red"]))
    parts.append(text(810, 214, "first positive rank: 1083", 15, 700, COLORS["muted"]))
    parts.append(text(810, 406, "same problem", 18, 800, COLORS["green"]))
    parts.append(text(810, 446, "first positive rank: 1", 15, 700, COLORS["muted"]))
    parts.append(rect(70, 548, 1040, 46, "#f8fafb", COLORS["line"], 10))
    parts.append(text(92, 578, "Interpretation: supervised positives reshape the embedding neighborhood, reducing template-level semantic confusion.", 16, 700))
    write_svg("fig5_repaired_case.svg", width, height, "\n".join(parts))


def write_index() -> None:
    content = """# Paper Figures

Standalone SVG figures generated from the project results.

- `fig1_retrieval_metrics.svg`: retrieval metrics grouped by method.
- `fig2_metric_matrix.svg`: compact metric heatmap.
- `fig3_mapr_improvement.svg`: MAP@R changes relative to UniXcoder.
- `fig4_method_framework.svg`: task-specific method framework.
- `fig5_repaired_case.svg`: representative retrieval error repaired by SupCon CE.
"""
    (FIGURES / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    rows = method_rows()
    figure_metric_bars(rows)
    figure_metric_matrix(rows)
    figure_mapr_improvement(rows)
    figure_method_framework()
    figure_error_repair_case()
    write_index()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"

METHODS = [
    ("TF-IDF", "tfidf_results.json", "#b07a16"),
    ("UniXcoder", "unixcoder_retrieval_results.json", "#2f63ad"),
    ("Label-aware", "unixcoder_label_aware_results.json", "#1b7f6d"),
    ("SupCon CE", "unixcoder_supcon_ce_k2_w02_results.json", "#6a56b8"),
]

METRICS = [
    ("map@r", "MAP@R"),
    ("recall@1", "R@1"),
    ("recall@5", "R@5"),
    ("mrr", "MRR"),
]


def set_cvpr_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_methods() -> list[dict]:
    rows = []
    for label, filename, color in METHODS:
        rows.append({"label": label, "metrics": load_json(filename), "color": color})
    return rows


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png")
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)


def draw_main_results(rows: list[dict]) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.75),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.28},
        constrained_layout=True,
    )

    ax = axes[0]
    x = list(range(len(METRICS)))
    bar_width = 0.18
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]
    for row, offset in zip(rows, offsets):
        values = [row["metrics"][key] for key, _ in METRICS]
        ax.bar([i + offset for i in x], values, width=bar_width, color=row["color"], label=row["label"])

    ax.set_title("(a) Retrieval metrics on POJ-104", loc="left", fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRICS])
    ax.grid(axis="y", color="#d9dfe6", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), handlelength=1.0, columnspacing=1.1)

    ax = axes[1]
    baseline = next(row for row in rows if row["label"] == "UniXcoder")["metrics"]["map@r"]
    labels = []
    deltas = []
    colors = []
    for row in rows:
        if row["label"] in {"TF-IDF", "UniXcoder"}:
            continue
        labels.append(row["label"])
        deltas.append(row["metrics"]["map@r"] - baseline)
        colors.append(row["color"] if row["metrics"]["map@r"] >= baseline else "#b64e43")
    ax.axvline(0, color="#596575", linewidth=0.8)
    ax.barh(labels, deltas, color=colors, height=0.55)
    for y, value in enumerate(deltas):
        ax.text(value + 0.00045, y, f"+{value:.4f}", va="center", ha="left", fontsize=7)
    ax.set_title("(b) MAP@R change vs. UniXcoder", loc="left", fontweight="bold")
    ax.set_xlabel("Delta MAP@R")
    ax.set_xlim(0, 0.0175)
    ax.grid(axis="x", color="#d9dfe6", linewidth=0.5, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    save_figure(fig, "fig1_main_results_cvpr")


def add_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], title: str, body: str, fc: str, ec: str) -> None:
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        linewidth=0.9,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + 0.03, y + h - 0.08, title, fontweight="bold", fontsize=8, va="top")
    ax.text(x + 0.03, y + h - 0.19, body, fontsize=7, va="top", color="#475569")


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#475569") -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=0.9, color=color)
    ax.add_patch(arrow)


def draw_method_framework() -> None:
    fig, ax = plt.subplots(figsize=(7.05, 2.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.02, 0.58), (0.18, 0.22), "POJ-104 code", "query + candidate pool", "#f8fafc", "#cbd5e1")
    add_box(ax, (0.27, 0.58), (0.20, 0.22), "UniXcoder", "shared dual-tower encoder", "#eef4ff", "#2f63ad")
    add_box(ax, (0.55, 0.58), (0.20, 0.22), "Embedding space", "cosine similarity ranking", "#edf8f4", "#1b7f6d")
    add_box(ax, (0.82, 0.58), (0.16, 0.22), "Top-K list", "same problem ranks high", "#f4f0ff", "#6a56b8")

    add_arrow(ax, (0.21, 0.69), (0.265, 0.69))
    add_arrow(ax, (0.48, 0.69), (0.545, 0.69))
    add_arrow(ax, (0.76, 0.69), (0.815, 0.69))

    add_box(ax, (0.25, 0.18), (0.24, 0.20), "Label-aware loss", "mask same-problem false negatives", "#fff7e8", "#b07a16")
    add_box(ax, (0.57, 0.18), (0.28, 0.20), "SupCon + CE", "pull positives close; separate classes", "#ecf7f3", "#1b7f6d")
    add_arrow(ax, (0.37, 0.39), (0.37, 0.57), "#b07a16")
    add_arrow(ax, (0.71, 0.39), (0.66, 0.57), "#1b7f6d")

    ax.text(0.01, 0.96, "Semantic retrieval framework", fontsize=10, fontweight="bold", va="top")
    ax.text(0.01, 0.90, "The task is candidate retrieval for code reuse detection, not fixed-pair classification.", fontsize=7, color="#475569")
    ax.text(0.50, 0.04, "Final MAP@R: 0.9254 (+0.0156 over local UniXcoder)", fontsize=8, fontweight="bold", ha="center")
    save_figure(fig, "fig2_method_framework_cvpr")


def draw_error_repair() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.30})

    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_box(ax, (0.03, 0.58), (0.32, 0.20), "Query", "test_7943 / problem_96", "#f8fafc", "#cbd5e1")
    add_box(ax, (0.56, 0.70), (0.39, 0.18), "Baseline top-1", "test_5720 / problem_92", "#fff1ed", "#b64e43")
    add_box(ax, (0.56, 0.32), (0.39, 0.18), "SupCon CE top-1", "test_7600 / problem_96", "#edf8f4", "#1b7f6d")
    add_arrow(ax, (0.36, 0.68), (0.55, 0.78), "#b64e43")
    add_arrow(ax, (0.36, 0.64), (0.55, 0.42), "#1b7f6d")
    ax.text(0.03, 0.96, "(a) Repaired retrieval neighborhood", fontweight="bold", fontsize=9, va="top")
    ax.text(0.56, 0.63, "wrong problem", color="#b64e43", fontsize=7, fontweight="bold")
    ax.text(0.56, 0.25, "same problem", color="#1b7f6d", fontsize=7, fontweight="bold")

    ax = axes[1]
    labels = ["UniXcoder", "SupCon CE"]
    ranks = [1083, 1]
    colors = ["#b64e43", "#1b7f6d"]
    ax.bar(labels, ranks, color=colors, width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("First positive rank (log)")
    ax.set_title("(b) Positive candidate moves to rank 1", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d9dfe6", linewidth=0.5, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, value in enumerate(ranks):
        ax.text(i, value * 1.25, str(value), ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("Error analysis aligned with the proposed improvement", x=0.01, y=1.02, ha="left", fontweight="bold")
    save_figure(fig, "fig3_error_repair_cvpr")


def write_readme() -> None:
    content = """# Paper Figures

These figures are generated with Python/matplotlib in a compact CVPR-style layout.
GitHub README embeds the PNG versions; the PDF versions are intended for reports or slides.

- `fig1_main_results_cvpr.png` / `.pdf`: main retrieval metrics and MAP@R deltas.
- `fig2_method_framework_cvpr.png` / `.pdf`: task-specific method framework.
- `fig3_error_repair_cvpr.png` / `.pdf`: representative retrieval error repaired by SupCon CE.
"""
    (FIGURES / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    set_cvpr_style()
    rows = load_methods()
    draw_main_results(rows)
    draw_method_framework()
    draw_error_repair()
    write_readme()
    print(f"Wrote matplotlib figures to {FIGURES}")


if __name__ == "__main__":
    main()

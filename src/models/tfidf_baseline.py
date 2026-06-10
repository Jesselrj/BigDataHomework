from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.io import configure_hf_environment, load_yaml, read_jsonl, write_json
from src.utils.logging import log_environment, setup_logger
from src.utils.metrics import retrieval_metrics

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|==|!=|<=|>=|&&|\|\||[{}()\[\];,.*+\-/<>=%]")


def code_tokenizer(code: str) -> list[str]:
    return TOKEN_RE.findall(code)


def as_tfidf_text(code: str) -> str:
    return " ".join(code_tokenizer(code))


class TfidfRetrievalModel:
    def __init__(self, max_features: int | None = None, ngram_range: tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=1)

    def fit_transform(self, code: list[str]):
        return self.vectorizer.fit_transform([as_tfidf_text(c) for c in code])


def evaluate(rows: list[dict], cfg: dict) -> dict:
    model = TfidfRetrievalModel(max_features=cfg.get("max_features"), ngram_range=tuple(cfg.get("ngram_range", [1, 2])))
    matrix = model.fit_transform([row["code"] for row in rows])
    scores = cosine_similarity(matrix)
    metrics = retrieval_metrics(
        np.asarray(scores),
        [row["id"] for row in rows],
        [row["problem_id"] for row in rows],
        [row["id"] for row in rows],
        [row["problem_id"] for row in rows],
    )
    metrics["method"] = "TF-IDF"
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tfidf.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    configure_hf_environment(cfg)
    logger = setup_logger("tfidf", cfg["paths"]["logs_dir"])
    log_environment(logger)
    rows = read_jsonl(cfg["paths"]["test_retrieval"])
    if args.debug:
        rows = rows[:128]
    metrics = evaluate(rows, cfg)
    out = Path(cfg.get("result_file", "outputs/results/tfidf_results.json"))
    write_json(out, metrics)
    logger.info("metrics=%s", metrics)


if __name__ == "__main__":
    main()

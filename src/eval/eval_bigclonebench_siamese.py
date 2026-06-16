from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.models.siamese_encoder import SiameseEncoder, load_code_tokenizer
from src.utils.io import configure_hf_environment, load_yaml, write_json, write_jsonl
from src.utils.metrics import classification_metrics


@torch.no_grad()
def encode_codes(model: SiameseEncoder, tokenizer, rows: list[dict], device: torch.device, max_length: int, batch_size: int) -> np.ndarray:
    model.eval()
    vectors = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch = tokenizer([r["code"] for r in batch_rows], truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        vectors.append(model.encode_batch(batch).cpu().numpy())
    return np.concatenate(vectors, axis=0)


def read_pairs_and_codes(paths: list[str]) -> tuple[dict[str, str], dict[str, list[dict]]]:
    code_by_id: dict[str, str] = {}
    pairs_by_split: dict[str, list[dict]] = {}
    for path in paths:
        split = Path(path).name.replace("_pairs.jsonl", "")
        pairs = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                row["id1"] = str(row["id1"])
                row["id2"] = str(row["id2"])
                code_by_id.setdefault(row["id1"], row["code1"])
                code_by_id.setdefault(row["id2"], row["code2"])
                pairs.append(row)
        pairs_by_split[split] = pairs
    return code_by_id, pairs_by_split


def scores_for_pairs(pairs: list[dict], vectors: np.ndarray, id_to_idx: dict[str, int]) -> tuple[list[int], list[float], list[dict]]:
    labels = []
    scores = []
    preds = []
    for row in pairs:
        left = vectors[id_to_idx[str(row["id1"])]]
        right = vectors[id_to_idx[str(row["id2"])]]
        cosine = float(np.dot(left, right))
        score = (cosine + 1.0) / 2.0
        label = int(row["label"])
        labels.append(label)
        scores.append(score)
        preds.append({"id": row.get("id"), "id1": row.get("id1"), "id2": row.get("id2"), "label": label, "score": score, "prediction": int(score >= 0.5)})
    return labels, scores, preds


def best_f1_threshold(labels: list[int], scores: list[float]) -> tuple[float, dict]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    positives = max(int((y == 1).sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    best = int(np.argmax(f1))
    threshold = float(s_sorted[best])
    return threshold, classification_metrics(labels, scores, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    configure_hf_environment(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = cfg["checkpoint_dir"]
    tokenizer = load_code_tokenizer(checkpoint)
    model = SiameseEncoder.from_checkpoint(checkpoint, cfg.get("pooling", "mean"), float(cfg.get("temperature", 0.05))).to(device)

    pair_paths = [cfg["paths"]["valid_pairs"], cfg["paths"]["test_pairs"]]
    code_by_id, pairs_by_split = read_pairs_and_codes(pair_paths)
    code_rows = [{"id": code_id, "code": code_by_id[code_id]} for code_id in sorted(code_by_id)]
    if args.debug:
        code_rows = code_rows[:256]
        keep = {row["id"] for row in code_rows}
        for split, pairs in list(pairs_by_split.items()):
            pairs_by_split[split] = [p for p in pairs if p["id1"] in keep and p["id2"] in keep][:1024]
    vectors = encode_codes(model, tokenizer, code_rows, device, int(cfg["max_length"]), int(cfg["eval_batch_size"]))
    id_to_idx = {row["id"]: idx for idx, row in enumerate(code_rows)}

    split_data = {}
    for split, pairs in pairs_by_split.items():
        labels, scores, preds = scores_for_pairs(pairs, vectors, id_to_idx)
        split_data[split] = {"labels": labels, "scores": scores, "preds": preds, "fixed": classification_metrics(labels, scores)}
    tuned_threshold, tuned_valid = best_f1_threshold(split_data["validation"]["labels"], split_data["validation"]["scores"])
    tuned_test = classification_metrics(split_data["test"]["labels"], split_data["test"]["scores"], threshold=tuned_threshold)
    result = {
        "method": cfg["experiment_name"],
        "checkpoint": checkpoint,
        "num_unique_eval_code_ids": len(code_rows),
        "fixed_threshold": 0.5,
        "validation": split_data["validation"]["fixed"],
        "test": split_data["test"]["fixed"],
        "validation_tuned_threshold": tuned_threshold,
        "validation_tuned": tuned_valid,
        "test_with_validation_tuned_threshold": tuned_test,
        "metric_protocol": "BigCloneBench binary clone detection: precision/recall/F1; fixed 0.5 plus validation-calibrated threshold for the siamese score",
    }
    write_json(cfg["bcb_result_file"], result)
    write_jsonl(cfg["bcb_prediction_file"], split_data["test"]["preds"])
    print(result)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.utils.io import read_jsonl, write_json, write_jsonl


def positive_rows(path: str, source: str) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        if int(row["label"]) != 1:
            continue
        rows.append({**row, "source_dataset": source})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poj-pairs", default="data/processed/train_pairs.jsonl")
    parser.add_argument("--bcb-pairs", default="data/processed/bigclonebench_eval/train_pairs.jsonl")
    parser.add_argument("--output", default="data/processed/poj_bcb_positive_pairs.jsonl")
    parser.add_argument("--metadata", default="data/processed/poj_bcb_positive_pairs_metadata.json")
    parser.add_argument("--poj-repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    poj = positive_rows(args.poj_pairs, "POJ-104")
    bcb = positive_rows(args.bcb_pairs, "BigCloneBench")
    rows = []
    for rep in range(args.poj_repeat):
        for row in poj:
            rows.append({**row, "id": f"poj_repeat_{rep}_{row.get('id')}", "repeat": rep})
    rows.extend(bcb)
    random.Random(args.seed).shuffle(rows)
    write_jsonl(args.output, rows)
    metadata = {
        "output": args.output,
        "poj_positive_pairs": len(poj),
        "poj_repeat": args.poj_repeat,
        "bcb_positive_pairs": len(bcb),
        "total_rows": len(rows),
    }
    write_json(args.metadata, metadata)
    print(metadata)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from src.utils.io import ensure_dir, read_json, write_json

DATASET_ID = "google/code_x_glue_cc_clone_detection_big_clone_bench"


def normalize_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "id": f"{split}_{row.get('id')}",
        "code1": str(row["func1"]),
        "code2": str(row["func2"]),
        "label": int(bool(row["label"])),
        "id1": str(row["id1"]),
        "id2": str(row["id2"]),
        "dataset": "BigCloneBench",
        "split": split,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--output-dir", default="data/processed/bigclonebench_eval")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    from datasets import load_dataset

    output_dir = ensure_dir(args.output_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
    else:
        metadata = {
            "dataset_id": args.dataset_id,
            "task": "clone pair binary classification",
            "format": "JSONL with code1, code2, label",
            "splits": {},
        }
    for split in args.splits:
        dataset = load_dataset(args.dataset_id, split=split, streaming=True)
        count = 0
        out_name = f"{split}_pairs.jsonl"
        out_path = output_dir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(normalize_row(row, split), ensure_ascii=False) + "\n")
                count += 1
                if args.debug and count >= 1024:
                    break
        metadata["splits"][split] = out_name
        print(f"{split}: wrote {count} rows to {out_path}")
    write_json(metadata_path, metadata)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from typing import Any

from src.utils.io import ensure_dir, write_jsonl

DATASET_ID = "google/code_x_glue_cc_clone_detection_poj104"


def normalize_row(row: dict[str, Any], split: str, index: int) -> dict[str, str]:
    problem_id = str(row.get("label", "")).strip()
    if not problem_id:
        raise ValueError(f"Missing POJ-104 label/problem_id in {split}:{index}")
    return {
        "id": f"{split}_{row.get('id', index)}",
        "code": str(row["code"]),
        "problem_id": f"problem_{problem_id}",
        "language": "cpp",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    from datasets import load_dataset
    output_dir = ensure_dir(args.output_dir)
    ds = load_dataset(args.dataset_id)
    for split_name, split in ds.items():
        rows = []
        for i, row in enumerate(split):
            rows.append(normalize_row(row, split_name, i))
            if args.debug and len(rows) >= 256:
                break
        out_name = "validation" if split_name in {"valid", "validation"} else split_name
        write_jsonl(output_dir / f"{out_name}.jsonl", rows)
    print(f"Wrote processed POJ-104 splits to {output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from src.utils.io import ensure_dir, write_jsonl


def discover_programs(raw_dir: Path) -> list[dict[str, str]]:
    rows = []
    suffixes = {".c", ".cc", ".cpp", ".cxx", ".txt"}
    for file in sorted(raw_dir.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in suffixes:
            continue
        problem_id = file.parent.name
        code = file.read_text(encoding="utf-8", errors="replace")
        rows.append({"id": f"{problem_id}_{file.stem}", "code": code, "problem_id": str(problem_id), "language": "cpp"})
    if not rows:
        raise ValueError(f"No source files found under {raw_dir}")
    return rows


def stratified_split(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["problem_id"]].append(row)
    splits = {"train": [], "validation": [], "test": []}
    for items in groups.values():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * 0.8))
        n_valid = max(1, int(n * 0.1)) if n >= 3 else 0
        splits["train"].extend(items[:n_train])
        splits["validation"].extend(items[n_train:n_train + n_valid])
        splits["test"].extend(items[n_train + n_valid:])
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/poj104")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    rows = discover_programs(Path(args.raw_dir))
    if args.debug:
        rows = rows[:256]
    output_dir = ensure_dir(args.output_dir)
    splits = stratified_split(rows, args.seed)
    for split, split_rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_rows)
    write_jsonl(output_dir / "all.jsonl", rows)
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()

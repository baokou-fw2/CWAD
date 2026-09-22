#!/usr/bin/env python3
"""Collect benchmark scores from judge JSON files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def is_correct(item: dict) -> bool:
    return str(item.get("judge", "")).strip().lower() == "yes"


def score(records: list[dict], benchmark: str) -> float:
    if not records:
        raise ValueError(f"empty judge output for {benchmark}")
    if benchmark == "visualprobe":
        stats = defaultdict(lambda: [0, 0])
        for item in records:
            category = str(item.get("category", "unknown") or "unknown")
            stats[category][1] += 1
            stats[category][0] += int(is_correct(item))
        values = []
        for category in ("Easy", "Medium", "Hard"):
            correct, total = stats[category]
            if total == 0:
                raise ValueError(f"VisualProbe category is missing: {category}")
            values.append(100.0 * correct / total)
        return sum(values) / len(values)
    if benchmark == "cv-bench":
        stats = defaultdict(lambda: [0, 0])
        for item in records:
            category = str(item.get("type", "unknown") or "unknown")
            stats[category][1] += 1
            stats[category][0] += int(is_correct(item))
        values = []
        for category in ("2D", "3D"):
            correct, total = stats[category]
            if total == 0:
                raise ValueError(f"CV-Bench category is missing: {category}")
            values.append(100.0 * correct / total)
        return sum(values) / len(values)
    return 100.0 * sum(is_correct(item) for item in records) / len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    metrics = {}
    counts = {}
    for benchmark in benchmarks:
        path = (
            args.run_dir
            / "judge"
            / benchmark
            / f"{args.model_tag}_answer.jsonl"
        )
        if not path.is_file():
            raise FileNotFoundError(f"judge output is missing: {path}")
        records = json.loads(path.read_text(encoding="utf-8"))
        metrics[benchmark] = round(score(records, benchmark), 4)
        counts[benchmark] = len(records)

    result = {
        "model_tag": args.model_tag,
        "protocol": {
            "image_scale_divisor": 1,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 32768,
            "enable_thinking": False,
            "seed": 42,
            "judge_enable_thinking": False,
            "judge_max_tokens": 64,
            "judge_tensor_parallel_size": 2,
        },
        "metrics": metrics,
        "counts": counts,
        "average": round(sum(metrics.values()) / len(metrics), 4),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

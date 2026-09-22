#!/usr/bin/env python3
"""Turn a CWBench release into the benchmark file eval/infer.py reads.

CWBench ships its test split as JSONL, one line per *member* of a pair: member A
is the original image with the dataset's ground truth, member B is the
counterfactual edit with the option that edit was built to make true. Both share
``pair_id`` and the same ``query``. That is already the shape eval/infer.py
consumes (``question_id`` / ``images`` / ``query``), except for the image paths:
the release stores them relative to its own root, and infer.py opens them as
given, so this script resolves them once and writes an absolute-path JSON array.

    python3 eval/prepare_cwbench.py --dataset /path/to/CWBench --output eval/cwbench_test.json

The output keeps ``pair_id`` and ``member`` so eval/score_cwbench_pairs.py can
rebuild the pairs after inference; ``response`` is the gold letter.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="CWBench release directory (the one holding test.jsonl)",
    )
    parser.add_argument("--split", default="test", help="Which split file to read (default: test)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    release = args.dataset.resolve()
    source = release / f"{args.split}.jsonl"
    if not source.is_file():
        raise SystemExit(f"no {source.name} in {release}")

    items = []
    members: dict[str, set[str]] = defaultdict(set)
    missing: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        images = [str((release / entry).resolve()) for entry in row.get("images") or []]
        if not images:
            raise SystemExit(f"row {row.get('question_id')}: no image")
        for entry in images:
            if not Path(entry).is_file():
                missing.append(entry)
        pair_id = str(row.get("pair_id") or f"q{row['question_id']}")
        members[pair_id].add(str(row.get("member")))
        items.append(
            {
                "question_id": row["question_id"],
                "pair_id": pair_id,
                "member": row.get("member"),
                "images": images,
                "query": row["query"],
                "response": row["response"],
                "category": row.get("category"),
                "source_idx": row.get("source_idx"),
            }
        )

    if missing:
        raise SystemExit(
            f"{len(missing)} image(s) of {len(items)} rows are not on disk, "
            f"first missing: {missing[0]}"
        )
    incomplete = {p: sorted(m) for p, m in members.items() if set(m) != {"A", "B"}}
    if incomplete:
        raise SystemExit(
            f"{len(incomplete)} pairs are missing a member (first: "
            f"{sorted(incomplete)[0]} -> {incomplete[sorted(incomplete)[0]]}); "
            f"pair accuracy is undefined without both"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.output} ({len(items)} items over {len(members)} pairs, "
        f"from {source.name})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

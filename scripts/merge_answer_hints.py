#!/usr/bin/env python3
"""Merge answer-hint shards into one file per world, and report the gaps.

Each shard writes its own file because concurrent appends to one file over NFS
are not atomic. Merging deduplicates by `idx`, drops the error/short records
that stand in for generations that failed or came back too short to use, and
prints any indices that are still missing so they can be regenerated (rerunning
the generator fills exactly the gaps, because it resumes from what it finds).

Usage:
    merge_answer_hints.py [--base DIR] [--world a|b|both] [--rows N] [--suffix ""]
    merge_answer_hints.py --suffix _4b   # merge 4B shards into answer_hints_a_4b.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os

# The directory the generator wrote its shards to; --base wins over it.
DEFAULT_BASE = os.environ.get("HINT_OUTPUT_DIR", "")
MIN_REASONING_CHARS = 100


def merge(base: str, world: str, rows: int, suffix: str = "") -> tuple[int, list[int]]:
    seen: dict[int, dict] = {}
    dropped = 0
    shard_pattern = f"answer_hints_{world}{suffix}_s*.jsonl"
    for src in sorted(glob.glob(os.path.join(base, shard_pattern))):
        with open(src, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    dropped += 1
                    continue
                reasoning = str(record.get("reasoning", "")).strip()
                if reasoning.startswith("[ERR:") or len(reasoning) < MIN_REASONING_CHARS:
                    dropped += 1
                    continue
                seen[int(record["idx"])] = record

    out = os.path.join(base, f"answer_hints_{world}{suffix}.jsonl")
    with open(out, "w", encoding="utf-8") as handle:
        for index in sorted(seen):
            handle.write(json.dumps(seen[index], ensure_ascii=False) + "\n")

    missing = [i for i in range(rows) if i not in seen] if rows else []
    counted = f"{len(seen)}/{rows}" if rows else str(len(seen))
    print(f"[world {world}{suffix}] {counted} usable, {dropped} dropped -> {out}")
    if missing:
        print(f"[world {world}{suffix}] missing {len(missing)}: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return len(seen), missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--world", default="both", choices=["a", "b", "both"])
    parser.add_argument("--rows", type=int, default=0,
                        help="rows the merged file should cover; extra indices are reported as missing")
    parser.add_argument("--suffix", default="", help="e.g. '_4b' for 4B model output")
    args = parser.parse_args()
    if not args.base:
        parser.error("pass --base DIR or set HINT_OUTPUT_DIR")

    worlds = ["a", "b"] if args.world == "both" else [args.world]
    incomplete = {}
    for world in worlds:
        _, missing = merge(args.base, world, args.rows, args.suffix)
        if missing:
            incomplete[world] = missing

    if incomplete:
        # A handful of rows never produce a usable hint -- the model can answer in a
        # couple of characters even with left padding, and rerunning only reproduces
        # that. Report the count and let the caller decide; build_hint_parquet.py
        # covers the gap by falling back to the bare ground-truth answer.
        print(
            "WARNING: incomplete -- rerun the generator to fill what it can "
            "(it resumes), then fall back to the GT answer for the rest: "
            + ", ".join(f"{w} ({len(m)} missing)" for w, m in incomplete.items())
        )
    else:
        print("all worlds complete")


if __name__ == "__main__":
    main()

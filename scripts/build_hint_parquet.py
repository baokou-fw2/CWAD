#!/usr/bin/env python3
"""Write the generated answer hints of a run into a training parquet.

The answer-hint variant hands the teacher the same image the student sees and
makes up the information gap in the prompt instead, with a reference solution.
That hint has to reach the trainer, and the generation-time key whitelist in
ray_trainer._get_gen_batch only lets a fixed set of non-tensor keys through --
`extra_info` is one of them, a brand new column is not. So the hint goes into
extra_info.answer_hint rather than a column of its own.

The input is any runtime parquet (absolute image paths, one row per sample) and
the hints are matched by row ordinal; the output is the same rows plus that
field. Nothing is copied or symlinked from a source directory.

Usage:
    build_hint_parquet.py --source-parquet FILE --hints JSONL --output-dir DIR \
        [--row-align-check FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HINT_FIELD = "answer_hint"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hints(path: Path, expected_rows: int) -> dict[int, str]:
    hints: dict[int, str] = {}
    errors = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        reasoning = str(record.get("reasoning", ""))
        if reasoning.startswith("[ERR:") or not reasoning.strip():
            errors += 1
            continue
        hints[int(record["idx"])] = reasoning
    if errors:
        raise SystemExit(f"{errors} hint rows are errors/empty; refusing to build")
    if len(hints) != expected_rows:
        raise SystemExit(f"expected {expected_rows} usable hints, found {len(hints)}")
    return hints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-parquet",
        type=Path,
        required=True,
        help="Runtime training parquet the hints were generated against.",
    )
    parser.add_argument("--hints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--row-align-check",
        type=Path,
        default=None,
        help="optional second parquet, used only to prove that it and the source "
             "parquet are in the same row order",
    )
    args = parser.parse_args()

    source_parquet = args.source_parquet.resolve()
    output = args.output_dir.resolve()
    if not source_parquet.is_file():
        parser.error(f"source parquet does not exist: {source_parquet}")

    table = pq.read_table(source_parquet)
    rows = table.to_pylist()
    hints = load_hints(args.hints, len(rows))

    # Align hints to rows. The hint file is indexed by the row ordinal of the
    # parquet it was generated from, so if that ordinal is not the source's,
    # every hint lands on the wrong question -- invisible downstream. Assert the
    # order instead of trusting it.
    align_check = args.row_align_check or (
        Path(os.environ["HINT_ROW_ALIGN_CHECK"])
        if os.environ.get("HINT_ROW_ALIGN_CHECK")
        else None
    )
    if align_check is not None and align_check.is_file():
        other_rows = pq.read_table(align_check).to_pylist()
        mismatches = [
            i
            for i in range(min(len(rows), len(other_rows)))
            if rows[i]["prompt"] != other_rows[i]["prompt"]
        ]
        if mismatches:
            raise SystemExit(
                f"source parquet and {align_check.name} are not row-aligned "
                f"({len(mismatches)} mismatching prompts, first at {mismatches[0]})"
            )
        print(f"[hint-parquet] row alignment verified against {align_check.name} ({len(rows)} rows)")

    for index, row in enumerate(rows):
        extra_info = dict(row.get("extra_info") or {})
        extra_info[HINT_FIELD] = hints[index]
        row["extra_info"] = extra_info

    field_index = table.schema.get_field_index("extra_info")
    # Infer the struct type from every row's metadata, not from the first row:
    # a field only some rows carry would otherwise be dropped or mistyped.
    inferred = pa.Table.from_pylist([row["extra_info"] for row in rows]).schema
    schema = table.schema.set(
        field_index,
        pa.field("extra_info", pa.struct([inferred.field(n) for n in inferred.names])),
    )
    out_table = pa.Table.from_pylist(rows, schema=schema)
    output.mkdir(parents=True, exist_ok=True)
    out_parquet = output / "hinted.parquet"
    pq.write_table(out_table, out_parquet, compression="zstd")

    summary = {
        "source_parquet": str(source_parquet),
        "hints": str(args.hints),
        "rows": len(rows),
        "hint_field": f"extra_info.{HINT_FIELD}",
        "output_parquet": str(out_parquet),
        "output_parquet_sha256": sha256_file(out_parquet),
    }
    (output / "hint_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

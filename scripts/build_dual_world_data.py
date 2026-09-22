#!/usr/bin/env python3
"""Build the dual-world training parquet CWAD trains on.

The input is a dataset of ordinary VQA rows -- one record per sample, holding
the question, the ground truth, and a path to each of the two visual worlds.
Any `.jsonl` or `.parquet` with those columns works; image paths may be
relative to the dataset directory (or to --root), which keeps a dataset tree
self-contained. Nothing here depends on where the rows or the images came from.

The two worlds:

  images         world A -- the student view. The student samples its rollout
                 here and the teacher re-scores that trajectory on this image.
  teacher_images world B -- the counterfactual edit at its native resolution
                 when one exists, otherwise the dataset's own world-B column.

Both student and teacher score every trajectory in both worlds, so the only
difference between worlds is *content*: the resolution privilege of the
single-world recipe is deliberately stripped out. Every row is annotated so a
run can tell the two cases apart and, if wanted, restrict itself to real edits.

Output is a runtime parquet with absolute image paths plus a summary; point
scripts/train.sh at it with --data-parquet.

This script stops at "a directory of edited images exists". How those edits are
planned, applied and quality-checked -- the prompts that decide whether a row is
a dual-world pair at all -- is scripts/cwbench_prompts.py.

Usage:
    python3 scripts/build_dual_world_data.py \
        --dataset    /path/to/CWBench/train.jsonl \
        --edit-dir   /path/to/edits \
        --output-dir /path/to/build

Run --help for the quality-filter and privileged-layout options.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def load_rows(path: Path) -> tuple[list[dict[str, Any]], pa.Schema]:
    """Read the dataset as records, and derive the schema they must keep."""
    if not path.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {path}")
    if path.suffix == ".parquet":
        table = pq.read_table(path)
        return table.to_pylist(), table.schema
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows, pa.Table.from_pylist(rows).schema


def field(row: dict[str, Any], dotted: str) -> Any:
    """Read a possibly nested field: 'extra_info.psr_ordinal' or 'idx'."""
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def id_list(path: Path, key: str) -> set[str]:
    """Read a JSONL list of ids; the key is the bare field name."""
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        value = record.get(key, record.get("idx"))
        if value is None:
            raise ValueError(f"{path} line has neither '{key}' nor 'idx'")
        ids.add(str(value))
    if not ids:
        raise ValueError(f"no id entries found in {path}")
    return ids


def edit_index(edit_dir: Path, pattern: re.Pattern[str]) -> set[str]:
    if not edit_dir.is_dir():
        raise FileNotFoundError(f"counterfactual edit directory is missing: {edit_dir}")
    ids: set[str] = set()
    for name in os.listdir(edit_dir):
        match = pattern.match(name)
        if match:
            ids.add(match.group("id"))
    if not ids:
        raise ValueError(
            f"no file in {edit_dir} matches the --edit-pattern {pattern.pattern!r}"
        )
    return ids


def resolved(root: Path, entry: Any, row_id: str, column: str) -> Path:
    """Absolute path for one image column of the source dataset."""
    if isinstance(entry, list):
        if len(entry) != 1:
            raise ValueError(f"row {row_id}: {column} must hold exactly one image")
        entry = entry[0]
    value = None
    if isinstance(entry, dict):
        value = entry.get("image") or entry.get("path")
    elif isinstance(entry, str):
        value = entry
    if not value:
        raise ValueError(f"row {row_id}: {column} has no image path")
    path = Path(str(value))
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(f"row {row_id}: {column} image is missing: {path}")
    return path


def single_image_list(root: Path, rows: list[dict[str, Any]], column: str) -> list[Any]:
    """Return the column's entries for every row, or [] if the dataset lacks it."""
    if any(column not in row for row in rows):
        return []
    return [row[column] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the dual-world training parquet from a VQA dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Source rows: .jsonl (one object per line) or .parquet.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root that relative image paths resolve against (default: the dataset's directory).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--world-a-field",
        default="images",
        help="Column holding world A (the student view).",
    )
    parser.add_argument(
        "--world-b-field",
        default="teacher_images",
        help=(
            "Column holding the dataset's own world-B image. A row without a "
            "usable counterfactual edit falls back to it, so every sample still "
            "supplies two worlds."
        ),
    )
    parser.add_argument(
        "--edit-id-field",
        default="extra_info.psr_ordinal",
        help=(
            "Row field naming each sample's counterfactual edit (dotted for a "
            "nested field). Falls back to the row ordinal when the field is "
            "absent, so a plain question list works too."
        ),
    )
    parser.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Directory of counterfactual edits. Without it world B is --world-b-field.",
    )
    parser.add_argument(
        "--edit-pattern",
        default="idx{id}_edited.jpg",
        help=(
            "File name of the edit for a given id, with {id} as the placeholder. "
            "It must match exactly one file per id and is also used to enumerate "
            "--edit-dir."
        ),
    )
    parser.add_argument(
        "--only-ids-file",
        type=Path,
        default=None,
        help=(
            "JSONL with one id per row (key: the last component of --edit-id-field, "
            "or idx). Only those rows are emitted, and each one must have a real "
            "counterfactual edit; rows without an edit are dropped rather than "
            "falling back. Use this when the run is meant to see edits and nothing "
            "else, so the fallback rows cannot dilute the objective."
        ),
    )
    parser.add_argument(
        "--good-edit-ids-file",
        type=Path,
        default=None,
        help=(
            "JSONL with one id per row listing the edits good enough to use. Every "
            "row is still emitted, but a row whose own edit is not in this list "
            "falls back to --world-b-field -- i.e. it is treated exactly like a row "
            "that never had an edit. Use this to drop low-quality edits without "
            "dropping the rows."
        ),
    )
    parser.add_argument(
        "--privileged-both-worlds",
        action="store_true",
        help=(
            "Emit the four-image layout the strict privileged CWAD variant needs: "
            "world A seen at student resolution by the student and at original "
            "resolution by the teacher, world B likewise (student sees a half-size "
            "edit, teacher the native one). Adds teacher_world_a_images and "
            "student_world_b_images, which config/best.env names to the trainer, "
            "and requires --world-a-original-field and --edit-dir-half."
        ),
    )
    parser.add_argument(
        "--world-a-original-field",
        default=None,
        help="Column holding world A at its original resolution (for --privileged-both-worlds).",
    )
    parser.add_argument(
        "--edit-dir-half",
        type=Path,
        default=None,
        help="Half-resolution copies of the counterfactual edits, named by --edit-pattern.",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    root = (args.root or dataset.parent).resolve()
    output_dir = args.output_dir.resolve()
    if "{id}" not in args.edit_pattern:
        parser.error("--edit-pattern must contain the {id} placeholder")
    pattern = re.compile("^" + re.escape(args.edit_pattern).replace("\\{id\\}", "(?P<id>.+?)") + "$")
    id_key = args.edit_id_field.rsplit(".", 1)[-1]

    rows, schema = load_rows(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = output_dir / "dual_world.parquet"

    edits: set[str] = set()
    edit_dir = None
    if args.edit_dir is not None:
        edit_dir = args.edit_dir.resolve()
        edits = edit_index(edit_dir, pattern)

    only: set[str] | None = None
    if args.only_ids_file is not None:
        only = id_list(args.only_ids_file, id_key)
    good_edits: set[str] | None = None
    if args.good_edit_ids_file is not None:
        good_edits = id_list(args.good_edit_ids_file, id_key)

    original_world_a = []
    if args.privileged_both_worlds:
        if args.edit_dir_half is None or not args.world_a_original_field:
            raise SystemExit(
                "--privileged-both-worlds requires --world-a-original-field and --edit-dir-half"
            )
        args.edit_dir_half = args.edit_dir_half.resolve()
        original_world_a = single_image_list(
            root, rows, args.world_a_original_field
        )
        if not original_world_a:
            raise SystemExit(
                f"--world-a-original-field column is missing from the dataset: "
                f"{args.world_a_original_field}"
            )

    used_edits: set[str] = set()
    kept: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_id = field(row, args.edit_id_field)
        row_id = str(index if row_id is None else row_id)
        if only is not None and row_id not in only:
            continue

        world_a = resolved(root, row.get(args.world_a_field), row_id, args.world_a_field)
        edit_name = args.edit_pattern.replace("{id}", row_id)
        edit_path = edit_dir / edit_name if edit_dir is not None else None
        edit_qualifies = edit_path is not None and row_id in edits and edit_path.is_file()
        if edit_qualifies and good_edits is not None and row_id not in good_edits:
            edit_qualifies = False
        if edit_qualifies:
            world_b = edit_path
            source = "counterfactual_edit"
            used_edits.add(row_id)
        else:
            # No usable edit for this row: fall back to the dataset's own world-B
            # image so every sample still supplies two worlds. This applies in
            # restricted runs too -- selecting rows is about which rows are
            # wanted, not about demanding that each one carry an edit, and
            # dropping them would quietly shrink the dataset.
            world_b = resolved(root, row.get(args.world_b_field), row_id, args.world_b_field)
            source = "dataset_world_b"

        row[args.world_a_field] = [{"image": str(world_a)}]
        row[args.world_b_field] = [{"image": str(world_b)}]
        if args.privileged_both_worlds:
            row["teacher_world_a_images"] = [
                {"image": str(resolved(root, original_world_a[index], row_id, args.world_a_original_field))}
            ]
            if edit_qualifies:
                half = args.edit_dir_half / edit_name
                if not half.is_file():
                    raise FileNotFoundError(f"missing half-size edit: {half}")
                student_b = half
            else:
                # World B already *is* the dataset's world-B image, so the
                # student's view of it is its own world-A image.
                student_b = world_a
            row["student_world_b_images"] = [{"image": str(student_b)}]
        extra_info = dict(row.get("extra_info") or {})
        extra_info["dual_world_source"] = source
        extra_info["dual_world_has_edit"] = source == "counterfactual_edit"
        row["extra_info"] = extra_info
        # The ordinal in the *built* parquet: the answer-hint tools key their
        # output on it, so it has to survive any row filtering.
        row["idx"] = len(kept)
        kept.append(row)

    if not kept:
        raise SystemExit("no rows selected; check --only-ids-file against --edit-id-field")

    # Keep every original field (and its type), append the id and the two
    # bookkeeping columns, so a dataset that carries extra metadata survives.
    images_type = schema.field(args.world_a_field).type
    appended = [args.world_b_field]
    if args.privileged_both_worlds:
        appended += ["teacher_world_a_images", "student_world_b_images"]
    for extra in appended:
        if schema.get_field_index(extra) < 0:
            schema = schema.append(pa.field(extra, images_type))
    if schema.get_field_index("idx") < 0:
        schema = schema.append(pa.field("idx", pa.int64()))
    # The two bookkeeping fields are rewritten on every row, so a dataset that
    # already carries them keeps its other metadata but not their old type: a
    # duplicate name in the struct makes the parquet unreadable.
    bookkeeping = {
        "dual_world_source": pa.string(),
        "dual_world_has_edit": pa.bool_(),
    }
    if schema.get_field_index("extra_info") < 0:
        schema = schema.append(
            pa.field(
                "extra_info",
                pa.struct([pa.field(n, t) for n, t in bookkeeping.items()]),
            )
        )
    else:
        kept_fields = [
            f for f in schema.field("extra_info").type if f.name not in bookkeeping
        ]
        schema = schema.set(
            schema.get_field_index("extra_info"),
            pa.field(
                "extra_info",
                pa.struct(kept_fields + [pa.field(n, t) for n, t in bookkeeping.items()]),
            ),
        )
    table = pa.Table.from_pylist(kept, schema=schema)
    pq.write_table(table, output_parquet, compression="zstd")

    with_edit = sum(1 for r in kept if r["extra_info"]["dual_world_has_edit"])
    summary = {
        "dataset": str(dataset),
        "root": str(root),
        "world_a_field": args.world_a_field,
        "world_b_field": args.world_b_field,
        "edit_dir": str(edit_dir) if edit_dir else None,
        "edit_pattern": args.edit_pattern,
        "rows": len(kept),
        "dataset_rows": len(rows),
        "restricted_to_ids_file": str(args.only_ids_file) if args.only_ids_file else None,
        "ids_requested": len(only) if only is not None else None,
        "good_edit_ids_file": str(args.good_edit_ids_file) if args.good_edit_ids_file else None,
        "good_edits_requested": len(good_edits) if good_edits is not None else None,
        "world_b_counterfactual_edit": with_edit,
        "world_b_dataset_fallback": len(kept) - with_edit,
        "edits_available": len(edits),
        "edits_used": len(used_edits),
        "edits_unmatched": len(edits - used_edits),
        "privileged_both_worlds": bool(args.privileged_both_worlds),
        "world_a": args.world_a_field,
        "world_b": "counterfactual edit, else " + args.world_b_field,
    }
    (output_dir / "dual_world_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output_parquet}")


if __name__ == "__main__":
    main()

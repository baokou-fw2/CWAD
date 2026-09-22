#!/usr/bin/env python3
"""Cache verl's overlong-prompt filter so every training run reuses one parallel pass.

verl filters inside the trainer (`RLHFDataset.maybe_filter_out_long_prompts`) with no on-disk cache,
so every launch re-decodes every image to measure its prompt length. On this dataset that is a
single-process walk over thousands of HD images and it repeats identically each run.

This script runs the *same* class with the same config, in parallel, and writes the surviving rows to
a parquet. Training then reads that file with `filter_overlong_prompts=False`, so the verdict is
computed once and reused. Nothing is reimplemented: the filter, the chat template and the processor
sizing all come from the production code path, which is what keeps the cache honest.

The cache key covers everything that can change the verdict, and a mismatch simply re-runs the filter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from transformers import AutoProcessor, AutoTokenizer

from verl.utils.chat_template import resolve_custom_chat_template
from verl.utils.dataset.rl_dataset import RLHFDataset


def cache_key(source: Path, args: argparse.Namespace, template: str | None) -> str:
    """Stable across runs that see the same rows, which the source file's bytes are NOT.

    An upstream filter pass can rewrite its output parquet on every launch (timestamps and
    pyarrow metadata differ), so hashing that file would miss the cache every single time.
    The caller passes `--cache-key` built from the inputs that file is a pure function of;
    fall back to the bytes only when no key is given.
    """
    digest = hashlib.sha256()
    if args.cache_key:
        digest.update(args.cache_key.encode())
    else:
        digest.update(source.read_bytes())
    digest.update(f"{args.max_prompt_length}|{args.image_patch_size}|{args.model_path}".encode())
    digest.update((template or "").encode())
    return digest.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True, help="the training parquet to filter")
    parser.add_argument("--model-path", required=True, help="student checkpoint the processor comes from")
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument("--image-patch-size", type=int, default=16,
                        help="Must equal processor.image_processor.patch_size, otherwise the filter "
                             "estimates vision tokens on a different grid than the rollout uses.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--custom-chat-template-file", default=None)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cache-key", default=None,
                        help="Stable identity of the row set. Build it from the parquet and the "
                             "flags that produced it; a regenerated file's bytes are not stable.")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = {"custom_chat_template_file": args.custom_chat_template_file}
    template = resolve_custom_chat_template(model_cfg)

    key = cache_key(args.parquet, args, template)
    out_parquet = cache_dir / f"filtered_{args.max_prompt_length}_{key}.parquet"
    marker = cache_dir / f"filtered_{args.max_prompt_length}_{key}.json"

    if out_parquet.is_file() and marker.is_file():
        state = json.loads(marker.read_text(encoding="utf-8"))
        print(f"[prefilter] cache hit: {out_parquet} ({state['rows']} rows)", file=sys.stderr)
        print(out_parquet)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if template is not None:
        processor.chat_template = template
        tokenizer.chat_template = template

    # Mirrors the keys the trainer passes down, minus anything the filter does not read.
    data_config = {
        "prompt_key": "prompt",
        "image_key": "images",
        "video_key": "videos",
        "max_prompt_length": args.max_prompt_length,
        "truncation": "error",
        "filter_overlong_prompts": True,
        "filter_overlong_prompts_workers": args.workers,
        "image_patch_size": args.image_patch_size,
        "return_multi_modal_inputs": True,
        "use_shm": False,
    }

    print(f"[prefilter] filtering with {args.workers} workers "
          f"(max_prompt_length={args.max_prompt_length}, image_patch_size={args.image_patch_size})",
          file=sys.stderr)
    dataset = RLHFDataset(
        data_files=[str(args.parquet)],
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )
    filtered = dataset.dataframe
    rows = len(filtered)

    filtered.to_parquet(out_parquet)
    marker.write_text(
        json.dumps(
            {
                "source": str(args.parquet),
                "rows": rows,
                "max_prompt_length": args.max_prompt_length,
                "image_patch_size": args.image_patch_size,
                "workers": args.workers,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[prefilter] kept {rows} rows -> {out_parquet}", file=sys.stderr)
    print(out_parquet)


if __name__ == "__main__":
    main()

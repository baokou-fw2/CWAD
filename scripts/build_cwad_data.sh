#!/usr/bin/env bash
# Build the dual-world (CWAD) training parquet from any VQA dataset.
#
# Thin wrapper over scripts/build_dual_world_data.py, so that the one step which
# needs a Python environment picks up the prepared one instead of the bare
# `python3` on PATH. Every path is a flag: nothing here depends on the machine
# that produced the reported runs.
#
#   ENV_DIR=/tmp/cwad-venv ./scripts/build_cwad_data.sh \
#     --dataset    <train.jsonl or .parquet of question rows, image paths relative
#                   to --root (default: the dataset's own directory)> \
#     --edit-dir   <directory of counterfactual edits> \
#     --output-dir <a new directory>
#
# The result is <output-dir>/dual_world.parquet; pass it to training with
# --data-parquet. Pass --privileged-both-worlds --world-a-original-field COL
# --edit-dir-half <dir> for the strict privileged four-column layout, and
# --only-ids-file <qc.jsonl> to keep only rows whose edit passed QC.
# See --help for the remaining options.
#
# This wrapper only assembles images. The prompts that produce the edits and
# decide which pairs survive QC are scripts/cwbench_prompts.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ENV_DIR="${ENV_DIR:-${CWAD_PACKAGE_ROOT}/.runtime/venv}"
require_file "${ENV_DIR}/bin/python"

exec "${ENV_DIR}/bin/python" "${SCRIPT_DIR}/build_dual_world_data.py" "$@"

#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
# Prefer the interpreter the pinned environment was built with; fall back to
# whatever python3 is on PATH for the commands that only need a stdlib script.
pick_python() {
  local candidate
  for candidate in "${PACKAGE_ROOT}/.runtime/venv/bin/python" "${ENV_DIR:-}/bin/python"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] && { printf '%s\n' "${candidate}"; return; }
  done
  command -v python3
}
PYTHON="$(pick_python)"

usage() {
  cat <<'EOF'
Usage: ./run.sh COMMAND [arguments]

CWAD (Cross-World Aligned Distillation). Two supported recipes, both thin
wrappers over scripts/train.sh:

  build-data       turn a VQA dataset + counterfactual edits into the
                   dual-world parquet CWAD trains on
  train-cwad       recipe 1: dual world, no hint, EMA self-teacher (or a frozen
                   larger teacher via --teacher-model-path <dir>)
  train-cwad-hint  recipe 2: answer-hint privilege, EMA self-teacher

Every other flag is forwarded untouched to scripts/train.sh, whose --help lists
them (--gpus, --steps, --smoke, --no-privilege, --resume, --cwad-lambda, ...).

Supporting commands:
  prepare-env   Build the pinned Python 3.12 environment for this repo.
  verify        Validate model, data, source, and optionally the runtime.
  smoke         Run one training step and validate the diagnostics.
  train         Raw training entry point (no recipe flags added).
  merge         Merge FSDP actor shards into Hugging Face weights.
  eval          Run the full-resolution, LLM-judged benchmark evaluation.
  eval-cwbench  Score CWBench: per-world accuracy, Cross-World Pair Accuracy and
                the Cross-World Failure Rate, against an already-running
                OpenAI-compatible server.

Notes:
  * build-data takes any dataset whose rows carry a question, an answer and one
    image path per world; scripts/build_dual_world_data.py --help lists the
    column names it reads and how to point them at your own fields. The
    counterfactual edits are not published with this repository.
  * A non-empty --teacher-model-path selects a frozen larger teacher; passing ""
    selects the EMA self-teacher, which is what the reported CWAD runs used.
  * train-cwad-hint needs a parquet whose extra_info carries the generated
    hints; see README.md, "Training", for the three preparation steps.
  * eval-cwbench scores the CWBench release (per-world accuracy and CWPA) and
    needs an OpenAI-compatible server already serving the merged weights; it
    starts none.
  * build-data/train/eval are GPU or CPU-heavy jobs.
EOF
}

[[ $# -gt 0 ]] || { usage; exit 2; }
COMMAND=$1
shift

case "${COMMAND}" in
  build-data)
    exec bash "${PACKAGE_ROOT}/scripts/build_cwad_data.sh" "$@"
    ;;
  train-cwad)
    exec bash "${PACKAGE_ROOT}/scripts/run_cwad.sh" "$@"
    ;;
  train-cwad-hint)
    exec bash "${PACKAGE_ROOT}/scripts/run_cwad_hint.sh" "$@"
    ;;
  prepare-env)
    exec bash "${PACKAGE_ROOT}/scripts/prepare_env.sh" "$@"
    ;;
  verify)
    exec "${PYTHON}" "${PACKAGE_ROOT}/scripts/verify.py" "$@"
    ;;
  smoke)
    exec bash "${PACKAGE_ROOT}/scripts/train.sh" --smoke "$@"
    ;;
  train)
    exec bash "${PACKAGE_ROOT}/scripts/train.sh" "$@"
    ;;
  merge)
    exec bash "${PACKAGE_ROOT}/scripts/merge.sh" "$@"
    ;;
  eval)
    exec bash "${PACKAGE_ROOT}/scripts/eval.sh" "$@"
    ;;
  eval-cwbench)
    exec bash "${PACKAGE_ROOT}/eval/run_cwbench.sh" "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    printf 'Unknown command: %s\n\n' "${COMMAND}" >&2
    usage >&2
    exit 2
    ;;
esac

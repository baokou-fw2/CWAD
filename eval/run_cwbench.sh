#!/usr/bin/env bash
# CWBench evaluation -- per-world accuracy, Cross-World Pair Accuracy (CWPA) and
# Cross-World Failure Rate (CWFR).
#
# Unlike the benchmark suite driven by scripts/eval.sh, this one needs no LLM
# judge: every CWBench query asks for the option letter, and both members of a
# pair are multiple choice over the same options, so a letter match decides
# correctness and the pair is the unit of measurement.
#
#   ./eval/run_cwbench.sh --dataset /path/to/CWBench \
#       --api-base http://localhost:8000/v1/ --model-id <served model id> \
#       --model-name my-cwad-4b --out-dir /path/to/eval_out
#
# --api-base must already serve the model under evaluation -- the same
# OpenAI-compatible endpoint eval/infer.py talks to (vllm serve <merged weights>,
# or scripts/eval.sh's server). Nothing here starts or stops a server.
#
# Extra flags after -- are forwarded to infer.py (e.g. --max_tokens 32768
# --image_scale_divisor 1 --parallel_workers 64 --enable_thinking False).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

die() {
  printf '[cwad] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"
}

DATASET=""
API_BASE="${API_BASE:-http://localhost:8000/v1/}"
MODEL_ID=""
MODEL_NAME=""
OUT_DIR=""
ENV_DIR="${ENV_DIR:-${SCRIPT_DIR}/../.runtime/venv}"
BENCHMARK=cwbench
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET=$2; shift 2 ;;
    --api-base) API_BASE=$2; shift 2 ;;
    --model-id) MODEL_ID=$2; shift 2 ;;
    --model-name) MODEL_NAME=$2; shift 2 ;;
    --out-dir) OUT_DIR=$2; shift 2 ;;
    --env-dir) ENV_DIR=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; PASSTHROUGH+=("$@"); break ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${DATASET}" ]] || die "--dataset is required (the CWBench release directory)"
[[ -n "${MODEL_ID}" ]] || die "--model-id is required (the id the server serves)"
[[ -n "${MODEL_NAME}" ]] || die "--model-name is required (names the answer file)"
[[ -n "${OUT_DIR}" ]] || die "--out-dir is required"
[[ -x "${ENV_DIR}/bin/python" ]] || die "no python in ${ENV_DIR}; run ./run.sh prepare-env --env-dir ${ENV_DIR}"

PYTHON="${ENV_DIR}/bin/python"
mkdir -p "${OUT_DIR}"
BENCHMARK_JSON="${OUT_DIR}/${BENCHMARK}.json"

if [[ -s "${BENCHMARK_JSON}" ]]; then
  echo "[cwbench] reusing ${BENCHMARK_JSON} (delete it to re-prepare from ${DATASET})"
else
  "${PYTHON}" "${SCRIPT_DIR}/prepare_cwbench.py" --dataset "${DATASET}" --output "${BENCHMARK_JSON}"
fi

cd "${SCRIPT_DIR}"
"${PYTHON}" infer.py \
  --benchmark "${BENCHMARK}" \
  --benchmark_json "${BENCHMARK_JSON}" \
  --out_dir "${OUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --model_id "${MODEL_ID}" \
  --api_base "${API_BASE}" \
  "${PASSTHROUGH[@]}"

"${PYTHON}" "${SCRIPT_DIR}/score_cwbench_pairs.py" \
  "${OUT_DIR}/${BENCHMARK}/${MODEL_NAME}_answer.jsonl" \
  --json "${OUT_DIR}/${BENCHMARK}/${MODEL_NAME}_cwpa.json"

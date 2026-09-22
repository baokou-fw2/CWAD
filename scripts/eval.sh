#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage: eval.sh --model-path DIR --judge-model-path DIR --output-dir DIR [options]

Options:
  --eval-data-dir DIR     Directory holding the benchmark JSON files (default
                          .runtime/eval_data). This script does not download or
                          build them: convert each public benchmark into the
                          shape eval/infer.py reads and drop it here under the
                          file name benchmark_json_name maps its name to.
  --env-dir DIR           Prepared reproduction environment.
  --benchmarks CSV        Which benchmarks to run, comma separated. Required
                          unless CWAD_EVAL_BENCHMARKS is preset in
                          config/best.env; the accepted names are the cases of
                          benchmark_json_name below.
  --gpu-ids CSV           Visible GPU IDs (default: 0,1,2,3,4,5,6,7).
  --judge-gpu-ids CSV     Serve the judge on this disjoint set of GPUs at the
                          same time as the target. Both servers come up before
                          any work starts, so the judge's model load overlaps
                          the target's inference instead of following it.
                          Omitted (the default) keeps the original behaviour,
                          where one GPU pool is reused for both servers.
  --target-port PORT      Target model API port (default: 8000).
  --judge-port PORT       Judge API port (default: 8001).
  --parallel-workers N    Concurrent requests (default: 256).
  --resume                Resume a partially completed evaluation directory.

Inference always finishes before judging begins, because the judge reads the
answers the target produced; --judge-gpu-ids only brings the judge's server up
early, it does not let the two stages overlap.

Every item here is answered as a letter or a yes/no, so one judge scores them
all. Open-ended benchmarks are not part of this path: OK-VQA is scored by
eval/score_okvqa.py (VQA count-accuracy) over the answer file eval/infer.py
writes, and CWBench by eval/run_cwbench.sh.
EOF
}

RUNTIME_ROOT="$(default_runtime_root)"
MODEL_PATH=""
JUDGE_MODEL_PATH=""
OUTPUT_DIR=""
EVAL_DATA_DIR=""
ENV_DIR="${RUNTIME_ROOT}/venv"
BENCHMARKS="${CWAD_EVAL_BENCHMARKS:-}"
GPU_IDS="0,1,2,3,4,5,6,7"
JUDGE_GPU_IDS_OPT=""
TARGET_PORT=8000
JUDGE_PORT=8001
PARALLEL_WORKERS=256
RESUME=0
MODEL_TAG_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH=$2; shift 2 ;;
    --judge-model-path) JUDGE_MODEL_PATH=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --eval-data-dir) EVAL_DATA_DIR=$2; shift 2 ;;
    --env-dir) ENV_DIR=$2; shift 2 ;;
    --benchmarks) BENCHMARKS=$2; shift 2 ;;
    --gpu-ids) GPU_IDS=$2; shift 2 ;;
    --judge-gpu-ids) JUDGE_GPU_IDS_OPT=$2; shift 2 ;;
    --target-port) TARGET_PORT=$2; shift 2 ;;
    --judge-port) JUDGE_PORT=$2; shift 2 ;;
    --parallel-workers) PARALLEL_WORKERS=$2; shift 2 ;;
    --model-tag) MODEL_TAG_OVERRIDE=$2; shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown eval argument: $1" ;;
  esac
done

[[ -n "${MODEL_PATH}" ]] || die "--model-path is required"
[[ -n "${JUDGE_MODEL_PATH}" ]] || die "--judge-model-path is required"
[[ -n "${OUTPUT_DIR}" ]] || die "--output-dir is required"
[[ "${TARGET_PORT}" =~ ^[0-9]+$ ]] || die "--target-port must be an integer"
[[ "${JUDGE_PORT}" =~ ^[0-9]+$ ]] || die "--judge-port must be an integer"
[[ "${PARALLEL_WORKERS}" =~ ^[1-9][0-9]*$ ]] || die "--parallel-workers must be positive"
[[ -n "${BENCHMARKS}" ]] || die "no benchmarks selected: pass --benchmarks CSV. The names are the cases of benchmark_json_name below."
require_command curl
require_command setsid

MODEL_PATH="$(absolute_path "${MODEL_PATH}")"
JUDGE_MODEL_PATH="$(absolute_path "${JUDGE_MODEL_PATH}")"
OUTPUT_DIR="$(absolute_path "${OUTPUT_DIR}")"
ENV_DIR="$(absolute_path "${ENV_DIR}")"
if [[ -z "${EVAL_DATA_DIR}" ]]; then
  EVAL_DATA_DIR="${RUNTIME_ROOT}/eval_data"
fi
EVAL_DATA_DIR="$(absolute_path "${EVAL_DATA_DIR}")"

require_dir "${MODEL_PATH}"
require_dir "${JUDGE_MODEL_PATH}"
require_file "${ENV_DIR}/bin/python"
export_c_include_path "${ENV_DIR}"
if [[ ${RESUME} -eq 0 ]]; then
  require_empty_output "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}/logs" "${EVAL_DATA_DIR}"

# Cross-model students differ in hidden size from the paper's 9B, so read each
# model's own config instead of assuming one.
detect_hidden_size() {
  "${ENV_DIR}/bin/python" - "$1" <<'PY'
import json
import pathlib
import sys

config = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text(encoding="utf-8"))
text_config = config.get("text_config") or {}
size = config.get("hidden_size") or text_config.get("hidden_size")
if not isinstance(size, int):
    raise SystemExit(f"cannot determine hidden_size from {sys.argv[1]}")
print(size)
PY
}

"${ENV_DIR}/bin/python" "${SCRIPT_DIR}/verify.py" \
  --model-path "${MODEL_PATH}" \
  --expected-hidden-size "$(detect_hidden_size "${MODEL_PATH}")"
"${ENV_DIR}/bin/python" "${SCRIPT_DIR}/verify.py" \
  --model-path "${JUDGE_MODEL_PATH}" \
  --expected-hidden-size "$(detect_hidden_size "${JUDGE_MODEL_PATH}")"

# The six suites of the reported table, and the file name each one's converted
# JSON has in --eval-data-dir. All six are answered as a letter or a yes/no, so
# they share one judge: infer.py has no benchmark-specific branch, and
# judge_qwenlm.py rules out the option-letter ones and asks about the rest.
benchmark_json_name() {
  case "$1" in
    vstar) printf 'vstar.json\n' ;;
    hrbench-4k) printf 'hr_bench_4k.json\n' ;;
    realworldqa) printf 'realworldqa.json\n' ;;
    mmvp) printf 'mmvp.json\n' ;;
    hallusionbench) printf 'hallusionbench.json\n' ;;
    seedbench) printf 'seedbench.json\n' ;;
    *) die "unsupported benchmark: $1" ;;
  esac
}

IFS=',' read -r -a BENCHMARK_ARRAY <<<"${BENCHMARKS}"
for index in "${!BENCHMARK_ARRAY[@]}"; do
  BENCHMARK_ARRAY[$index]="$(printf '%s' "${BENCHMARK_ARRAY[$index]}" | xargs)"
  [[ -n "${BENCHMARK_ARRAY[$index]}" ]] || die "empty benchmark in --benchmarks"
  JSON_NAME="$(benchmark_json_name "${BENCHMARK_ARRAY[$index]}")"
  if [[ ! -f "${EVAL_DATA_DIR}/${JSON_NAME}" ]]; then
    die "missing ${EVAL_DATA_DIR}/${JSON_NAME}. Convert that benchmark into the
shape eval/infer.py reads (images / query / response / category / index) and put
it in --eval-data-dir, or drop the name from --benchmarks."
  fi
done

IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
TP_SIZE="${#GPU_ARRAY[@]}"
[[ ${TP_SIZE} -gt 0 ]] || die "--gpu-ids is empty"

# Each start_server records its process group here; both the sequential and the
# concurrent path stop what they started.
TARGET_SERVER_PID=""
JUDGE_SERVER_PID=""

stop_pid() {
  local pid=$1
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

stop_servers() {
  stop_pid "${TARGET_SERVER_PID}"
  stop_pid "${JUDGE_SERVER_PID}"
  TARGET_SERVER_PID=""
  JUDGE_SERVER_PID=""
}
trap stop_servers EXIT INT TERM

wait_for_server() {
  local port=$1
  local pid=$2
  local served_name=$3
  local deadline=$((SECONDS + 1800))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 1
    fi
    # The name has to be checked, not just "something answers on the port".
    # A previous checkpoint's server can still be listening while it shuts down,
    # and it serves a DIFFERENT model name -- so the readiness test passed, the
    # run went straight to inference, and every request came back
    # "404 model does not exist" until the new server won the port. Those
    # failures were written into the answers as [API_ERROR], which the scorer
    # counts as a wrong answer: 11 of the 17 CWAD 500-1300 checkpoints ended up
    # 23-65% contaminated this way, and the depression is random per checkpoint,
    # so it reads as model oscillation rather than as a broken pipeline.
    if curl --silent --fail "http://127.0.0.1:${port}/v1/models" 2>/dev/null \
      | grep -q "\"${served_name}\""; then
      return 0
    fi
    sleep 5
  done
  return 1
}

start_server() {
  local model_path=$1
  local served_name=$2
  local port=$3
  local log_path=$4
  local visible_gpus=$5
  local tensor_parallel_size=$6
  local max_model_len=${7:-}
  local -a serve_args=(
    serve "${model_path}"
    --host 127.0.0.1
    --port "${port}"
    --served-model-name "${served_name}"
    --tensor-parallel-size "${tensor_parallel_size}"
    --gpu-memory-utilization "${CWAD_EVAL_GPU_MEMORY_UTILIZATION}"
    --trust-remote-code
    --disable-custom-all-reduce
  )
  [[ -n "${max_model_len}" ]] && serve_args+=(--max-model-len "${max_model_len}")
  # A single-card judge is memory-bound: the 9B weights take ~17.7 GiB and the
  # CUDA-graph profiling pass then asks for another ~1 GiB, which OOMs a 24 GiB
  # card even at gpu-memory-utilization 0.90 ("Engine core initialization
  # failed", torch.OutOfMemoryError inside profile_cudagraph_memory). The slurm
  # single-card judge avoided it with --enforce-eager; expose the same knob here,
  # opt-in so the canonical multi-card path is untouched.
  [[ "${CWAD_EVAL_ENFORCE_EAGER:-0}" == "1" ]] && serve_args+=(--enforce-eager)
  log "starting vLLM service ${served_name} on ${visible_gpus}, TP=${tensor_parallel_size}"
  # vLLM 0.18 removed --disable-log-requests; --disable-uvicorn-access-log is
  # the current way to keep per-request access logging out of the server log.
  CUDA_VISIBLE_DEVICES="${visible_gpus}" setsid "${ENV_DIR}/bin/vllm" "${serve_args[@]}" \
    --disable-uvicorn-access-log \
    >"${log_path}" 2>&1 &
  # Handed back through LAST_SERVER_PID rather than a fixed global, so the
  # caller can hold a target and a judge server at the same time.
  LAST_SERVER_PID=$!
  if ! wait_for_server "${port}" "${LAST_SERVER_PID}" "${served_name}"; then
    tail -n 200 "${log_path}" >&2 || true
    die "vLLM service failed to become ready: ${served_name}"
  fi
}

# Label results after the model actually under test rather than the paper's 9B.
TARGET_SLUG="$(basename "${MODEL_PATH}" | tr '[:upper:].' '[:lower:]-')"
JUDGE_SLUG="$(basename "${JUDGE_MODEL_PATH}" | tr '[:upper:].' '[:lower:]-')"
# Default to a slug of the model directory, but let the caller name the result
# explicitly: the merged checkpoints all live in directories called step_50,
# step_100, ... so the slug alone cannot say which run a score belongs to, and
# the cross-machine tables are keyed by names like cwad-full-s200_seed42.
MODEL_TAG="${MODEL_TAG_OVERRIDE:-cwad_${TARGET_SLUG}_seed${CWAD_SEED}}"
TARGET_SERVED_NAME="cwad-${TARGET_SLUG}"
JUDGE_SERVED_NAME="${JUDGE_SLUG}-judge"
RUN_DIR="${OUTPUT_DIR}/evaluation"
mkdir -p "${RUN_DIR}/model_answer" "${RUN_DIR}/judge" "${RUN_DIR}/scores"

# Where the judge lives. Without --judge-gpu-ids it reuses the target's pool
# later, once the target has been shut down; with it, the judge gets its own
# disjoint cards and both servers stay resident together.
if [[ -n "${JUDGE_GPU_IDS_OPT}" ]]; then
  IFS=',' read -r -a JUDGE_GPU_ARRAY <<<"${JUDGE_GPU_IDS_OPT}"
  JUDGE_TP_SIZE="${#JUDGE_GPU_ARRAY[@]}"
  [[ ${JUDGE_TP_SIZE} -ge ${CWAD_JUDGE_TP} ]] || die \
    "--judge-gpu-ids must provide at least ${CWAD_JUDGE_TP} GPUs"
  for gpu in "${JUDGE_GPU_ARRAY[@]}"; do
    [[ ",${GPU_IDS}," != *",${gpu},"* ]] || die \
      "GPU ${gpu} is in both --gpu-ids and --judge-gpu-ids; the two servers must not share cards"
  done
  JUDGE_GPU_IDS="${JUDGE_GPU_IDS_OPT}"
  CONCURRENT_SERVERS=1
else
  JUDGE_TP_SIZE="${CWAD_JUDGE_TP}"
  [[ ${TP_SIZE} -ge ${CWAD_JUDGE_TP} ]] || die \
    "the canonical judge requires at least ${CWAD_JUDGE_TP} GPUs"
  JUDGE_GPU_IDS="$(IFS=,; printf '%s' "${GPU_ARRAY[*]:0:${CWAD_JUDGE_TP}}")"
  CONCURRENT_SERVERS=0
fi

start_server \
  "${MODEL_PATH}" \
  "${TARGET_SERVED_NAME}" \
  "${TARGET_PORT}" \
  "${OUTPUT_DIR}/logs/target_vllm.log" \
  "${GPU_IDS}" \
  "${TP_SIZE}"
TARGET_SERVER_PID="${LAST_SERVER_PID}"

if [[ ${CONCURRENT_SERVERS} -eq 1 ]]; then
  start_server \
    "${JUDGE_MODEL_PATH}" \
    "${JUDGE_SERVED_NAME}" \
    "${JUDGE_PORT}" \
    "${OUTPUT_DIR}/logs/judge_vllm.log" \
    "${JUDGE_GPU_IDS}" \
    "${JUDGE_TP_SIZE}" \
    "${CWAD_JUDGE_MAX_MODEL_LEN}"
  JUDGE_SERVER_PID="${LAST_SERVER_PID}"
  log "judge server resident on ${JUDGE_GPU_IDS} while the target infers on ${GPU_IDS}"
fi

for benchmark in "${BENCHMARK_ARRAY[@]}"; do
  JSON_NAME="$(benchmark_json_name "${benchmark}")"
  (
    cd "${RUN_DIR}"
    "${ENV_DIR}/bin/python" "${CWAD_PACKAGE_ROOT}/eval/infer.py" \
      --benchmark "${benchmark}" \
      --benchmark_json "${EVAL_DATA_DIR}/${JSON_NAME}" \
      --out_dir model_answer \
      --model_name "${MODEL_TAG}" \
      --seed "${CWAD_SEED}" \
      --api_base "http://127.0.0.1:${TARGET_PORT}/v1" \
      --api_key EMPTY \
      --model_id "${TARGET_SERVED_NAME}" \
      --max_tokens "${CWAD_EVAL_MAX_TOKENS}" \
      --top_p 1.0 \
      --max_retries 3 \
      --parallel_workers "${PARALLEL_WORKERS}" \
      --image_scale_divisor 1 \
      --enable_thinking False
  ) 2>&1 | tee "${OUTPUT_DIR}/logs/infer_${benchmark}.log"
done

if [[ ${CONCURRENT_SERVERS} -eq 0 ]]; then
  # Sequential mode: the target has to release its cards before the judge can
  # take them.
  stop_pid "${TARGET_SERVER_PID}"
  TARGET_SERVER_PID=""
  start_server \
    "${JUDGE_MODEL_PATH}" \
    "${JUDGE_SERVED_NAME}" \
    "${JUDGE_PORT}" \
    "${OUTPUT_DIR}/logs/judge_vllm.log" \
    "${JUDGE_GPU_IDS}" \
    "${JUDGE_TP_SIZE}" \
    "${CWAD_JUDGE_MAX_MODEL_LEN}"
  JUDGE_SERVER_PID="${LAST_SERVER_PID}"
fi

for benchmark in "${BENCHMARK_ARRAY[@]}"; do
  JSON_NAME="$(benchmark_json_name "${benchmark}")"
  (
    cd "${RUN_DIR}"
    "${ENV_DIR}/bin/python" "${CWAD_PACKAGE_ROOT}/eval/judge_qwenlm.py" \
      --benchmark "${benchmark}" \
      --model "${MODEL_TAG}" \
      --api_base "http://127.0.0.1:${JUDGE_PORT}/v1" \
      --api_key EMPTY \
      --judge_model "${JUDGE_SERVED_NAME}" \
      --judge_max_tokens "${CWAD_JUDGE_MAX_TOKENS}" \
      --judge_enable_thinking False
    "${ENV_DIR}/bin/python" "${CWAD_PACKAGE_ROOT}/eval/cal_acc.py" \
      --benchmark "${benchmark}" \
      --judge_json "judge/${benchmark}/${MODEL_TAG}_answer.jsonl" \
      --benchmark_json "${EVAL_DATA_DIR}/${JSON_NAME}"
  ) 2>&1 | tee "${RUN_DIR}/scores/${benchmark}.log"
done
# Releases whichever servers this run left standing: the judge alone in
# sequential mode, both of them when they ran concurrently.
stop_servers

"${ENV_DIR}/bin/python" "${SCRIPT_DIR}/collect_metrics.py" \
  --run-dir "${RUN_DIR}" \
  --model-tag "${MODEL_TAG}" \
  --benchmarks "${BENCHMARKS}" \
  --output "${OUTPUT_DIR}/metrics.json"

log "evaluation complete: ${OUTPUT_DIR}/metrics.json"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage: train.sh [--model-path DIR] [--teacher-model-path DIR] [options]

The training entry point for CWAD and for the single-world methods it is
compared against. It assembles the Hydra override list, verifies the model and
data, and launches the bundled verl runtime; ./run.sh train-cwad and
./run.sh train-cwad-hint are thin wrappers that add the recipe flags.

Two axes are independent and can be combined:
  world structure   --dual-world (CWAD: original image + counterfactual edit)
                    vs --single-world (one image, one distillation)
  teacher privilege --resolution (the default: teacher gets the original image)
                    --teacher-same-image (none at all: plain OPD)
                    --answer-hint (reference solution in the teacher's prompt)

By default a frozen larger teacher from CWAD_TEACHER_MODEL_PATH supervises a
smaller student (Qwen3.5-4B -> Qwen3.5-0.8B in the shipped config). Passing
--teacher-model-path "" selects the EMA self-teacher instead, which is what every
CWAD run in the experiment log used.

Options:
  --model-path DIR        Student model (default: CWAD_STUDENT_MODEL_PATH).
  --teacher-model-path DIR
                          Frozen teacher model. Pass an empty string to fall
                          back to the paper's EMA self-teacher instead.
                          (default: CWAD_TEACHER_MODEL_PATH)
  --gpus N                Number of FSDP ranks to train on (default: CWAD_GPUS).
                          Batch size is derived so every rank gets
                          CWAD_PROMPTS_PER_RANK prompts.
  --data-parquet FILE     Runtime training parquet: one row per sample, with
                          absolute image paths in the world-A and world-B
                          columns. Build one with ./run.sh build-data (see
                          scripts/build_dual_world_data.py --help); any dataset
                          of that shape works. Default: CWAD_DATA_PARQUET.
  --env-dir DIR           Prepared Python 3.12 environment.
  --work-dir DIR          Local runtime data and rollout directory.
  --steps N               Training steps (default: 55).
  --smoke                 Run one step and validate the diagnostics.
  --quick-verify          Sample 32 rows instead of scanning every image.
                          Use only after a full data verification has passed.
  --dual-world            Train CWAD on the counterfactual dual-world dataset
                          instead of single-world training.
  --single-world          Force single world (overrides CWAD_DUAL_WORLD).
  --teacher-same-image    Give the teacher the student's own low-resolution
                          image instead of the teacher-resolution one, i.e.
                          plain OPD with the resolution privilege removed.
  --no-privilege          Ignore the privileged image columns for this run: the
                          teacher copies the student's own image and both sides of
                          world B score the same image. Overrides
                          CWAD_PRIV_TEACHER_A_KEY / CWAD_PRIV_STUDENT_B_KEY,
                          which are set globally in best.env.
  --answer-hint           OPSD-style: give the teacher the same image as the
                          student (no resolution privilege) but append a
                          reference solution to its prompt. The hint is the
                          generated reasoning chain from
                          extra_info.<CWAD_ANSWER_HINT_KEY> when present,
                          otherwise the bare ground-truth answer.
  --answer-hint-key NAME  Which extra_info field holds world A's hint (default:
                          CWAD_ANSWER_HINT_KEY, else "answer_hint"). In dual
                          world runs, world B's hint is read from
                          CWAD_ANSWER_HINT_KEY_B, else "answer_hint_b".
  --prefilter-cache-dir DIR
                          Where the cached overlong-prompt filter lives
                          (default: CWAD_PREFILTER_CACHE_DIR).
  --no-prefilter          Skip the cache and filter inside the trainer instead.
  --cwad-lambda X         Override the CWAD belief-transition weight for a
                          lambda ablation (default: CWAD_LAMBDA).
  --cwad-tau X            Override the CWAD belief-shift temperature, i.e. how
                          peaked softmax(delta/tau) is over the top-k support
                          (default: CWAD_TAU).
  --data-seed N           Set data.seed, which controls the order the training
                          data is shuffled in. Omitted (the default) leaves it
                          null, which is a fixed order shared by every run so
                          far; pass a number to get a different data order.
  --rollout-n N           Override the rollout group size for a controlled
                          comparison against a run with a different n.
  --resume                Continue an interrupted run from the newest checkpoint
                          in <output-dir>/checkpoints. Requires that directory to
                          already exist; the training log is appended to rather
                          than truncated.
  --reset-local-ray       Stop a pre-existing local Ray runtime before launch.

This command never deletes or overwrites an existing output directory.
EOF
}

RUNTIME_ROOT="$(default_runtime_root)"
MODEL_PATH="${CWAD_STUDENT_MODEL_PATH}"
TEACHER_MODEL_PATH="${CWAD_TEACHER_MODEL_PATH}"
OUTPUT_DIR=""
DATA_PARQUET="${CWAD_DATA_PARQUET:-}"
ENV_DIR="${RUNTIME_ROOT}/venv"
WORK_DIR=""
TOTAL_STEPS="${CWAD_TOTAL_STEPS}"
GPUS="${CWAD_GPUS}"
SMOKE=0
QUICK_VERIFY=0
RESET_LOCAL_RAY=0
DUAL_WORLD="${CWAD_DUAL_WORLD}"
PREFILTER_CACHE_DIR="${CWAD_PREFILTER_CACHE_DIR:-${RUNTIME_ROOT}/prefilter_cache}"
PREFILTER=1
ROLLOUT_N_OVERRIDE=""
CWAD_LAMBDA="${CWAD_LAMBDA}"
CWAD_TAU="${CWAD_TAU}"
# Empty means "pass nothing", which leaves data.seed at its null default. That
# default is NOT arbitrary: an unseeded torch.Generator() starts from a fixed
# seed, so a run with data.seed unset reproduces the same data order as every
# other run that left it unset -- which is what every experiment so far has done.
DATA_SEED=""
RESUME=0
TEACHER_SAME_IMAGE=0
ANSWER_HINT=0
NO_PRIVILEGE=0
ANSWER_HINT_KEY="${CWAD_ANSWER_HINT_KEY:-answer_hint}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH=$2; shift 2 ;;
    --teacher-model-path) TEACHER_MODEL_PATH=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --data-parquet) DATA_PARQUET=$2; shift 2 ;;
    --env-dir) ENV_DIR=$2; shift 2 ;;
    --work-dir) WORK_DIR=$2; shift 2 ;;
    --steps) TOTAL_STEPS=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --smoke) SMOKE=1; TOTAL_STEPS=1; shift ;;
    --quick-verify) QUICK_VERIFY=1; shift ;;
    --dual-world) DUAL_WORLD=1; shift ;;
    --single-world) DUAL_WORLD=0; shift ;;
    --teacher-same-image) TEACHER_SAME_IMAGE=1; shift ;;
    --answer-hint) ANSWER_HINT=1; shift ;;
    --no-privilege) NO_PRIVILEGE=1; shift ;;
    --answer-hint-key) ANSWER_HINT_KEY=$2; shift 2 ;;
    --rollout-n) ROLLOUT_N_OVERRIDE=$2; shift 2 ;;
    --cwad-lambda) CWAD_LAMBDA=$2; shift 2 ;;
    --cwad-tau) CWAD_TAU=$2; shift 2 ;;
    --data-seed) DATA_SEED=$2; shift 2 ;;
    --prefilter-cache-dir) PREFILTER_CACHE_DIR=$2; shift 2 ;;
    --no-prefilter) PREFILTER=0; shift ;;
    --resume) RESUME=1; shift ;;
    --reset-local-ray) RESET_LOCAL_RAY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown train argument: $1" ;;
  esac
done

[[ -n "${OUTPUT_DIR}" ]] || die "--output-dir is required"
[[ -n "${DATA_PARQUET}" ]] || die \
  "--data-parquet is required (or set CWAD_DATA_PARQUET in config/best.env); build one with ./run.sh build-data"
[[ "${TOTAL_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "--steps must be a positive integer"
[[ "${GPUS}" =~ ^[1-9][0-9]*$ ]] || die "--gpus must be a positive integer"

# The distillation loss mixes student and teacher logits, so both sides must
# share a vocabulary. Guard before spending time on model loading.
CROSS_MODEL=0
if [[ -n "${TEACHER_MODEL_PATH}" ]]; then
  CROSS_MODEL=1
fi

if [[ ${DUAL_WORLD} -eq 1 ]]; then
  PROMPTS_PER_RANK="${CWAD_DUAL_WORLD_PROMPTS_PER_RANK}"
  ROLLOUT_N="${CWAD_DUAL_WORLD_ROLLOUT_N}"
  MAX_PROMPT_LENGTH="${CWAD_DUAL_WORLD_MAX_PROMPT_LENGTH}"
  MAX_RESPONSE_LENGTH="${CWAD_DUAL_WORLD_MAX_RESPONSE_LENGTH}"
  ROLLOUT_TEMPERATURE="${CWAD_DUAL_WORLD_ROLLOUT_TEMPERATURE}"
  ROLLOUT_GPU_MEMORY_UTILIZATION="${CWAD_DUAL_WORLD_ROLLOUT_GPU_MEMORY_UTILIZATION}"
else
  PROMPTS_PER_RANK="${CWAD_PROMPTS_PER_RANK}"
  ROLLOUT_N="${CWAD_CROSS_ROLLOUT_N}"
  MAX_PROMPT_LENGTH="${CWAD_CROSS_MAX_PROMPT_LENGTH}"
  MAX_RESPONSE_LENGTH="${CWAD_CROSS_MAX_RESPONSE_LENGTH}"
  ROLLOUT_TEMPERATURE="${CWAD_CROSS_ROLLOUT_TEMPERATURE}"
  ROLLOUT_GPU_MEMORY_UTILIZATION="${CWAD_CROSS_ROLLOUT_GPU_MEMORY_UTILIZATION}"
fi
[[ -n "${ROLLOUT_N_OVERRIDE}" ]] && ROLLOUT_N="${ROLLOUT_N_OVERRIDE}"
TRAIN_BATCH_SIZE=$(( GPUS * PROMPTS_PER_RANK ))
PPO_MINI_BATCH_SIZE="${TRAIN_BATCH_SIZE}"
PPO_MAX_TOKEN_LEN_PER_GPU=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))
# The rollout engine's context window has to cover prompt + response or vLLM
# rejects every request, but the actor's micro-batch budget is independent of
# it: `use_dynamic_bsz` only uses this to decide how many sequences to pack,
# and it sets the forward activation peak. Dual-world runs three forwards per
# step (student A, student B, teacher), so that peak is what OOMs first.
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${CWAD_ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}"
# FSDP splits every mini-batch evenly across ranks.
[[ $(( PPO_MINI_BATCH_SIZE * ROLLOUT_N % GPUS )) -eq 0 ]] || die \
  "ppo_mini_batch_size * rollout.n (${PPO_MINI_BATCH_SIZE} * ${ROLLOUT_N}) must be divisible by --gpus (${GPUS})"

# Save on this interval as well as on the final step, so a late crash still
# leaves a usable checkpoint. 0 keeps the reference behaviour (final step only).
SAVE_FREQ="${CWAD_CHECKPOINT_EVERY:-0}"
[[ "${SAVE_FREQ}" -gt 0 ]] || SAVE_FREQ="${TOTAL_STEPS}"

MODEL_PATH="$(absolute_path "${MODEL_PATH}")"
OUTPUT_DIR="$(absolute_path "${OUTPUT_DIR}")"
DATA_PARQUET="$(absolute_path "${DATA_PARQUET}")"
if [[ -n "${TEACHER_MODEL_PATH}" ]]; then
  TEACHER_MODEL_PATH="$(absolute_path "${TEACHER_MODEL_PATH}")"
fi
ENV_DIR="$(absolute_path "${ENV_DIR}")"
if [[ -z "${WORK_DIR}" ]]; then
  RUN_TAG="$(basename "${OUTPUT_DIR}")"
  WORK_DIR="${TMPDIR:-/tmp}/cwad-${RUN_TAG}"
fi
WORK_DIR="$(absolute_path "${WORK_DIR}")"

require_dir "${MODEL_PATH}"
require_file "${DATA_PARQUET}"
if [[ ${CROSS_MODEL} -eq 1 ]]; then
  require_dir "${TEACHER_MODEL_PATH}"
  require_file "${TEACHER_MODEL_PATH}/config.json"
  require_file "${MODEL_PATH}/config.json"
fi
if [[ ${RESUME} -eq 1 ]]; then
  # Resume needs the checkpoint tracker `resume_mode=auto` reads to find where
  # to pick up. A plain launch still refuses to touch a populated directory.
  require_dir "${OUTPUT_DIR}"
  require_file "${OUTPUT_DIR}/checkpoints/latest_checkpointed_iteration.txt"
else
  # A fresh run must not land on top of existing results. "Empty" here means no
  # previous run's data, not literally nothing: train_supervisor.sh writes its
  # log to logs/ and ckpt_sentinel.sh creates checkpoints/ and ckpt_archive/
  # before the first launch, so those three may exist as long as they hold
  # nothing. A populated checkpoints/ means this is not a fresh run at all.
  leftovers="$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 \
    ! -name logs ! -name checkpoints ! -name ckpt_archive -print 2>/dev/null || true)"
  [[ -z "${leftovers}" ]] || die \
    "output directory is not empty: ${OUTPUT_DIR} (contains: $(printf '%s ' ${leftovers}))"
  if [[ -d "${OUTPUT_DIR}/checkpoints" ]]; then
    populated="$(find "${OUTPUT_DIR}/checkpoints" -mindepth 1 -print -quit 2>/dev/null || true)"
    [[ -z "${populated}" ]] || die \
      "output directory already holds checkpoints: ${OUTPUT_DIR} (found ${populated})"
  fi
fi

require_file "${ENV_DIR}/bin/python"

mkdir -p "${OUTPUT_DIR}/logs" "${WORK_DIR}"
ROLLOUT_DIR="${WORK_DIR}/rollouts"
TRAIN_LOG="${OUTPUT_DIR}/logs/train.log"
VERIFY_LOG="${OUTPUT_DIR}/logs/verify.json"

# The parquet is already a runtime file: absolute image paths, one row per
# sample. --data-root only matters for a dataset that kept relative paths.
RUNTIME_PARQUET="${DATA_PARQUET}"
DATA_DIR="$(dirname "${RUNTIME_PARQUET}")"

# The trainer's own overlong-prompt filter re-decodes every image on every launch
# (one CPU pass over the whole dataset). Cache that verdict and reuse it: the
# surviving rows are written to a parquet the trainer can read directly with
# filter_overlong_prompts=False.
# A resumed run writes into the same log file; truncating it there would throw
# away the record of every step before the crash.
TEE_ARGS=()
[[ ${RESUME} -eq 1 ]] && TEE_ARGS=(-a)

# Dual-world is allocator-fragmentation bound: it OOMs on a small request with a
# bone-dry card rather than on a large one. expandable_segments lets the
# allocator reuse holes, which is the fix PyTorch's own OOM message recommends.
# vLLM asserts on it in this build, so it is opt-in rather than always on.
if [[ -n "${CWAD_ALLOC_CONF:-}" ]]; then
  export PYTORCH_ALLOC_CONF="${CWAD_ALLOC_CONF}"
  log "allocator: PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}"
fi

FILTER_OVERLONG=True
if [[ ${PREFILTER} -eq 1 ]]; then
  PREFILTER_CACHE_DIR="$(absolute_path "${PREFILTER_CACHE_DIR}")"
  # The helper imports verl, and the trainer's own PYTHONPATH export happens
  # further down, so set it for this call explicitly.
  # RLHFDataset prints its own "dataset len: ..." line to stdout, so take only
  # the helper's last line (the parquet path) and ignore that noise.
  #
  # The cache key has to identify the ROW SET, not just the file name. Every
  # build is called dual_world.parquet, so keying on the basename alone made a
  # QC-filtered rebuild compute the exact key of the full build and silently
  # reuse its cache -- training on the wrong data with no error anywhere. Hash
  # the bytes: this parquet is written once and never rewritten, so its bytes are
  # a stable identity of the rows it holds.
  PREFILTER_CACHE_KEY="$(basename "${RUNTIME_PARQUET}"):$(sha256sum "${RUNTIME_PARQUET}" | cut -c1-16)"
  log "prefilter cache key: ${PREFILTER_CACHE_KEY}"

  RUNTIME_PARQUET="$(PYTHONPATH="${CWAD_PACKAGE_ROOT}:${PYTHONPATH:-}" \
    "${ENV_DIR}/bin/python" "${SCRIPT_DIR}/prefilter_prompts.py" \
    --parquet "${RUNTIME_PARQUET}" \
    --model-path "${MODEL_PATH}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --custom-chat-template-file "${CWAD_PACKAGE_ROOT}/chat_templates/perception_chat_template_qwen35.jinja" \
    --cache-dir "${PREFILTER_CACHE_DIR}" \
    --cache-key "${PREFILTER_CACHE_KEY}" | tail -n 1)"
  require_file "${RUNTIME_PARQUET}"
  FILTER_OVERLONG=False
fi

if [[ ${CROSS_MODEL} -eq 1 ]]; then
  EXPECTED_HIDDEN_SIZE="${CWAD_STUDENT_MODEL_HIDDEN_SIZE}"
else
  EXPECTED_HIDDEN_SIZE="${CWAD_MODEL_HIDDEN_SIZE}"
fi

# Pin the visible devices so --gpus, the runtime check, and the FSDP world size
# all agree on this box's GPU count.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $(( GPUS - 1 )))"
fi
export CUDA_VISIBLE_DEVICES

verify_args=(
  --model-path "${MODEL_PATH}"
  --expected-hidden-size "${EXPECTED_HIDDEN_SIZE}"
  --data-parquet "${RUNTIME_PARQUET}"
  --data-root "${DATA_DIR}"
  --check-source
  --check-runtime
  --require-gpus "${GPUS}"
  --output-json "${VERIFY_LOG}"
)
[[ ${QUICK_VERIFY} -eq 1 ]] && verify_args+=(--quick)
[[ ${DUAL_WORLD} -eq 1 ]] && verify_args+=(--skip-resolution-ratio)

# verify.py's row / MCQ / OpenQA counts are opt-in: they come from
# CWAD_EXPECTED_* in config/best.env and are unchecked when empty. Pin them to
# catch a build that silently lost rows, leave them empty to train on a
# deliberately filtered subset.
"${ENV_DIR}/bin/python" "${SCRIPT_DIR}/verify.py" "${verify_args[@]}"

if [[ ${CROSS_MODEL} -eq 1 ]]; then
  # The teacher is scored against the student's vocabulary, so its embedding
  # table must match the student's exactly.
  "${ENV_DIR}/bin/python" - "${MODEL_PATH}/config.json" "${TEACHER_MODEL_PATH}/config.json" <<'PY'
import json
import sys


def vocab_size(path):
    config = json.load(open(path, encoding="utf-8"))
    text_config = config.get("text_config") or {}
    value = config.get("vocab_size") or text_config.get("vocab_size")
    if not isinstance(value, int):
        raise SystemExit(f"cannot find vocab_size in {path}")
    return value


student, teacher = vocab_size(sys.argv[1]), vocab_size(sys.argv[2])
if student != teacher:
    raise SystemExit(f"student/teacher vocabulary mismatch: {student} != {teacher}")
print(f"student/teacher vocabulary: {student}")
PY
  "${ENV_DIR}/bin/python" "${SCRIPT_DIR}/verify.py" \
    --model-path "${TEACHER_MODEL_PATH}" \
    --expected-hidden-size "${CWAD_TEACHER_MODEL_HIDDEN_SIZE}" \
    --output-json "${OUTPUT_DIR}/logs/verify_teacher.json"
fi

if [[ ${RESET_LOCAL_RAY} -eq 1 ]]; then
  log "stopping the local Ray runtime by explicit request"
  "${ENV_DIR}/bin/ray" stop --force || true
  unset RAY_ADDRESS
fi

# The runtime environment may be a venv, a conda prefix, or a venv whose
# `activate` script was never generated. Putting its bin directory first is
# what the launcher actually needs, and it always exists.
export VIRTUAL_ENV="${ENV_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export_c_include_path "${ENV_DIR}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export PYTHONPATH="${CWAD_PACKAGE_ROOT}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export CWAD_LAUNCHER_CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
export CWAD_LAUNCHER_ROLLOUT_DIR="${ROLLOUT_DIR}"
unset VLLM_ATTENTION_BACKEND

MODEL_TAG="$(basename "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')"
if [[ ${CROSS_MODEL} -eq 1 ]]; then
  TEACHER_TAG="$(basename "${TEACHER_MODEL_PATH}" | tr '[:upper:]' '[:lower:]')"
  EXPERIMENT_NAME="cwad_${MODEL_TAG}_t${TEACHER_TAG}_${GPUS}gpu_bs${TRAIN_BATCH_SIZE}_$(
    [[ ${SMOKE} -eq 1 ]] && printf 'smoke1' || printf "${TOTAL_STEPS}step"
  )"
else
  EXPERIMENT_NAME="cwad_${MODEL_TAG}_ema_${GPUS}gpu_bs${TRAIN_BATCH_SIZE}_$(
    [[ ${SMOKE} -eq 1 ]] && printf 'smoke1' || printf "${TOTAL_STEPS}step"
  )"
fi
PROJECT_NAME="CWAD"

log "student=${MODEL_PATH}"
log "teacher=$([[ ${CROSS_MODEL} -eq 1 ]] && printf '%s (frozen)' "${TEACHER_MODEL_PATH}" || printf 'EMA self-teacher')"
log "data=${RUNTIME_PARQUET}"
log "output=${OUTPUT_DIR}"
log "gpus=${GPUS} prompts_per_rank=${CWAD_PROMPTS_PER_RANK} batch=${TRAIN_BATCH_SIZE} n=${ROLLOUT_N} lr=${CWAD_LEARNING_RATE} warmup=${CWAD_WARMUP_STEPS}"
log "steps=${TOTAL_STEPS} objective=teacher-selected Top-${CWAD_TOPK} bias-corrected reverse KL"
if [[ ${TEACHER_SAME_IMAGE} -eq 1 ]]; then
  log "student=physical half resolution teacher=the student's own image (no resolution privilege)"
else
  log "student=physical half resolution teacher=original resolution"
fi
if [[ ${DUAL_WORLD} -eq 1 ]]; then
  log "dual-world CWAD: cwad_lambda=${CWAD_LAMBDA} cwad_tau=${CWAD_TAU}"
  if [[ ${NO_PRIVILEGE} -eq 1 ]]; then
    log "resolution privilege: DISABLED (teacher copies the student's image in both worlds)"
  fi
fi

# Passed as an array so the empty case contributes no argument at all: a bare
# `data.seed=` would be a hydration error, and dropping it keeps the null
# default (the fixed order every earlier run used).
SEED_ARGS=()
if [[ -n "${DATA_SEED}" ]]; then
  SEED_ARGS=("data.seed=${DATA_SEED}")
  log "data.seed=${DATA_SEED} (data order will differ from every unseeded run)"
else
  log "data.seed unset (default fixed order, same as every earlier run)"
fi

cd "${CWAD_PACKAGE_ROOT}"
set +e
bash scripts/run_verl.sh \
  "data.train_files=[\"${RUNTIME_PARQUET}\"]" \
  "data.val_files=[]" \
  "data.image_key=images" \
  "${SEED_ARGS[@]}" \
  "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}" \
  "data.max_response_length=${MAX_RESPONSE_LENGTH}" \
  "data.filter_overlong_prompts=${FILTER_OVERLONG}" \
  "data.dataloader_num_workers=2" \
  "actor_rollout_ref.model.path=${MODEL_PATH}" \
  "actor_rollout_ref.model.use_remove_padding=True" \
  "+actor_rollout_ref.model.override_config.attn_implementation=${CWAD_ATTN_IMPLEMENTATION}" \
  "critic.model.path=${MODEL_PATH}" \
  "actor_rollout_ref.actor.optim.lr=${CWAD_LEARNING_RATE}" \
  "actor_rollout_ref.actor.optim.lr_warmup_steps=${CWAD_WARMUP_STEPS}" \
  "actor_rollout_ref.actor.optim.weight_decay=${CWAD_WEIGHT_DECAY:-1e-2}" \
  "actor_rollout_ref.actor.optim.lr_scheduler_type=${CWAD_LR_SCHEDULER:-constant}" \
  "actor_rollout_ref.actor.freeze_vision_tower=${CWAD_FREEZE_VISION_TOWER:-False}" \
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
  "actor_rollout_ref.actor.use_dynamic_bsz=True" \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}" \
  "actor_rollout_ref.actor.policy_loss.loss_mode=cwad" \
  "actor_rollout_ref.actor.self_distillation.full_logit_distillation=True" \
  "actor_rollout_ref.actor.self_distillation.distillation_topk=${CWAD_TOPK}" \
  "actor_rollout_ref.actor.self_distillation.distillation_add_tail=False" \
  "actor_rollout_ref.actor.self_distillation.distillation_objective=cwad_topk_reverse_kl" \
  "actor_rollout_ref.actor.self_distillation.distillation_topk_source=teacher" \
  "actor_rollout_ref.actor.self_distillation.alpha=${CWAD_ALPHA}" \
  "actor_rollout_ref.actor.self_distillation.teacher_model_source=$(
    [[ ${CROSS_MODEL} -eq 1 ]] && printf 'fixed' || printf 'legacy'
  )" \
  "actor_rollout_ref.actor.self_distillation.teacher_model_path=$(
    [[ ${CROSS_MODEL} -eq 1 ]] && printf '%s' "${TEACHER_MODEL_PATH}" || printf 'null'
  )" \
  "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema" \
  "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$(
    [[ ${CROSS_MODEL} -eq 1 ]] && printf '0.0' || printf '%s' "${CWAD_TEACHER_UPDATE_RATE}"
  )" \
  "actor_rollout_ref.actor.self_distillation.teacher_always_on=True" \
  "actor_rollout_ref.actor.self_distillation.teacher_image_key=teacher_images" \
  "actor_rollout_ref.actor.self_distillation.teacher_use_student_image=$(
    [[ ${TEACHER_SAME_IMAGE} -eq 1 ]] && printf 'True' || printf 'False'
  )" \
  "actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=$(
    [[ ${ANSWER_HINT} -eq 1 ]] && printf 'answer_hint' || printf 'null'
  )" \
  "actor_rollout_ref.actor.self_distillation.answer_hint_key=$(
    [[ ${ANSWER_HINT} -eq 1 ]] && printf '%s' "${ANSWER_HINT_KEY}" || printf 'null'
  )" \
  "actor_rollout_ref.actor.self_distillation.answer_hint_key_b=$(
    [[ ${ANSWER_HINT} -eq 1 ]] && printf '%s' "${CWAD_ANSWER_HINT_KEY_B:-answer_hint_b}" || printf 'null'
  )" \
  "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True" \
  "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False" \
  "actor_rollout_ref.actor.self_distillation.is_clip=2.0" \
  "actor_rollout_ref.actor.self_distillation.dual_world=$([[ ${DUAL_WORLD} -eq 1 ]] && printf 'True' || printf 'False')" \
  "actor_rollout_ref.actor.self_distillation.cwad_lambda=${CWAD_LAMBDA}" \
  "actor_rollout_ref.actor.self_distillation.cwad_tau=${CWAD_TAU}" \
  "actor_rollout_ref.actor.self_distillation.privileged_teacher_a_image_key=$(
    [[ ${NO_PRIVILEGE} -eq 1 ]] && printf 'null' || printf '%s' "${CWAD_PRIV_TEACHER_A_KEY:-null}"
  )" \
  "actor_rollout_ref.actor.self_distillation.privileged_student_b_image_key=$(
    [[ ${NO_PRIVILEGE} -eq 1 ]] && printf 'null' || printf '%s' "${CWAD_PRIV_STUDENT_B_KEY:-null}"
  )" \
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
  "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}" \
  "actor_rollout_ref.rollout.top_p=${CWAD_ROLLOUT_TOP_P:-1.0}" \
  "actor_rollout_ref.rollout.top_k=-1" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  "actor_rollout_ref.rollout.enforce_eager=True" \
  "actor_rollout_ref.rollout.max_num_seqs=${CWAD_CROSS_MAX_NUM_SEQS}" \
  "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}" \
  "actor_rollout_ref.rollout.max_model_len=${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  "actor_rollout_ref.rollout.max_num_batched_tokens=${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  "actor_rollout_ref.rollout.agent.num_workers=2" \
  "algorithm.adv_estimator=grpo" \
  "algorithm.norm_adv_by_std_in_grpo=False" \
  "algorithm.use_kl_in_reward=False" \
  "algorithm.rollout_correction.rollout_is=token" \
  "algorithm.rollout_correction.rollout_is_threshold=2.0" \
  "reward_model.enable=False" \
  "reward_model.use_reward_loop=False" \
  "trainer.project_name=${PROJECT_NAME}" \
  "trainer.group_name=${EXPERIMENT_NAME}" \
  "trainer.experiment_name=${EXPERIMENT_NAME}" \
  "trainer.n_gpus_per_node=${GPUS}" \
  "trainer.nnodes=1" \
  "trainer.total_epochs=1" \
  "trainer.total_training_steps=${TOTAL_STEPS}" \
  "trainer.save_freq=${SAVE_FREQ}" \
  "trainer.test_freq=-1" \
  "trainer.max_actor_ckpt_to_keep=1" \
  "trainer.val_before_train=False" \
  "trainer.resume_mode=auto" \
  "trainer.default_local_dir=${OUTPUT_DIR}/checkpoints" \
  "trainer.rollout_data_dir=${ROLLOUT_DIR}" \
  2>&1 | tee "${TEE_ARGS[@]}" "${TRAIN_LOG}"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
[[ ${TRAIN_STATUS} -eq 0 ]] || die "training failed; see ${TRAIN_LOG}"

# The trainer saves on `save_freq` boundaries and on what it considers the last
# step, but `is_last_step` is evaluated before the step counter advances -- so a
# target that is not a multiple of save_freq never gets a directory of its own
# (a 1324-step run with save_freq 50 or 100 stops at 1300). The run still
# completed; fall back to the newest checkpoint that exists rather than failing
# it. The full epoch is trained either way, only the final snapshot is coarser.
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/global_step_${TOTAL_STEPS}"
if [[ ! -d "${CHECKPOINT_DIR}/actor" ]]; then
  latest="$(find "${OUTPUT_DIR}/checkpoints" -maxdepth 1 -type d -name 'global_step_*' \
    -printf '%f\n' 2>/dev/null | sort -t_ -k3 -n | tail -1)"
  [[ -n "${latest}" ]] || die "no checkpoint was written under ${OUTPUT_DIR}/checkpoints"
  CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/${latest}"
  log "WARNING: no checkpoint at the target step ${TOTAL_STEPS} (${TOTAL_STEPS} is not a save boundary); using ${latest}"
fi
require_dir "${CHECKPOINT_DIR}/actor"

if [[ ${SMOKE} -eq 1 ]]; then
  "${ENV_DIR}/bin/python" - "${TRAIN_LOG}" "${DUAL_WORLD}" <<'PY'
import math
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
dual_world = sys.argv[2] == "1"

required = {
    # OPSD answer-hint reports the same idea under its own name: every sample
    # carried a hint. Accept either so the smoke check works for both modes.
    "teacher_image_swap_fraction": r"(?:teacher_image_swap_fraction|answer_hint_fraction):([0-9.eE+-]+)",
    # Both objectives report a total distillation loss under this name.
    "distillation_loss": r"raw_distillation_token_mean:([0-9.eE+-]+)",
}
if dual_world:
    # CWAD must actually exercise both terms of the objective.
    required.update(
        {
            "cwad_world": r"cwad_world_token_mean:([0-9.eE+-]+)",
            "cwad_trans": r"cwad_trans_token_mean:([0-9.eE+-]+)",
        }
    )
else:
    required.update(
        {
            "teacher_topk_mass": r"teacher_topk_mass_mean:([0-9.eE+-]+)",
            "bias_correction": r"cwad_bias_correction_mean:([0-9.eE+-]+)",
        }
    )

values = {}
for name, pattern in required.items():
    matches = re.findall(pattern, text)
    if not matches:
        raise SystemExit(f"smoke metric missing: {name}")
    values[name] = float(matches[-1])
    if not math.isfinite(values[name]):
        raise SystemExit(f"smoke metric is non-finite: {name}={values[name]}")
if abs(values["teacher_image_swap_fraction"] - 1.0) > 1e-6:
    raise SystemExit(f"teacher image swap fraction is not 1: {values}")
if values["distillation_loss"] <= 0:
    raise SystemExit(f"distillation loss must be non-zero and positive: {values}")
print(values)
PY
fi

if [[ ${CROSS_MODEL} -eq 1 ]]; then
  TEACHER_SUMMARY="\"$(basename "${TEACHER_MODEL_PATH}")\""
else
  TEACHER_SUMMARY="null"
fi

cat >"${OUTPUT_DIR}/run_summary.json" <<EOF
{
  "student": "$(basename "${MODEL_PATH}")",
  "teacher": ${TEACHER_SUMMARY},
  "data_parquet": "$(basename "${DATA_PARQUET}")",
  "source_manifest_sha256": "${CWAD_SOURCE_MANIFEST_SHA256}",
  "checkpoint": "checkpoints/global_step_${TOTAL_STEPS}",
  "steps": ${TOTAL_STEPS},
  "gpus": ${GPUS},
  "train_batch_size": ${TRAIN_BATCH_SIZE},
  "rollout_n": ${ROLLOUT_N},
  "smoke": $([[ ${SMOKE} -eq 1 ]] && printf true || printf false)
}
EOF

log "training complete: ${CHECKPOINT_DIR}"

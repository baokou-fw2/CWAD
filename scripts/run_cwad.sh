#!/usr/bin/env bash
# CWAD recipe 1 — dual-world distillation, no answer hint.
#
#   ./run.sh train-cwad --data-parquet <build/dual_world.parquet> \
#                       --output-dir <new directory> [--gpus N] [--steps N] [extra train.sh flags]
#
# Defaults to the EMA self-teacher (an empty --teacher-model-path), which is the
# CWAD row of the paper's table. To distil from a frozen larger
# teacher instead, pass --teacher-model-path <dir>: the two worlds then use that
# model for both worlds' teacher distribution, and the student/teacher hidden
# sizes must share a vocabulary (train.sh checks this and refuses otherwise).
# That is the CWAD (OPD) row.
#
# The strict privileged layout (teacher sees world A at original resolution,
# student sees world B at half resolution) is selected by the two column names in
# config/best.env, CWAD_PRIV_TEACHER_A_KEY and CWAD_PRIV_STUDENT_B_KEY, and
# only applies to a parquet built with build_cwad_data.sh --privileged-both-worlds.
# Pass --no-privilege to a run to ignore both columns.
#
# Everything is forwarded to scripts/train.sh, which parses last-wins, so
# overriding any flag below is just a matter of passing it again.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# =============================================================================
# CONFIGURATION — the reported Qwen3.5-4B settings
#
# One row of the paper's "Key hyperparameters used for Qwen3.5-4B" table per
# line, so the run that produced the reported numbers is readable here instead
# of being spread across a config file. Applied before scripts/common.sh loads
# config/best.env, and only for names nobody has set yet, which fixes the
# precedence: your shell environment > this recipe > config/best.env. Paths stay
# in config/best.env because they belong to the machine, not to the recipe.
# =============================================================================
RECIPE=(
  # --- Optimization & training ---
  CWAD_LEARNING_RATE=2e-6
  CWAD_WEIGHT_DECAY=1e-2                    # AdamW with weight decay 1e-2
  CWAD_LR_SCHEDULER=constant                # no decay curve
  CWAD_WARMUP_STEPS=10
  CWAD_FREEZE_VISION_TOWER=False            # the vision encoder trains
  # Global batch size 48. The paper fixes the product, not the split, so this is
  # 8 cards x 6 prompts; re-split it with --gpus and the two prompts-per-rank
  # names (dual world scores every rollout twice, so its count is the one that
  # sets the batch here).
  CWAD_GPUS=8
  CWAD_PROMPTS_PER_RANK=6
  CWAD_DUAL_WORLD_PROMPTS_PER_RANK=6
  # One epoch (trainer.total_epochs=1, set by scripts/train.sh). The step budget
  # is what actually ends a run, so it stays yours: --steps or CWAD_TOTAL_STEPS.

  # --- On-policy rollout ---
  CWAD_DUAL_WORLD_MAX_PROMPT_LENGTH=8192
  CWAD_DUAL_WORLD_MAX_RESPONSE_LENGTH=1024
  CWAD_DUAL_WORLD_ROLLOUT_TEMPERATURE=2.0
  CWAD_CROSS_MAX_PROMPT_LENGTH=8192
  CWAD_CROSS_MAX_RESPONSE_LENGTH=1024
  CWAD_CROSS_ROLLOUT_TEMPERATURE=2.0
  CWAD_ROLLOUT_TOP_P=1.0
  # The table does not fix the rollout group size or the vLLM memory reservation,
  # so those keep their config/best.env values and remain overridable with
  # --rollout-n and CWAD_*_ROLLOUT_GPU_MEMORY_UTILIZATION.

  # --- EMA teacher (OPSD) ---
  # rate 0.05 is the whole entry: decay is 1 - 0.05 = 0.95, the update runs every
  # optimizer step, the teacher starts as a frozen copy of the student and its
  # buffer is never refreshed. Passing --teacher-model-path <dir> makes train.sh
  # force this to 0.0 -- a frozen teacher -- which is the CWAD (OPD) row.
  CWAD_TEACHER_UPDATE_RATE=0.05

  # --- CWAD ---
  CWAD_LAMBDA=1.0
  # tau, the softmax temperature over the centered cross-world log-prob
  # difference, is not in that table: it comes from config/best.env and is
  # ablated with --cwad-tau.
)
for assignment in "${RECIPE[@]}"; do
  name="${assignment%%=*}"
  [[ -n "${!name+set}" ]] && continue
  export "${assignment?}"
done

# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

passed_parquet=0
for arg in "$@"; do
  [[ "${arg}" == --data-parquet* ]] && passed_parquet=1
done

# A missing parquet is only fatal when the caller did not pass one: train.sh
# would otherwise die on its require_file check deep into startup.
if [[ ${passed_parquet} -eq 0 && ! -f "${CWAD_DATA_PARQUET:-}" ]]; then
  die "no training parquet. CWAD_DATA_PARQUET is empty in config/best.env and
no --data-parquet was passed. Build one from your dataset with
./run.sh build-data (see scripts/build_dual_world_data.py --help)."
fi

log "recipe: batch=${CWAD_GPUS}x${CWAD_DUAL_WORLD_PROMPTS_PER_RANK}=$(( CWAD_GPUS * CWAD_DUAL_WORLD_PROMPTS_PER_RANK )) lr=${CWAD_LEARNING_RATE} wd=${CWAD_WEIGHT_DECAY} schedule=${CWAD_LR_SCHEDULER} warmup=${CWAD_WARMUP_STEPS} freeze_vision=${CWAD_FREEZE_VISION_TOWER} prompt=${CWAD_DUAL_WORLD_MAX_PROMPT_LENGTH}/${CWAD_DUAL_WORLD_MAX_RESPONSE_LENGTH} rollout_T=${CWAD_DUAL_WORLD_ROLLOUT_TEMPERATURE} top_p=${CWAD_ROLLOUT_TOP_P} ema_rate=${CWAD_TEACHER_UPDATE_RATE} lambda=${CWAD_LAMBDA} tau=${CWAD_TAU}"

exec bash "${SCRIPT_DIR}/train.sh" \
  --dual-world \
  --teacher-model-path "" \
  "$@"

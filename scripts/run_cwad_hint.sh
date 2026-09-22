#!/usr/bin/env bash
# CWAD recipe 2 — answer-hint privilege with an EMA self-teacher.
#
#   ./run.sh train-cwad-hint --data-parquet <hint parquet> \
#                            --output-dir <new directory> [--gpus N] [--steps N]
#
# The teacher gets the same image as the student (no resolution gap) plus a
# reference solution appended to its prompt. That hint has to be in the data
# already, so this recipe needs a parquet built by the three steps in the README
# (generate_answer_hints.py -> merge_answer_hints.py -> build_hint_parquet.py),
# which writes the chain into extra_info.answer_hint of every row.
#
# --single-world is the arm the log reports as std-hint. Add --dual-world plus a
# --data-parquet built with --privileged-both-worlds for the cwad-hint arm (then
# world B's hint is read from extra_info.answer_hint_b).
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
  # 8 cards x 6 prompts; re-split it with --gpus and CWAD_PROMPTS_PER_RANK.
  CWAD_GPUS=8
  CWAD_PROMPTS_PER_RANK=6
  CWAD_DUAL_WORLD_PROMPTS_PER_RANK=6
  # One epoch (trainer.total_epochs=1, set by scripts/train.sh). The step budget
  # is what actually ends a run, so it stays yours: --steps or CWAD_TOTAL_STEPS.

  # --- On-policy rollout ---
  CWAD_CROSS_MAX_PROMPT_LENGTH=8192
  CWAD_CROSS_MAX_RESPONSE_LENGTH=1024
  CWAD_DUAL_WORLD_MAX_PROMPT_LENGTH=8192
  CWAD_DUAL_WORLD_MAX_RESPONSE_LENGTH=1024
  CWAD_CROSS_ROLLOUT_TEMPERATURE=2.0
  CWAD_DUAL_WORLD_ROLLOUT_TEMPERATURE=2.0
  CWAD_ROLLOUT_TOP_P=1.0
  # The table does not fix the rollout group size or the vLLM memory reservation,
  # so those keep their config/best.env values and remain overridable with
  # --rollout-n and CWAD_*_ROLLOUT_GPU_MEMORY_UTILIZATION.

  # --- EMA teacher (OPSD) ---
  # rate 0.05 is the whole entry: decay is 1 - 0.05 = 0.95, the update runs every
  # optimizer step, the teacher starts as a frozen copy of the student and its
  # buffer is never refreshed. scripts/train.sh forces the rate to 0.0 -- a frozen
  # teacher -- whenever --teacher-model-path names another model, which is the OPD
  # arm of the comparison.
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

log "recipe: batch=${CWAD_GPUS}x${CWAD_PROMPTS_PER_RANK}=$(( CWAD_GPUS * CWAD_PROMPTS_PER_RANK )) lr=${CWAD_LEARNING_RATE} wd=${CWAD_WEIGHT_DECAY} schedule=${CWAD_LR_SCHEDULER} warmup=${CWAD_WARMUP_STEPS} freeze_vision=${CWAD_FREEZE_VISION_TOWER} prompt=${CWAD_DUAL_WORLD_MAX_PROMPT_LENGTH}/${CWAD_DUAL_WORLD_MAX_RESPONSE_LENGTH} rollout_T=${CWAD_DUAL_WORLD_ROLLOUT_TEMPERATURE} top_p=${CWAD_ROLLOUT_TOP_P} ema_rate=${CWAD_TEACHER_UPDATE_RATE} lambda=${CWAD_LAMBDA} tau=${CWAD_TAU}"

exec bash "${SCRIPT_DIR}/train.sh" \
  --single-world \
  --answer-hint \
  --answer-hint-key "${CWAD_ANSWER_HINT_KEY:-answer_hint}" \
  --teacher-model-path "" \
  "$@"

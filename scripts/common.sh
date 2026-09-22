#!/usr/bin/env bash
set -euo pipefail

CWAD_PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# config/best.env holds the defaults for every CWAD_* name. It is loaded
# one-name-at-a-time rather than sourced, so a value that is already in the
# environment survives: a recipe launcher (run_cwad.sh, run_cwad_hint.sh) states
# its own concrete hyperparameters and then execs train.sh, which loads this file
# again. Sourcing unconditionally would silently undo the recipe. Precedence is
# therefore shell environment > recipe launcher > config/best.env.
load_config_defaults() {
  local file=$1 line name
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ "${line}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
    name="${line%%=*}"
    [[ -n "${!name+set}" ]] && continue
    printf -v "${name}" '%s' "${line#*=}"
  done < "${file}"
}
# shellcheck source=../config/best.env
load_config_defaults "${CWAD_PACKAGE_ROOT}/config/best.env"

log() {
  printf '[cwad] %s\n' "$*"
}

# Triton and vLLM JIT-compile small CUDA helpers with gcc. Those helpers include
# Python.h, which a bare python3.12 interpreter (unlike python3.12-dev) does not
# install system-wide, so point gcc at the environment's own headers. Every
# entry point that may import triton or start vLLM has to call this.
export_c_include_path() {
  local env_dir=$1
  if [[ -d "${env_dir}/include/python3.12" ]]; then
    export C_INCLUDE_PATH="${env_dir}/include/python3.12${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
  fi
}

die() {
  printf '[cwad] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "required directory not found: $1"
}

absolute_path() {
  realpath -m "$1"
}

require_empty_output() {
  local output=$1
  if [[ -e "${output}" ]] && [[ -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    die "output directory is not empty: ${output}"
  fi
}

default_runtime_root() {
  printf '%s\n' "${CWAD_RUNTIME_ROOT:-${CWAD_PACKAGE_ROOT}/.runtime}"
}

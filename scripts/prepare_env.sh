#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage: prepare_env.sh [options]

Options:
  --env-dir DIR          Virtual environment destination.
  --python PATH          Python 3.12 executable.
  --skip-kernels         Do not install flash-attn and causal-conv1d.

Put --env-dir on a LOCAL disk. Over NFS the ~95k small files page fault at
roughly 100 KB/s and a bare `import torch` takes over seven minutes, which looks
like a hang; a local venv starts in seconds. The reported runs used
/tmp/cwad-venv, which is why --env-dir is passed explicitly everywhere.

Optional environment variables:
  FLASH_ATTN_WHEEL       Public or local wheel path/URL.
  CAUSAL_CONV_WHEEL      Public or local wheel path/URL.
EOF
}

RUNTIME_ROOT="$(default_runtime_root)"
ENV_DIR="${RUNTIME_ROOT}/venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
SKIP_KERNELS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-dir) ENV_DIR=$2; shift 2 ;;
    --python) PYTHON_BIN=$2; shift 2 ;;
    --skip-kernels) SKIP_KERNELS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown prepare-env argument: $1" ;;
  esac
done

ENV_DIR="$(absolute_path "${ENV_DIR}")"

require_command "${PYTHON_BIN}"
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  mkdir -p "$(dirname "${ENV_DIR}")"
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
fi

"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
# --no-deps is required, not an optimisation: the lock is a `pip freeze`, and it
# pins numpy==1.26.4 while cupy-cuda12x declares numpy>=2. Letting pip resolve
# that conflict ends in ResolutionImpossible. A freeze already lists every
# transitive dependency, so nothing is left uninstalled.
"${ENV_DIR}/bin/python" -m pip install --no-deps \
  --requirement "${CWAD_PACKAGE_ROOT}/environment/requirements.lock.txt"
"${ENV_DIR}/bin/python" -m pip install --no-deps --editable "${CWAD_PACKAGE_ROOT}"

if [[ ${SKIP_KERNELS} -eq 0 ]]; then
  if ! "${ENV_DIR}/bin/python" -c 'import flash_attn' >/dev/null 2>&1; then
    if [[ -n "${FLASH_ATTN_WHEEL:-}" ]]; then
      "${ENV_DIR}/bin/python" -m pip install "${FLASH_ATTN_WHEEL}"
    else
      "${ENV_DIR}/bin/python" -m pip install flash-attn==2.8.3 --no-build-isolation
    fi
  fi
  if ! "${ENV_DIR}/bin/python" -c 'import causal_conv1d' >/dev/null 2>&1; then
    if [[ -n "${CAUSAL_CONV_WHEEL:-}" ]]; then
      "${ENV_DIR}/bin/python" -m pip install "${CAUSAL_CONV_WHEEL}"
    else
      "${ENV_DIR}/bin/python" -m pip install causal-conv1d==1.6.1 --no-build-isolation
    fi
  fi
fi

"${ENV_DIR}/bin/python" -m pip check
"${ENV_DIR}/bin/python" - "${CWAD_PACKAGE_ROOT}/environment/versions.json" <<'PY'
from importlib.metadata import version
import json
import sys

expected = json.load(open(sys.argv[1], encoding="utf-8"))
distributions = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "vllm": "vllm",
    "ray": "ray",
    "qwen_vl_utils": "qwen-vl-utils",
}
actual = {key: version(name) for key, name in distributions.items()}
for key in ("torch", "torchvision"):
    actual[key] = actual[key].split("+", 1)[0]
for key in distributions:
    if actual[key] != expected[key]:
        raise SystemExit(
            f"{key} version mismatch: expected {expected[key]}, got {actual[key]}"
        )
if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"Python version mismatch: expected 3.12, got {sys.version_info.major}.{sys.version_info.minor}"
    )
print(json.dumps(actual, indent=2, sort_keys=True))
PY

log "environment ready: ${ENV_DIR}"

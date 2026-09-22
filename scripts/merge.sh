#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage: merge.sh --checkpoint-dir DIR --output-dir DIR [options]

Options:
  --env-dir DIR           Prepared reproduction environment.
  --expected-hidden-size N  Hidden size the merged model must have. By default
                            this is read from the checkpoint's own actor config,
                            so cross-model students validate without extra flags.

The FSDP actor shards remain untouched. Merged Hugging Face weights are written
to a separate, initially empty output directory.
EOF
}

RUNTIME_ROOT="$(default_runtime_root)"
CHECKPOINT_DIR=""
OUTPUT_DIR=""
ENV_DIR="${RUNTIME_ROOT}/venv"
EXPECTED_HIDDEN_SIZE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir) CHECKPOINT_DIR=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --env-dir) ENV_DIR=$2; shift 2 ;;
    --expected-hidden-size) EXPECTED_HIDDEN_SIZE=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown merge argument: $1" ;;
  esac
done

[[ -n "${CHECKPOINT_DIR}" ]] || die "--checkpoint-dir is required"
[[ -n "${OUTPUT_DIR}" ]] || die "--output-dir is required"
CHECKPOINT_DIR="$(absolute_path "${CHECKPOINT_DIR}")"
OUTPUT_DIR="$(absolute_path "${OUTPUT_DIR}")"
ENV_DIR="$(absolute_path "${ENV_DIR}")"

require_dir "${CHECKPOINT_DIR}/actor"
require_empty_output "${OUTPUT_DIR}"
require_file "${ENV_DIR}/bin/python"

# The actor may be a cross-model student of any size, so take the expected
# hidden size from the checkpoint rather than assuming the paper's 9B.
if [[ -z "${EXPECTED_HIDDEN_SIZE}" ]]; then
  EXPECTED_HIDDEN_SIZE="$("${ENV_DIR}/bin/python" - "${CHECKPOINT_DIR}/actor/huggingface" <<'PY'
import json
import pathlib
import sys

config = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text(encoding="utf-8"))
text_config = config.get("text_config") or {}
size = config.get("hidden_size") or text_config.get("hidden_size")
if not isinstance(size, int):
    raise SystemExit("cannot determine hidden_size from the actor checkpoint config")
print(size)
PY
)"
fi
log "expected hidden size: ${EXPECTED_HIDDEN_SIZE}"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${CWAD_PACKAGE_ROOT}:${PYTHONPATH:-}"
"${ENV_DIR}/bin/python" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${CHECKPOINT_DIR}/actor" \
  --target_dir "${OUTPUT_DIR}"

"${ENV_DIR}/bin/python" "${SCRIPT_DIR}/verify.py" \
  --model-path "${OUTPUT_DIR}" \
  --expected-hidden-size "${EXPECTED_HIDDEN_SIZE}"
"${ENV_DIR}/bin/python" - "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

from transformers import AutoConfig, AutoProcessor, AutoTokenizer

root = pathlib.Path(sys.argv[1])
config = AutoConfig.from_pretrained(root, trust_remote_code=True)
AutoTokenizer.from_pretrained(root, trust_remote_code=True)
AutoProcessor.from_pretrained(root, trust_remote_code=True)
weights = list(root.glob("*.safetensors"))
if not weights:
    raise SystemExit(f"no merged safetensors found in {root}")
summary = {
    "architectures": getattr(config, "architectures", None),
    "hidden_size": getattr(getattr(config, "text_config", config), "hidden_size", None),
    "safetensor_files": len(weights),
}
(root / "merge_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

log "merged Hugging Face checkpoint: ${OUTPUT_DIR}"

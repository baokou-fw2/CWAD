#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

(
  cd "${CWAD_PACKAGE_ROOT}"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum --check provenance/source_files.sha256
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 --check provenance/source_files.sha256
  else
    die "neither sha256sum nor shasum is available"
  fi
)

log "all recorded native source files match"

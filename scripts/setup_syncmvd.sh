#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYNCMVD_ROOT="${1:-${ROOT_DIR}/third_party/SyncMVD}"
OVERLAY_DIR="${ROOT_DIR}/integrations/syncmvd/overlay"

if [[ ! -d "${SYNCMVD_ROOT}/.git" && ! -f "${SYNCMVD_ROOT}/.git" ]]; then
    echo "SyncMVD checkout not found: ${SYNCMVD_ROOT}" >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

cp -a "${OVERLAY_DIR}/." "${SYNCMVD_ROOT}/"
"${PYTHON_BIN:-python}" "${ROOT_DIR}/scripts/check_syncmvd.py" --root "${SYNCMVD_ROOT}"

echo "Luiviton SyncMVD overlay installed in ${SYNCMVD_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFUSION_NET_ROOT="${1:-${ROOT_DIR}/third_party/diffusion_net}"
OVERLAY_DIR="${ROOT_DIR}/integrations/diffusion_net/overlay"

if [[ ! -d "${DIFFUSION_NET_ROOT}/src/diffusion_net" ]]; then
    echo "DiffusionNet source tree not found: ${DIFFUSION_NET_ROOT}" >&2
    echo "Clone https://github.com/nmwsharp/diffusion-net.git there first." >&2
    exit 1
fi

cp -a "${OVERLAY_DIR}/." "${DIFFUSION_NET_ROOT}/"
"${PYTHON_BIN:-python}" "${ROOT_DIR}/scripts/check_diffusion_net.py" \
    --root "${DIFFUSION_NET_ROOT}"

echo "Luiviton DiffusionNet overlay installed in ${DIFFUSION_NET_ROOT}"

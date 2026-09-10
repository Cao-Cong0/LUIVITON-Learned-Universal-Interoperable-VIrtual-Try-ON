#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-${ROOT_DIR}/third_party}"
PYTHON_BIN="${PYTHON_BIN:-python}"

clone_if_missing() {
    local url="$1"
    local destination="$2"
    if [[ -d "${destination}/.git" || -f "${destination}/.git" ]]; then
        echo "Reusing checkout: ${destination}"
        return
    fi
    if [[ -d "${destination}" && -n "$(ls -A "${destination}")" ]]; then
        echo "Reusing source tree without Git metadata: ${destination}"
        return
    fi
    if [[ -e "${destination}" && ! -d "${destination}" ]]; then
        echo "Path exists but is not a directory: ${destination}" >&2
        exit 1
    fi
    git clone --depth 1 "${url}" "${destination}"
}

mkdir -p "${THIRD_PARTY_ROOT}"
if [[ "${THIRD_PARTY_ROOT}" == "${ROOT_DIR}/third_party" &&
      ( -d "${ROOT_DIR}/.git" || -f "${ROOT_DIR}/.git" ) ]]; then
    git -C "${ROOT_DIR}" submodule update --init --recursive
fi

clone_if_missing https://github.com/MPI-IS/mesh.git "${THIRD_PARTY_ROOT}/psbody_mesh"
clone_if_missing https://github.com/nmwsharp/diffusion-net.git "${THIRD_PARTY_ROOT}/diffusion_net"
clone_if_missing https://github.com/Dolorousrtur/ContourCraft.git "${THIRD_PARTY_ROOT}/ContourCraft"
clone_if_missing https://github.com/LIU-Yuxin/SyncMVD.git "${THIRD_PARTY_ROOT}/SyncMVD"
clone_if_missing https://github.com/bharat-b7/RVH_Mesh_Registration.git \
    "${THIRD_PARTY_ROOT}/RVH_Mesh_Registration"

"${PYTHON_BIN}" -m pip install --no-build-isolation --no-deps \
    "${THIRD_PARTY_ROOT}/psbody_mesh"

PYTHON_BIN="${PYTHON_BIN}" bash "${ROOT_DIR}/scripts/setup_diffusion_net.sh" \
    "${THIRD_PARTY_ROOT}/diffusion_net"
PYTHON_BIN="${PYTHON_BIN}" bash "${ROOT_DIR}/scripts/setup_syncmvd.sh" \
    "${THIRD_PARTY_ROOT}/SyncMVD"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/install_contourcraft_compat.py" \
    --contourcraft-root "${THIRD_PARTY_ROOT}/ContourCraft"
CONTOURCRAFT_DATA_ROOT="${THIRD_PARTY_ROOT}/contourcraft_data" \
PYTHON_BIN="${PYTHON_BIN}" bash "${ROOT_DIR}/scripts/setup_rvh.sh" \
    "${THIRD_PARTY_ROOT}/RVH_Mesh_Registration"

echo "All Luiviton third-party integrations are ready under ${THIRD_PARTY_ROOT}"

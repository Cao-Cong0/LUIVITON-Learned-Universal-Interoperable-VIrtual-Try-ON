#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVH_ROOT="${1:-${ROOT_DIR}/third_party/RVH_Mesh_Registration}"
OVERLAY_DIR="${ROOT_DIR}/integrations/rvh/overlay"
CONTOURCRAFT_DATA_ROOT="${CONTOURCRAFT_DATA_ROOT:-${ROOT_DIR}/third_party/contourcraft_data}"

if [[ ! -d "${RVH_ROOT}/.git" && ! -f "${RVH_ROOT}/.git" ]]; then
    echo "RVH Mesh Registration checkout not found: ${RVH_ROOT}" >&2
    echo "Clone https://github.com/bharat-b7/RVH_Mesh_Registration.git there first." >&2
    exit 1
fi

cp -a "${OVERLAY_DIR}/." "${RVH_ROOT}/"

link_smpl_model() {
    local source_name="$1"
    local target_name="$2"
    local required="${3:-1}"
    local source="${CONTOURCRAFT_DATA_ROOT}/aux_data/body_models/smpl/${source_name}"
    local target_dir="${RVH_ROOT}/assets"
    local target="${target_dir}/${target_name}"

    if [[ ! -f "${source}" ]]; then
        if [[ "${required}" == "1" ]]; then
            echo "Required SMPL model not found: ${source}" >&2
        fi
        return
    fi
    if [[ -e "${target}" && ! -L "${target}" ]]; then
        echo "Keeping existing RVH SMPL model: ${target}"
        return
    fi

    mkdir -p "${target_dir}"
    ln -sfn "$(realpath --relative-to="${target_dir}" "${source}")" "${target}"
    echo "Linked ${target_name} to ContourCraft data"
}

link_smpl_model SMPL_FEMALE.pkl SMPL_female.pkl
link_smpl_model SMPL_MALE.pkl SMPL_male.pkl 0
"${PYTHON_BIN:-python}" "${ROOT_DIR}/scripts/check_rvh.py" --root "${RVH_ROOT}"

echo "Luiviton RVH overlay installed in ${RVH_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_pipeline.sh BODY.obj CLOTH.obj

Optional environment variables:
  LUIVITON_ENV=Luiviton        Environment for body and garment registration
  CCRAFT_ENV=ccraft            Environment for ContourCraft transfer
  DEVICE=cuda:0                Torch device used by all GPU stages
  GENDER=female                Shared SMPL gender
  OUTPUT_ROOT=PATH            Output directory (default: release outputs/)
  SMPL_MODEL_ROOT=PATH         Shared licensed SMPL asset directory
  CONTOURCRAFT_DATA=PATH      ContourCraft data root
  SOURCE_FEATURE_CACHE=PATH    Precomputed source-SMPL features (default: bundled asset)
  SMPL_SOURCE_RESULTS=PATH     Optional fallback used to recompute source features
  SMPL_REFERENCE_MESH=PATH     Source-SMPL mesh (default: bundled asset)
  SYNCMVD_ROOT=PATH            Prepared SyncMVD checkout
  RUN_RECONSTRUCTION=0|1       Reconstruct the body mesh (default: 1)
  RUN_SYNCMVD=0|1              Run target-body SyncMVD (default: 1)
  SETUP_SYNCMVD=0|1            Apply the SyncMVD overlay (default: RUN_BODY)
  RUN_BODY=0|1                 Run body correspondence/registration (default: 1)
  RUN_CLOTH_REGISTRATION=0|1  Run garment correspondence/registration (default: 1)
  RUN_TRANSFER=0|1             Run ContourCraft transfer (default: 1)
EOF
}

if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
fi

BODY_MESH="$(realpath "$1")"
CLOTH_OBJ="$(realpath "$2")"
BODY_NAME="$(basename "${BODY_MESH}" .obj)"
CLOTH_NAME="$(basename "${CLOTH_OBJ}" .obj)"

# Keep the previous override spelling compatible with existing launch commands.
LUIVITON_ENV="${LUIVITON_ENV:-${LUVITON_ENV:-Luiviton}}"
CCRAFT_ENV="${CCRAFT_ENV:-ccraft}"
DEVICE="${DEVICE:-cuda:0}"
GENDER="${GENDER:-female}"
CONTOURCRAFT_DATA="${CONTOURCRAFT_DATA:-${ROOT_DIR}/third_party/contourcraft_data}"
SMPL_MODEL_ROOT="${SMPL_MODEL_ROOT:-${ROOT_DIR}/third_party/RVH_Mesh_Registration/assets}"
SOURCE_FEATURE_CACHE="${SOURCE_FEATURE_CACHE:-${ROOT_DIR}/assets/source_smpl/smpl_features_18_views_0.pt}"
SMPL_SOURCE_RESULTS="${SMPL_SOURCE_RESULTS:-}"
SMPL_REFERENCE_MESH="${SMPL_REFERENCE_MESH:-${ROOT_DIR}/assets/source_smpl/smpl_wide.obj}"
SYNCMVD_ROOT="${SYNCMVD_ROOT:-${ROOT_DIR}/third_party/SyncMVD}"

RUN_RECONSTRUCTION="${RUN_RECONSTRUCTION:-1}"
RUN_SYNCMVD="${RUN_SYNCMVD:-1}"
RUN_BODY="${RUN_BODY:-1}"
RUN_CLOTH_REGISTRATION="${RUN_CLOTH_REGISTRATION:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"
SETUP_SYNCMVD="${SETUP_SYNCMVD:-${RUN_BODY}}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs}"
if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${ROOT_DIR}/${OUTPUT_ROOT}"
fi
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"

BODY_INPUT_DIR="${OUTPUT_ROOT}/inputs/${BODY_NAME}"
BODY_OUTPUT="${OUTPUT_ROOT}/body"
BODY_REG="${BODY_OUTPUT}/registration/${BODY_NAME}"
GARMENT_OUTPUT="${OUTPUT_ROOT}/garment"
CLOTH_CORR="${GARMENT_OUTPUT}/correspondence/cloth/top_1_indices_${CLOTH_NAME}.npz"
CLOTH_SMPL_PKL="${GARMENT_OUTPUT}/cloth_registered_smpl/${CLOTH_NAME}_smpl.pkl"
FINAL_OUTPUT="${FINAL_OUTPUT:-${GARMENT_OUTPUT}/transferred/${CLOTH_NAME}_on_${BODY_NAME}.obj}"

require_file() {
    [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }
}

require_dir() {
    [[ -d "$1" ]] || { echo "Required directory not found: $1" >&2; exit 1; }
}

require_file "${BODY_MESH}"
require_file "${CLOTH_OBJ}"
if [[ "${RUN_BODY}" == "1" ]]; then
    require_file "${SMPL_REFERENCE_MESH}"
    if [[ ! -f "${SOURCE_FEATURE_CACHE}" ]]; then
        [[ -n "${SMPL_SOURCE_RESULTS}" ]] || {
            echo "Required source feature file not found: ${SOURCE_FEATURE_CACHE}" >&2
            exit 1
        }
        require_dir "${SMPL_SOURCE_RESULTS}"
    fi
    require_dir "${SMPL_MODEL_ROOT}"
fi
if [[ "${RUN_CLOTH_REGISTRATION}" == "1" ]]; then
    require_dir "${SMPL_MODEL_ROOT}"
fi
if [[ "${RUN_BODY}" == "1" || "${RUN_CLOTH_REGISTRATION}" == "1" ]]; then
    for smpl_asset in \
        SMPL_female.pkl \
        priors/body_prior.pkl; do
        require_file "${SMPL_MODEL_ROOT}/${smpl_asset}"
    done
fi
if [[ "${SETUP_SYNCMVD}" == "1" || "${RUN_BODY}" == "1" ]]; then
    require_dir "${SYNCMVD_ROOT}"
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set -u

activate_conda_env() {
    # Some Conda activation hooks inspect optional variables before defining them.
    set +u
    conda activate "$1"
    set -u
}

mkdir -p "${BODY_INPUT_DIR}" "$(dirname "${FINAL_OUTPUT}")"
ln -sfn "${BODY_MESH}" "${BODY_INPUT_DIR}/${BODY_NAME}.obj"
cd "${ROOT_DIR}"

activate_conda_env "${LUIVITON_ENV}"
if [[ "${SETUP_SYNCMVD}" == "1" ]]; then
    echo "=== 1/5: Prepare SyncMVD ==="
    PYTHON_BIN="$(command -v python)" \
        bash "${ROOT_DIR}/scripts/setup_syncmvd.sh" "${SYNCMVD_ROOT}"
else
    echo "=== 1/5: Reuse prepared SyncMVD ==="
fi

if [[ "${RUN_BODY}" == "1" ]]; then
    echo "=== 2/5: Body correspondence and registration [${LUIVITON_ENV}] ==="
    INPUT_MESH_DIR="${BODY_INPUT_DIR}" \
    SYNCMVD_ROOT="${SYNCMVD_ROOT}" \
    SYNCMVD_CONFIG="${ROOT_DIR}/configs/syncmvd_body.yml" \
    SYNCMVD_OUTPUT_ROOT="${BODY_OUTPUT}/syncmvd" \
    SMPL_SOURCE_RESULTS="${SMPL_SOURCE_RESULTS}" \
    SMPL_REFERENCE_MESH="${SMPL_REFERENCE_MESH}" \
    SOURCE_FEATURE_CACHE="${SOURCE_FEATURE_CACHE}" \
    SMPL_CONFIG="${ROOT_DIR}/smpl_registration/smpl_registration/config.yml" \
    SMPL_MODEL_ROOT="${SMPL_MODEL_ROOT}" \
    OUTPUT_ROOT="${BODY_OUTPUT}" \
    GENDER="${GENDER}" \
    DEVICE="${DEVICE}" \
    RUN_RECONSTRUCTION="${RUN_RECONSTRUCTION}" \
    RUN_SYNCMVD="${RUN_SYNCMVD}" \
    RUN_FILTER=1 \
    PYTHON_BIN="$(command -v python)" \
        bash "${ROOT_DIR}/scripts/full_body_registration.sh" /dev/null
else
    echo "=== 2/5: Reuse existing body registration ==="
fi

require_file "${BODY_REG}/textured_smpld.obj"
require_file "${BODY_REG}/textured_smpld.pkl"
require_file "${BODY_REG}/mesh_transform.npy"

if [[ "${RUN_CLOTH_REGISTRATION}" == "1" ]]; then
    echo "=== 3/5: Clothing correspondence [${LUIVITON_ENV}] ==="
    python -m luiviton correspondence \
        --repo-root "${ROOT_DIR}" \
        --output-root "${GARMENT_OUTPUT}" \
        --cloth "${CLOTH_OBJ}" \
        --smpl-template-body "${ROOT_DIR}/models/smpl_uv_free.obj" \
        --correspondence-checkpoint "${ROOT_DIR}/models/cloth_correspondence.pth" \
        --device "${DEVICE}" \
        --overwrite

    echo "=== 4/5: Clothing SMPL registration [${LUIVITON_ENV}] ==="
    python -m luiviton register \
        --repo-root "${ROOT_DIR}" \
        --output-root "${GARMENT_OUTPUT}" \
        --cloth "${CLOTH_OBJ}" \
        --correspondence "${CLOTH_CORR}" \
        --smpl-config "${ROOT_DIR}/configs/smpl.yml" \
        --smpl-model-root "${SMPL_MODEL_ROOT}" \
        --smpl-template-body "${ROOT_DIR}/models/smpl_uv_free.obj" \
        --gender "${GENDER}" \
        --device "${DEVICE}" \
        --overwrite
else
    echo "=== 3/5: Reuse existing clothing correspondence ==="
    echo "=== 4/5: Reuse existing clothing registration ==="
fi

require_file "${CLOTH_SMPL_PKL}"

if [[ "${RUN_TRANSFER}" == "1" ]]; then
    echo "=== 5/5: ContourCraft clothing transfer [${CCRAFT_ENV}] ==="
    activate_conda_env "${CCRAFT_ENV}"
    python -m luiviton transfer \
        --repo-root "${ROOT_DIR}" \
        --output-root "${GARMENT_OUTPUT}" \
        --cloth "${CLOTH_OBJ}" \
        --source-body-pkl "${CLOTH_SMPL_PKL}" \
        --body "${BODY_REG}/textured_smpld.obj" \
        --target-body-pkl "${BODY_REG}/textured_smpld.pkl" \
        --body-transform "${BODY_REG}/mesh_transform.npy" \
        --contourcraft-root "${ROOT_DIR}/third_party/ContourCraft" \
        --contourcraft-data "${CONTOURCRAFT_DATA}" \
        --device "${DEVICE}" \
        --num-frames 15 \
        --collision-frames 10 \
        --output "${FINAL_OUTPUT}"
else
    echo "=== 5/5: Skip ContourCraft transfer ==="
fi

echo "Finished: ${FINAL_OUTPUT}"

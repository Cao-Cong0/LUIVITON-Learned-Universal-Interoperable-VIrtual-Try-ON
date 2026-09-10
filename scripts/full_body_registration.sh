#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-${ROOT_DIR}/configs/paths.example.sh}"

if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
fi

: "${INPUT_MESH_DIR:?Set INPUT_MESH_DIR to a folder containing .obj files.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/body}"
SYNCMVD_ROOT="${SYNCMVD_ROOT:-${ROOT_DIR}/third_party/SyncMVD}"
SYNCMVD_CONFIG="${SYNCMVD_CONFIG:-${ROOT_DIR}/configs/syncmvd_body.yml}"
SYNCMVD_OUTPUT_ROOT="${SYNCMVD_OUTPUT_ROOT:-${OUTPUT_ROOT}/syncmvd}"
SMPL_SOURCE_RESULTS="${SMPL_SOURCE_RESULTS:-}"
SMPL_REFERENCE_MESH="${SMPL_REFERENCE_MESH:-${ROOT_DIR}/assets/source_smpl/smpl_wide.obj}"
SMPL_CONFIG="${SMPL_CONFIG:-${ROOT_DIR}/smpl_registration/smpl_registration/config.yml}"
SMPL_MODEL_ROOT="${SMPL_MODEL_ROOT:-}"
GENDER="${GENDER:-female}"
DEVICE="${DEVICE:-cuda:0}"
RUN_RECONSTRUCTION="${RUN_RECONSTRUCTION:-1}"
RUN_SYNCMVD="${RUN_SYNCMVD:-1}"
RUN_FILTER="${RUN_FILTER:-1}"

absolute_path() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        realpath -m -- "${path}"
    else
        realpath -m -- "${ROOT_DIR}/${path}"
    fi
}

INPUT_MESH_DIR="$(absolute_path "${INPUT_MESH_DIR}")"
SYNCMVD_ROOT="$(absolute_path "${SYNCMVD_ROOT}")"
SYNCMVD_CONFIG="$(absolute_path "${SYNCMVD_CONFIG}")"
if [[ -n "${SMPL_SOURCE_RESULTS}" ]]; then
    SMPL_SOURCE_RESULTS="$(absolute_path "${SMPL_SOURCE_RESULTS}")"
fi
SMPL_REFERENCE_MESH="$(absolute_path "${SMPL_REFERENCE_MESH}")"
OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT}")"
SYNCMVD_OUTPUT_ROOT="$(absolute_path "${SYNCMVD_OUTPUT_ROOT}")"
SMPL_CONFIG="$(absolute_path "${SMPL_CONFIG}")"
if [[ -n "${SMPL_MODEL_ROOT}" ]]; then
    SMPL_MODEL_ROOT="$(absolute_path "${SMPL_MODEL_ROOT}")"
fi
SOURCE_FEATURE_CACHE="${SOURCE_FEATURE_CACHE:-${ROOT_DIR}/assets/source_smpl/smpl_features_18_views_0.pt}"
SOURCE_FEATURE_CACHE="$(absolute_path "${SOURCE_FEATURE_CACHE}")"

require_source_file() {
    local path="$1"
    [[ -f "${path}" ]] || { echo "Required source-SMPL file not found: ${path}" >&2; exit 1; }
}

[[ -d "${INPUT_MESH_DIR}" ]] || { echo "Input mesh directory not found: ${INPUT_MESH_DIR}" >&2; exit 1; }
[[ -d "${SYNCMVD_ROOT}" ]] || { echo "SyncMVD checkout not found: ${SYNCMVD_ROOT}" >&2; exit 1; }
[[ -f "${SYNCMVD_CONFIG}" ]] || { echo "SyncMVD config not found: ${SYNCMVD_CONFIG}" >&2; exit 1; }
[[ -f "${SMPL_REFERENCE_MESH}" ]] || { echo "SMPL reference mesh not found: ${SMPL_REFERENCE_MESH}" >&2; exit 1; }
[[ -f "${SMPL_CONFIG}" ]] || { echo "SMPL config not found: ${SMPL_CONFIG}" >&2; exit 1; }
if [[ ! -f "${SOURCE_FEATURE_CACHE}" ]]; then
    [[ -n "${SMPL_SOURCE_RESULTS}" ]] || {
        echo "Source feature file not found: ${SOURCE_FEATURE_CACHE}" >&2
        echo "Set SMPL_SOURCE_RESULTS to compute it from source SyncMVD results." >&2
        exit 1
    }
    [[ -d "${SMPL_SOURCE_RESULTS}" ]] || { echo "SMPL source results not found: ${SMPL_SOURCE_RESULTS}" >&2; exit 1; }
    require_source_file "${SMPL_SOURCE_RESULTS}/f_maps.npz"
    require_source_file "${SMPL_SOURCE_RESULTS}/verts.npy"
    require_source_file "${SMPL_SOURCE_RESULTS}/textured.obj"
    [[ -d "${SMPL_SOURCE_RESULTS}/rgb_views" ]] || {
        echo "Required source-SMPL render directory not found: ${SMPL_SOURCE_RESULTS}/rgb_views" >&2
        exit 1
    }
fi

if [[ "${RUN_SYNCMVD}" == "1" ]]; then
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_syncmvd.py" --root "${SYNCMVD_ROOT}"
fi

RECON_DIR="${OUTPUT_ROOT}/reconstructed_meshes"
CORR_DIR="${OUTPUT_ROOT}/correspondences"
REG_DIR="${OUTPUT_ROOT}/registration"
mkdir -p "${RECON_DIR}" "${CORR_DIR}" "${REG_DIR}" "$(dirname "${SOURCE_FEATURE_CACHE}")"

ORIGINAL_PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="${ROOT_DIR}/correspondence/body:${ROOT_DIR}/third_party/psbody_mesh:${ROOT_DIR}/smpl_registration:${ORIGINAL_PYTHONPATH}"

shopt -s nullglob
meshes=("${INPUT_MESH_DIR}"/*.obj)
if (( ${#meshes[@]} == 0 )); then
    echo "No .obj files found in ${INPUT_MESH_DIR}" >&2
    exit 1
fi

for input_mesh in "${meshes[@]}"; do
    start_time=$(date +%s)
    mesh_name="$(basename "${input_mesh}" .obj)"
    echo "Processing ${mesh_name}"

    reconstructed_mesh="${RECON_DIR}/${mesh_name}.obj"
    if [[ "${RUN_RECONSTRUCTION}" == "1" ]]; then
        "${PYTHON_BIN}" "${ROOT_DIR}/mesh_reconstruct/watertight_3d_scan.py" \
            --input_obj_path "${input_mesh}" \
            --output_obj_path "${reconstructed_mesh}"
    else
        reconstructed_mesh="${input_mesh}"
    fi

    if [[ "${RUN_SYNCMVD}" == "1" ]]; then
        echo "Running SyncMVD with mesh ${reconstructed_mesh}"
        (cd "${SYNCMVD_ROOT}" && "${PYTHON_BIN}" run_experiment.py \
            --config "${SYNCMVD_CONFIG}" \
            --mesh "${reconstructed_mesh}" \
            --output "${SYNCMVD_OUTPUT_ROOT}")
    fi

    target_results="${SYNCMVD_OUTPUT_ROOT}/scan_${mesh_name}/results"
    scan_path="${target_results}/textured.obj"
    transform_path="${target_results}/mesh_transform.npy"
    verts_path="${target_results}/verts.npy"
    correspondence_path="${CORR_DIR}/scan_${mesh_name}.npy"
    rejected_path="${CORR_DIR}/scan_${mesh_name}_rejected.npy"
    registration_output_dir="${REG_DIR}/${mesh_name}"

    for required_result in "${scan_path}" "${target_results}/f_maps.npz" "${verts_path}" "${transform_path}"; do
        if [[ ! -f "${required_result}" ]]; then
            echo "Required SyncMVD result not found: ${required_result}" >&2
            exit 1
        fi
    done

    feature_args=(
        --target_file_folder "${target_results}"
        --correspondence_path "${correspondence_path}"
        --source_feature_save_path "${SOURCE_FEATURE_CACHE}"
    )
    if [[ -n "${SMPL_SOURCE_RESULTS}" ]]; then
        feature_args+=(--source_file_folder "${SMPL_SOURCE_RESULTS}")
    fi
    "${PYTHON_BIN}" "${ROOT_DIR}/correspondence/body/ablation_src/feature_aggregation.py" "${feature_args[@]}"

    registration_args=(
        "${scan_path}"
        "${correspondence_path}"
        "${registration_output_dir}"
        -transformation_path "${transform_path}"
        -gender "${GENDER}"
        --device "${DEVICE}"
        --config-path "${SMPL_CONFIG}"
    )

    if [[ -n "${SMPL_MODEL_ROOT}" ]]; then
        registration_args+=(--smpl-model-root "${SMPL_MODEL_ROOT}")
    fi

    if [[ "${RUN_FILTER}" == "1" ]]; then
        "${PYTHON_BIN}" "${ROOT_DIR}/correspondence/body/filter_iterative.py" \
            --source_mesh_path "${SMPL_REFERENCE_MESH}" \
            --target_file_folder "${target_results}" \
            --target_name "${mesh_name}" \
            --correspondence_folder "${CORR_DIR}"
        registration_args+=(-rejected_path "${rejected_path}")
    fi

    (
        cd "${ROOT_DIR}/smpl_registration"
        PYTHONPATH="${ROOT_DIR}/smpl_registration:${ROOT_DIR}/third_party/psbody_mesh:${ROOT_DIR}/correspondence/body:${ORIGINAL_PYTHONPATH}" \
            "${PYTHON_BIN}" smpl_registration/fit_SMPLHD_body.py "${registration_args[@]}"
    )

    cp -- "${transform_path}" "${registration_output_dir}/mesh_transform.npy"
    cp -- "${verts_path}" "${registration_output_dir}/verts.npy"

    elapsed=$(( $(date +%s) - start_time ))
    echo "Finished ${mesh_name} in ${elapsed}s"
done

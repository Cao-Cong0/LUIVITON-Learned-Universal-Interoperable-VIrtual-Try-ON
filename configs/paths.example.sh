# Copy this file or pass it directly to scripts/full_body_registration.sh.

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Folder with input body OBJ meshes.
export INPUT_MESH_DIR="/path/to/input_meshes"

# Release outputs and the prepared SyncMVD submodule.
export OUTPUT_ROOT="${RELEASE_ROOT}/outputs/body"
export SYNCMVD_ROOT="${RELEASE_ROOT}/third_party/SyncMVD"
export SYNCMVD_CONFIG="${RELEASE_ROOT}/configs/syncmvd_body.yml"
export SYNCMVD_OUTPUT_ROOT="${OUTPUT_ROOT}/syncmvd"

# Bundled source-SMPL feature tensor. Raw source SyncMVD results are not needed while this exists.
export SOURCE_FEATURE_CACHE="${RELEASE_ROOT}/assets/source_smpl/smpl_features_18_views_0.pt"

# Optional fallback used only to recompute SOURCE_FEATURE_CACHE.
# export SMPL_SOURCE_RESULTS="/path/to/source_smpl/results"

# Reference mesh whose vertex order matches the correspondence indices.
export SMPL_REFERENCE_MESH="${RELEASE_ROOT}/assets/source_smpl/smpl_wide.obj"

# Registration config. Its model and asset paths must point to your licensed
# SMPL files; these files are not distributed with this repository.
export SMPL_CONFIG="${RELEASE_ROOT}/smpl_registration/smpl_registration/config.yml"

# Run switches.
export GENDER="female"
export DEVICE="cuda:0"
export RUN_RECONSTRUCTION=1
export RUN_SYNCMVD=1
export RUN_FILTER=1
export PYTHON_BIN="python"

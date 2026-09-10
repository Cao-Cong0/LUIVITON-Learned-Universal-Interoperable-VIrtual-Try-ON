"""Compatibility checks for the ContourCraft runtime used by Luiviton."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REQUIRED_PROJECT_FILES = (
    "configs/aux/from_any_pose_resize.yaml",
    "datasets/from_any_pose_resize.py",
    "runners/from_any_pose_resize.py",
    "utils/Luviton.py",
    "utils/arguments.py",
    "utils/datasets.py",
    "utils/defaults.py",
    "utils/cloth_and_material.py",
    "utils/io.py",
    "utils/mesh_creation.py",
    "utils/validation.py",
)

REQUIRED_DATA_FILES = (
    "trained_models/contourcraft.pth",
    "aux_data/smpl_aux.pkl",
    "aux_data/body_models/smpl/SMPL_FEMALE.pkl",
)

UPSTREAM_DATA_DIRECTORIES = (
    "aux_data/datasplits",
    "aux_data/body_models/smpl",
    "aux_data/body_models/smplx",
    "aux_data/garment_dicts",
    "aux_data/garment_meshes",
    "trained_models",
    "examples/fromanypose",
    "examples/unpose",
)

UPSTREAM_DATA_FILES = (
    "aux_data/body_models/smpl/SMPL_FEMALE.pkl",
    "aux_data/body_models/smpl/SMPL_MALE.pkl",
    "aux_data/body_models/smplx/SMPLX_NEUTRAL.pkl",
    "aux_data/body_models/smplx/SMPLX_FEMALE.pkl",
    "aux_data/body_models/smplx/SMPLX_MALE.pkl",
    "aux_data/smpl_aux.pkl",
    "trained_models/hood_cvpr.pth",
    "trained_models/hood_final.pth",
    "trained_models/contourcraft.pth",
)


@dataclass(frozen=True)
class ContourCraftValidation:
    project_root: Path
    data_root: Path
    git_commit: Optional[str]
    missing_project_files: tuple[str, ...]
    missing_data_files: tuple[str, ...]
    missing_upstream_data: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_project_files and not self.missing_data_files

    @property
    def full_data_ok(self) -> bool:
        return not self.missing_upstream_data


def git_commit(project_root: Path) -> Optional[str]:
    project_root = project_root.expanduser().resolve()
    try:
        top_level = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    detected_root = Path(top_level.stdout.strip()).expanduser().resolve()
    if detected_root != project_root:
        return None

    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def validate_contourcraft(project_root: str | Path, data_root: str | Path) -> ContourCraftValidation:
    project_root = Path(project_root).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    missing_project = tuple(
        relative for relative in REQUIRED_PROJECT_FILES if not (project_root / relative).is_file()
    )
    missing_data = tuple(
        relative for relative in REQUIRED_DATA_FILES if not (data_root / relative).is_file()
    )
    missing_upstream_data = tuple(
        relative for relative in UPSTREAM_DATA_DIRECTORIES if not (data_root / relative).is_dir()
    ) + tuple(relative for relative in UPSTREAM_DATA_FILES if not (data_root / relative).is_file())
    return ContourCraftValidation(
        project_root=project_root,
        data_root=data_root,
        git_commit=git_commit(project_root),
        missing_project_files=missing_project,
        missing_data_files=missing_data,
        missing_upstream_data=missing_upstream_data,
    )


def format_validation_error(validation: ContourCraftValidation) -> str:
    lines = [
        "The ContourCraft checkout is not compatible with the Luiviton transfer adapter.",
        f"Project: {validation.project_root}",
        f"Data: {validation.data_root}",
    ]
    if validation.git_commit:
        lines.append(f"Detected commit: {validation.git_commit}")
    if validation.missing_project_files:
        lines.append("Missing Luiviton integration files:")
        lines.extend(f"  - {path}" for path in validation.missing_project_files)
    if validation.missing_data_files:
        lines.append("Missing ContourCraft runtime data:")
        lines.extend(f"  - {path}" for path in validation.missing_data_files)
    lines.extend(
        [
            "See external/CONTOURCRAFT_SETUP.md.",
            "For a fresh clone, apply the Luiviton integration and run this check again:",
            "  python scripts/install_contourcraft_compat.py --contourcraft-root /path/to/ContourCraft",
        ]
    )
    return "\n".join(lines)

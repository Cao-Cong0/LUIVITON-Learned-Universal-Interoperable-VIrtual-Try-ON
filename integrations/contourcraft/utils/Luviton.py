"""Luiviton motion helpers used by the ContourCraft transfer adapter."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, MutableMapping

import numpy as np
import torch
from smplx import SMPL

from luiviton.motion import interpolate_axis_angle_poses
from utils.defaults import DEFAULTS


__all__ = (
    "generate_complex_mesh_sequence_single_obj",
    "generate_pose_sequence",
    "process_sample",
)


def process_sample(
    sample: MutableMapping[str, Any], pose_sequence_path: str | Path
) -> MutableMapping[str, Any]:
    """Translate a cloth sample into the generated body sequence coordinate frame."""
    with Path(pose_sequence_path).open("rb") as file:
        motion_sequence = pickle.load(file)

    cloth = sample["cloth"]
    cloth_vertices = cloth["pos"]
    translation = torch.as_tensor(
        motion_sequence["transl_diff"],
        dtype=cloth_vertices.dtype,
        device=cloth_vertices.device,
    )
    translated_vertices = cloth_vertices + translation

    cloth["pos"] = translated_vertices
    cloth["prev_pos"] = translated_vertices
    cloth["target_pos"] = translated_vertices
    cloth["rest_pos"] = cloth["rest_pos"] + translation
    return sample


def generate_complex_mesh_sequence_single_obj(
    input_obj: str | Path, output_path: str | Path, num_frames: int
) -> None:
    """Repeat one body mesh to create a static collision-resolution sequence."""
    _require_positive_frames(num_frames)
    vertices, faces = _load_triangle_obj(input_obj)
    sequence = {
        "verts": np.repeat(vertices[np.newaxis, ...], num_frames, axis=0),
        "faces": faces,
        "transl_diff": np.zeros(3, dtype=vertices.dtype),
    }
    _write_sequence(sequence, output_path)


def generate_pose_sequence(
    source_pkl: str | Path,
    target_pkl: str | Path,
    transformation_path: str | Path | None,
    output_path: str | Path,
    num_frames: int,
) -> np.ndarray:
    """Interpolate registered source and target SMPL+D bodies for clothing transfer."""
    _require_positive_frames(num_frames)
    source = _read_pickle(source_pkl)
    target = _read_pickle(target_pkl)

    source_pose = np.asarray(source["pose"]).reshape(1, -1)
    source_betas = np.asarray(source["betas"]).reshape(1, -1)
    source_translation = np.asarray(source["trans"]).reshape(1, -1)

    target_pose = np.asarray(target["pose"]).reshape(1, -1)
    target_betas = np.asarray(target["betas"]).reshape(1, -1)
    target_translation = np.asarray(target["trans"]).reshape(1, -1)
    target_offsets = np.asarray(target["offsets"])
    target_scale = np.asarray(target["scale"])

    center, transform_scale = _load_mesh_transform(transformation_path)
    final_scale = target_scale / transform_scale
    final_translation = target_translation / transform_scale + center
    final_offsets = target_offsets / transform_scale

    poses = interpolate_axis_angle_poses(source_pose, target_pose, num_frames)
    betas = np.linspace(source_betas, target_betas, num_frames)
    scales = np.linspace(np.ones(3), final_scale, num_frames)
    offsets = np.linspace(np.zeros_like(final_offsets), final_offsets, num_frames)

    model_root = Path(DEFAULTS.aux_data) / "body_models" / "smpl"
    smpl = SMPL(model_path=str(model_root), gender="female")
    vertices_sequence = []
    for frame in range(num_frames):
        output = smpl.forward(
            betas=torch.as_tensor(betas[frame], dtype=torch.float32),
            body_pose=torch.as_tensor(poses[frame][:, 3:], dtype=torch.float32),
            global_orient=torch.as_tensor(poses[frame][:, :3], dtype=torch.float32),
        )
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        vertices *= scales[frame]
        vertices += offsets[frame]
        vertices += final_translation
        vertices_sequence.append(vertices)

    sequence = {
        "verts": np.asarray(vertices_sequence),
        "faces": smpl.faces,
        "transl_diff": final_translation - source_translation,
    }
    _write_sequence(sequence, output_path)
    return final_scale


def _load_mesh_transform(path: str | Path | None) -> tuple[np.ndarray, float | np.ndarray]:
    if path is None:
        return np.zeros(3), np.ones(3)
    transform = np.load(path, allow_pickle=True).item()
    return np.asarray(transform["center"]), np.asarray(transform["scale"])


def _read_pickle(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def _load_triangle_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "f":
                faces.append([int(value.split("/")[0]) - 1 for value in fields[1:4]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _write_sequence(sequence: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(sequence, file)
    print(f"Mesh sequence saved to {output_path}")


def _require_positive_frames(num_frames: int) -> None:
    if num_frames < 1:
        raise ValueError("num_frames must be positive")

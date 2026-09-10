"""SMPL pose interpolation for clothing transfer."""

from __future__ import annotations

import numpy as np


DEFAULT_TRANSITION_FRAMES = 15


def interpolate_axis_angle_poses(
    source_pose: np.ndarray, target_pose: np.ndarray, num_frames: int
) -> np.ndarray:
    """Interpolate joint rotations with quaternion slerp.

    The input arrays retain their original shape after a leading frame dimension
    is added. Their final flattened dimension must contain axis-angle triplets.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be positive")

    source = np.asarray(source_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError(f"Pose shapes differ: {source.shape} != {target.shape}")
    if source.size % 3:
        raise ValueError("Pose arrays must contain axis-angle triplets")

    source_quaternions = _axis_angle_to_quaternion(source.reshape(-1, 3))
    target_quaternions = _axis_angle_to_quaternion(target.reshape(-1, 3))

    dot = np.sum(source_quaternions * target_quaternions, axis=-1, keepdims=True)
    target_quaternions = np.where(dot < 0, -target_quaternions, target_quaternions)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    frames = []
    for alpha in np.linspace(0.0, 1.0, num_frames):
        linear = (1.0 - alpha) * source_quaternions + alpha * target_quaternions
        linear /= np.maximum(np.linalg.norm(linear, axis=-1, keepdims=True), 1e-12)

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        spherical = (
            np.sin((1.0 - alpha) * theta) / np.maximum(sin_theta, 1e-12)
        ) * source_quaternions + (
            np.sin(alpha * theta) / np.maximum(sin_theta, 1e-12)
        ) * target_quaternions
        quaternion = np.where(dot > 0.9995, linear, spherical)
        frames.append(_quaternion_to_axis_angle(quaternion).reshape(source.shape))

    return np.asarray(frames, dtype=np.result_type(source_pose, target_pose, np.float32))


def _axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    scale = np.empty_like(angle)
    small = angle < 1e-8
    scale[small] = 0.5
    scale[~small] = np.sin(half_angle[~small]) / angle[~small]
    return np.concatenate((np.cos(half_angle), axis_angle * scale), axis=-1)


def _quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12
    )
    quaternion = np.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(quaternion[..., :1], -1.0, 1.0))
    scale = np.empty_like(vector_norm)
    small = vector_norm < 1e-8
    scale[small] = 2.0
    scale[~small] = angle[~small] / vector_norm[~small]
    return vector * scale

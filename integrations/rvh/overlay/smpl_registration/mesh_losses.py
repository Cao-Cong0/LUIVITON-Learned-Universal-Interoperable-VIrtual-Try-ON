"""Point-to-surface losses for SMPL registration."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as functional
from pytorch3d import _C
from pytorch3d.loss.point_mesh_distance import (
    _DEFAULT_MIN_TRIANGLE_AREA,
    point_face_distance,
)
from pytorch3d.structures import Meshes, Pointclouds


def point_to_mesh_distance(meshes: Meshes, pointclouds: Pointclouds) -> Tensor:
    """Return the batch mean of per-point squared distance to a mesh surface."""
    squared_distances, _ = point_to_mesh_signed_distances(meshes, pointclouds)
    point_to_cloud = pointclouds.packed_to_cloud_idx()
    points_per_cloud = pointclouds.num_points_per_cloud().gather(0, point_to_cloud)
    return (squared_distances / points_per_cloud.float()).sum() / len(meshes)


def point_to_mesh_signed_distances(
    meshes: Meshes,
    pointclouds: Pointclouds,
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
) -> tuple[Tensor, Tensor]:
    """Return squared distances and oriented distances to each closest triangle."""
    if len(meshes) != len(pointclouds):
        raise ValueError("meshes and pointclouds must contain the same batch size")

    points = pointclouds.points_packed()
    points_first_idx = pointclouds.cloud_to_packed_first_idx()
    max_points = pointclouds.num_points_per_cloud().max().item()

    vertices = meshes.verts_packed()
    triangles = vertices[meshes.faces_packed()]
    triangles_first_idx = meshes.mesh_to_faces_packed_first_idx()

    squared_distances = point_face_distance(
        points,
        points_first_idx,
        triangles,
        triangles_first_idx,
        max_points,
        min_triangle_area,
    )
    with torch.no_grad():
        _, closest_faces = _C.point_face_dist_forward(
            points,
            points_first_idx,
            triangles,
            triangles_first_idx,
            max_points,
            min_triangle_area,
        )

    face_normals = functional.normalize(
        torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1,
        ),
        dim=1,
    )
    selected_triangles = triangles[closest_faces]
    selected_normals = face_normals[closest_faces]
    signed_distances = torch.sum(
        (points - selected_triangles[:, 0]) * selected_normals,
        dim=1,
    )
    return squared_distances, signed_distances

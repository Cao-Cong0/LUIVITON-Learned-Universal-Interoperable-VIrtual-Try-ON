"""Mesh conversion, correspondence comparison, and optional visualization helpers."""

import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import Textures
from pytorch3d.structures import Meshes


def parse_obj(file_path):
    vertices = []
    uvs = []
    vertex_to_uv = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':  # Vertex definition
                # Convert the vertex coordinates to floats and add to the list
                vertex = tuple(map(float, parts[1:4]))
                vertices.append(vertex)

            elif parts[0] == 'vt':  # UV coordinate definition
                # Convert UV coordinates to floats and add to the list
                uv = tuple(map(float, parts[1:3]))
                uvs.append(uv)

            elif parts[0] == 'f':  # Face definition
                # Each part of the face definition corresponds to 'vertex/uv'
                for face_part in parts[1:]:
                    vertex_index, uv_index = map(int, face_part.split('/')[:2])
                    vertex_to_uv[vertex_index - 1] = uv_index - 1  # Adjust for 0-based index

    return vertices, uvs, vertex_to_uv



def extract_vertices_and_uvs(obj_file_path):
    vertices, uvs, vertex_to_uv = parse_obj(obj_file_path)

    # Create a list to hold the UVs in the order of the 3D vertices
    ordered_uvs = [None] * len(vertices)

    # Map the UVs to the corresponding vertex index
    for vertex_index, uv_index in vertex_to_uv.items():
        ordered_uvs[vertex_index] = uvs[uv_index]

    return vertices, ordered_uvs



def get_vertex_colors_from_obj(obj_path, texture_image_path):
    """
    Extract vertex colors from an OBJ file with a texture image, ignoring face information.
    
    Parameters:
        obj_path: Path to the OBJ file.
        texture_image_path: Path to the texture image.
    
    Returns:
        vertices: (N, 3) array of vertex positions.
        vertex_colors: (N, 3) array of RGB colors for each vertex.
    """
    # Load the OBJ file
    vertices, uv_coords = extract_vertices_and_uvs(obj_path)

    if uv_coords is None:
        raise ValueError("UV coordinates not found in the OBJ file.")

    # Load the texture image
    texture_image = Image.open(texture_image_path)

    texture_array = np.asarray(texture_image)/255.0  # Normalize to 0-1 range
    
    # Get texture dimensions
    texture_height, texture_width, _ = texture_array.shape

    # Map UV coordinates to image pixel indices
    uv_pixels = uv_coords * np.array([texture_width, texture_height])  # Scale UV to image resolution
    uv_pixels = np.clip(uv_pixels, 0, [texture_width - 1, texture_height - 1])  # Clip to valid range
    uv_pixels = uv_pixels.astype(int)
    
    # Retrieve colors for each vertex
    vertex_colors = texture_array[uv_pixels[:, 1], uv_pixels[:, 0]]  # Get RGB colors from texture image

    return vertex_colors



def convert_mesh_container_to_torch_mesh(tm, device, is_tosca=True):
    verts_1, faces_1 = torch.tensor(tm.vert, dtype=torch.float32), torch.tensor(
        tm.face, dtype=torch.float32
    )
    if is_tosca:
        verts_1 = verts_1 / 10
    verts_rgb = torch.ones_like(verts_1)[None] * 0.8
    textures = Textures(verts_rgb=verts_rgb)
    mesh = Meshes(verts=[verts_1], faces=[faces_1], textures=textures)
    mesh = mesh.to(device)
    return mesh



def cosine_similarity(a, b):
    # if len(a) > 30000:
    #     return cosine_similarity_batch(a, b, batch_size=30000)
    if a.dtype == torch.float16:
        a = a.to(torch.float32)
    if b.dtype == torch.float16:
        b = b.to(torch.float32)
    dot_product = torch.mm(a, b.t())
    norm_a = torch.norm(a, dim=1, keepdim=True)
    norm_b = torch.norm(b, dim=1, keepdim=True)
    similarity = dot_product / (norm_a * norm_b.t())

    return similarity

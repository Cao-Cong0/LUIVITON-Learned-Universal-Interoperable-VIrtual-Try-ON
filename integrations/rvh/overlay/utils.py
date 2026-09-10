import enum
import math
import os
import pickle
import random

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch

import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.ops import knn_points

from scipy.spatial import Delaunay
import pymeshlab as ml
# from SMPL_Anthropometry.measure import MeasureBody
# from SMPL_Anthropometry.measurement_definitions import STANDARD_LABELS
import json 




def make_einops_str(ndims, insert_k=None):
    linds = ['l', 'm', 'n', 'o', 'p']

    if insert_k is None:
        symbols = linds[:ndims]
    else:
        symbols = linds[:insert_k]
        symbols.append('k')
        symbols += linds[insert_k:ndims]

    out_str = ' '.join(symbols)
    return out_str

def make_repeat_str(tensor, dim):
    ndims = len(tensor.shape)

    out_str = []
    out_str.append(make_einops_str(ndims))
    out_str.append('->')
    out_str.append(make_einops_str(ndims, insert_k=dim))

    out_str = ' '.join(out_str)

    return out_str

def gather(data: torch.Tensor, index: torch.LongTensor, dim_gather: int, dim_data: int, dim_index: int):
    input_repeat_str = make_repeat_str(data, dim_data)
    index_repeat_str = make_repeat_str(index, dim_index + 1)

    data_repeat = einops.repeat(data, input_repeat_str, k=index.shape[dim_index])
    index_repeat = einops.repeat(index, index_repeat_str, k=data.shape[dim_data])

    out = torch.gather(data_repeat, dim_gather, index_repeat)

    return out

def unsorted_segment_sum(data, index, dim_sum: int, dim_input: int, dim_index: int, n_verts=None):
    input_repeat_str = make_repeat_str(data, dim_input)
    index_repeat_str = make_repeat_str(index, dim_index + 1)    

    data_repeat = einops.repeat(data, input_repeat_str, k=index.shape[dim_index])
    index_repeat = einops.repeat(index, index_repeat_str, k=data.shape[dim_input])

    B = data.shape[:dim_sum]
    n_verts = n_verts or index.max().item() + 1

    out = torch.zeros(*B, n_verts, index.shape[dim_index], data.shape[dim_input]).to(data.device)
    out = out.scatter_add(dim_sum, index_repeat, data_repeat)
    out = out.sum(dim=-2)

    return out

class FaceNormals(nn.Module):
    """
    torch Module that computes face normals for a batch of meshes
    """

    def __init__(self, normalize=True):
        """
        :param normalize: Whether to normalize the face normals
        """

        super().__init__()
        self.normalize = normalize

    def forward(self, vertices, faces):
        """

        :param vertices: FloatTensor of shape (batch_size, num_vertices, 3)
        :param faces: LongTensor of shape (batch_size, num_faces, 3)
        :return: face_normals: FloatTensor of shape (batch_size, num_faces, 3)
        """
        v = vertices
        f = faces

        if v.shape[0] > 1 and f.shape[0] == 1:
            f = f.repeat(v.shape[0], 1, 1)

        v_repeat = einops.repeat(v, 'b m n -> b m k n', k=f.shape[-1])
        f_repeat = einops.repeat(f, 'b m n -> b m n k', k=v.shape[-1])
        triangles = torch.gather(v_repeat, 1, f_repeat)

        # Compute face normals
        v0, v1, v2 = torch.unbind(triangles, dim=-2)
        e1 = v0 - v1
        e2 = v2 - v1
        face_normals = torch.linalg.cross(e2, e1)

        if self.normalize:
            face_normals = F.normalize(face_normals, dim=-1)

        return face_normals

def VertexNormals(v, f):
    triangles = gather(v, f, 0, 1, 1)  # F x 3 x 3
    v0, v1, v2 = torch.unbind(triangles, dim=-2)  # F x 3
    e0 = v1 - v0  # F x 3
    e1 = v2 - v1
    e2 = v0 - v2

    # F x 3
    face_normals = torch.linalg.cross(e0, e1) + torch.linalg.cross(e1, e2) + torch.linalg.cross(e2, e0)

    # V x 3
    vn = unsorted_segment_sum(face_normals, f, 0, 1, 1, n_verts=v.shape[0])
    vn = F.normalize(vn, dim=-1)

    return vn

def find_nearest_points(cloth_vertices, body_vertices):
    # cloth_vertices: Tensor of cloth vertices [num_cloth_verts, 3]
    # body_vertices: Tensor of body vertices [num_body_verts, 3]
    # Returns: Tensor of indices of the nearest body vertices for each cloth vertex

    # Perform KNN search
    nearest_points_idx = knn_points(cloth_vertices.unsqueeze(0), body_vertices.unsqueeze(0), K=1, return_nn=False)
    return nearest_points_idx.idx.squeeze(-1)  # Squeeze to remove the K dimension

def triangles_to_edges(faces: torch.LongTensor):
    """Computes unique mesh edges from triangles."""

    # Collect edges from triangles
    edges_list = [
        faces[:, 0:2],  # Edges from first to second vertex of each face
        faces[:, 1:3],  # Edges from second to third vertex
        torch.stack([faces[:, 2], faces[:, 0]], dim=-1)  # Edges from third to first vertex
    ]
    edges = torch.cat(edges_list, dim=0)

    # Remove duplicate edges
    # Sort each edge for consistent ordering
    edges = torch.sort(edges, dim=1)[0]

    # Remove duplicate edges
    unique_edges = torch.unique(edges, dim=0)

    return unique_edges

def sample_points_from_obj(file_path, num_points=2048):
    mesh = trimesh.load(file_path)
    points, _ = trimesh.sample.sample_surface_even(mesh, num_points)
    return points

def simplify_mesh(input_mesh_path, target_vertex_count):
    # https://stackoverflow.com/questions/65419221/how-to-use-pymeshlab-to-reduce-vertex-number-to-a-certain-number
    # Load the mesh
    ms = ml.MeshSet()
    ms.load_new_mesh(input_mesh_path)
    initial_mesh = ms.current_mesh()
    # print('Input mesh has', initial_mesh.vertex_number(), 'vertices and', initial_mesh.face_number(), 'faces')

    # Clean-up in case the input has duplicate/unreferenced vertices
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()

    # Initial estimation number of faces, designed to be superior to the theoretical value
    numFaces = 100 + ms.current_mesh().face_number() - (ms.current_mesh().vertex_number() - target_vertex_count)

    # Simplify the mesh
    while ms.current_mesh().vertex_number() > target_vertex_count:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=numFaces, preservenormal=True)
        # print("Mesh decimated to", numFaces, "faces contains", ms.current_mesh().vertex_number(), "vertices")
        # Refine our estimation to slowly converge to the target vertex number
        numFaces -= (ms.current_mesh().vertex_number() - target_vertex_count)

    # Final mesh details
    final_mesh = ms.current_mesh()
    # print('Output mesh has', final_mesh.vertex_number(), 'vertices and', final_mesh.face_number(), 'faces')

    # # Save the final mesh as OBJ
    # ms.save_current_mesh(output_mesh_path, save_vertex_normal=True)
    # print('Mesh saved to', output_mesh_path)

    vertices = np.array(final_mesh.vertex_matrix())
    faces = np.array(final_mesh.face_matrix())
    return vertices, faces

def compute_face_normals(vertices, faces, normalize=True):
    """
    Computes face normals for a batch of meshes.

    :param vertices: FloatTensor of shape (batch_size, num_vertices, 3)
    :param faces: LongTensor of shape (batch_size, num_faces, 3)
    :param normalize: Whether to normalize the face normals
    :return: FloatTensor of shape (batch_size, num_faces, 3) representing face normals
    """
    if len(vertices.shape) == 2:
        v = vertices.unsqueeze(0)
        f = faces.unsqueeze(0)
    else:
        v = vertices
        f = faces

    # Repeat faces to match the batch size of vertices, if necessary
    if v.shape[0] > 1 and f.shape[0] == 1:
        f = f.repeat(v.shape[0], 1, 1)

    # Arrange vertices according to faces
    v_repeat = einops.repeat(v, 'b m n -> b m k n', k=f.shape[-1])
    f_repeat = einops.repeat(f, 'b m n -> b m n k', k=v.shape[-1])
    triangles = torch.gather(v_repeat, 1, f_repeat)

    # Compute face normals
    v0, v1, v2 = torch.unbind(triangles, dim=-2)
    e1 = v0 - v1
    e2 = v2 - v1
    face_normals = torch.linalg.cross(e2, e1)

    # Normalize face normals, if specified
    if normalize:
        face_normals = F.normalize(face_normals, dim=-1)

    return face_normals

    
def get_face_areas(vertices, faces):
    """
    Computes the area of each face in the mesh

    :param vertices: FloatTensor or numpy array of shape (num_vertices, 3)
    :param faces: LongTensor or numpy array of shape (num_faces, 3)
    :return: areas: FloatTensor or numpy array of shape (num_faces,)
    """
    if type(vertices) == torch.Tensor:
        vertices = vertices.detach().cpu().numpy()

    if type(faces) == torch.Tensor:
        faces = faces.detach().cpu().numpy()
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    u = v2 - v0
    v = v1 - v0

    if u.shape[-1] == 2:
        out = np.abs(np.cross(u, v)) / 2.0
    else:
        out = np.linalg.norm(np.cross(u, v), axis=-1) / 2.0
    return out
    
def make_f_area(v, f, device):
    """
    Compute areas of each face
    :param v: vertex positions [Vx3]
    :param f: faces [Fx3]
    :param device: pytorch device
    :return: face areas [Fx1]
    """
    f_area = torch.FloatTensor(get_face_areas(v, f)).to(device)  # +
    f_area = f_area.unsqueeze(-1)
    return f_area

def get_vertex_connectivity(faces):
    '''
    Returns a list of unique edges in the mesh.
    Each edge contains the indices of the vertices it connects
    '''
    device = 'cpu'
    if type(faces) == torch.Tensor:
        device = faces.device
        faces = faces.detach().cpu().numpy()

    edges = set()
    for f in faces:
        num_vertices = len(f)
        for i in range(num_vertices):
            j = (i + 1) % num_vertices
            edges.add(tuple(sorted([f[i], f[j]])))

    edges = torch.LongTensor(list(edges)).to(device)
    return edges

def get_face_connectivity_combined(faces):
    """
    Finds the faces that are connected in a mesh
    :param faces: LongTensor of shape (num_faces, 3)
    :return: adjacent_faces: pairs of face indices LongTensor of shape (num_edges, 2)
    :return: adjacent_face_edges: pairs of node indices that comprise the edges connecting the corresponding faces
     LongTensor of shape (num_edges, 2)
    """

    device = 'cpu'
    if type(faces) == torch.Tensor:
        device = faces.device
        faces = faces.detach().cpu().numpy()

    edges = get_vertex_connectivity(faces).cpu().numpy()

    G = {tuple(e): [] for e in edges}
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e] += [i]

    adjacent_faces = []
    adjacent_face_edges = []

    for key in G:
        if len(G[key]) >= 3:
            G[key] = G[key][:2]
        if len(G[key]) == 2:
            adjacent_faces += [G[key]]
            adjacent_face_edges += [list(key)]

    adjacent_faces = torch.LongTensor(adjacent_faces).to(device)
    adjacent_face_edges = torch.LongTensor(adjacent_face_edges).to(device)

    return adjacent_faces, adjacent_face_edges

def load_obj(filename, tex_coords=False):
    """
    Load a mesh from an obj file

    :param filename: path to the obj file
    :param tex_coords: whether to load texture (UV) coordinates
    :return: vertices: numpy array of shape (num_vertices, 3)
    :return: faces: numpy array of shape (num_faces, 3)
    """
    vertices = []
    faces = []
    uvs = []
    faces_uv = []

    with open(filename, 'r') as fp:
        for line in fp:
            line_split = line.split()

            if not line_split:
                continue

            elif tex_coords and line_split[0] == 'vt':
                uvs.append([line_split[1], line_split[2]])

            elif line_split[0] == 'v':
                vertices.append([line_split[1], line_split[2], line_split[3]])

            elif line_split[0] == 'f':
                vertex_indices = [s.split("/")[0] for s in line_split[1:]]
                faces.append(vertex_indices)

                if tex_coords:
                    uv_indices = [s.split("/")[1] for s in line_split[1:]]
                    faces_uv.append(uv_indices)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32) - 1

    if tex_coords:
        uvs = np.array(uvs, dtype=np.float32)
        faces_uv = np.array(faces_uv, dtype=np.int32) - 1
        return vertices, faces, uvs, faces_uv

    return vertices, faces

def save_obj(vertices, faces, file_path):
    with open(file_path, 'w') as f:
        for vertex in vertices:
            f.write(f'v {vertex[0].item()} {vertex[1].item()} {vertex[2].item()}\n')
        for face in faces:
            f.write(f'f {face[0].item() + 1} {face[1].item() + 1} {face[2].item() + 1}\n')


def compute_covariance_matrix(vertex_positions, neighbors_dict):
    # the covariance matrix for each vertex
    covariance_matrices = {}

    for i, neighbors in neighbors_dict.items():
        # Extract the positions of the neighboring vertices
        neighbor_positions = vertex_positions[neighbors]

        # Compute the mean position of the neighboring vertices
        mean_position = neighbor_positions.mean(dim=0)

        # Center the neighbor positions by subtracting the mean position
        centered_positions = neighbor_positions - mean_position

        # Compute the covariance matrix using the outer product of the centered positions
        covariance_matrix = centered_positions.T @ centered_positions / neighbors.size(0)

        # Store the covariance matrix for vertex i
        covariance_matrices[i] = covariance_matrix

    return covariance_matrices

# def RQ_curvature(single_vertex_position, single_covariance_matrices):
#     # single_vertex_position shape is [1, 3]
#     RQ = (single_vertex_position @ single_covariance_matrices @ single_vertex_position.T)/(single_vertex_position @ single_vertex_position.T)
#     return RQ

# def compute_RQ_curvature_feature(vertex_positions, neighbors_dict, covariance_matrices):
#     RQ = {}
#     for i in range(vertex_positions.shape[0]):
#         for j in neighbors_dict[i]:
#             RQ[(i, j)] = RQ_curvature(vertex_positions[j].unsqueeze(0), covariance_matrices[i])

def RQ_curvature(single_vertex_position, single_covariance_matrix):
    numerator = single_vertex_position @ single_covariance_matrix @ single_vertex_position.T
    denominator = single_vertex_position @ single_vertex_position.T
    RQ = numerator / denominator
    return RQ.squeeze()

def compute_RQ_curvature_feature(vertex_positions, neighbors_dict):
    # RQ_min_curvatures = []
    # RQ_max_curvatures = []
    # initialize RQ_min_curvatures and RQ_max_curvatures as empty tensor arrays with shape [num_vertices, 1]
    RQ_min_curvatures = torch.empty(vertex_positions.shape[0], 1)
    RQ_max_curvatures = torch.empty(vertex_positions.shape[0], 1)

    for i, neighbors in neighbors_dict.items():
        # Calculate the mean position for vertex i's neighborhood
        neighbor_positions = vertex_positions[neighbors]
        mean_position = neighbor_positions.mean(dim=0)
        
        # Subtract mean position to centralize the neighbors
        centralized_positions = neighbor_positions - mean_position
        
        # Compute the covariance matrix for the centralized positions
        cov_matrix = centralized_positions.T @ centralized_positions / len(neighbors)
        
        # Compute RQ for vertex i with respect to all neighbors j and find min and max
        RQ_values = []
        for j in neighbors:
            centralized_single_vertex_position = (vertex_positions[j] - mean_position).unsqueeze(0)
            RQ = RQ_curvature(centralized_single_vertex_position, cov_matrix)
            RQ_values.append(RQ.item())

        # Save min and max RQ values for vertex i
        # RQ_min_curvatures.append(torch.tensor(min(RQ_values)))
        # RQ_max_curvatures.append(torch.tensor(max(RQ_values)))
        RQ_min_curvatures[i] = torch.tensor(min(RQ_values))
        RQ_max_curvatures[i] = torch.tensor(max(RQ_values))

    return RQ_min_curvatures, RQ_max_curvatures

def get_measurements_from_smplx(body_vertices, indicator = 'top'):
    # body_vertices: Tensor of body vertices [num_body_verts, 3]
    # Returns: Dictionary of anthropometric measurements
    measurer = MeasureBody('smplx')
    measurer.from_verts(verts=body_vertices) 
    measurement_names = measurer.all_possible_measurements
    measurer.measure(measurement_names) 
    measurements = measurer.measurements
    # if indicator == 'top':
    #     return measurements['arm_right_length'], measurements['chest_circumference'], measurements['waist_circumference']
    # else:
    #     return measurements['hip_circumference'], measurements['thigh_circumference'], measurements['calf_circumference']
    # if indicator == 'top':
    #     # stack them together
    #     return np.stack((measurements['height'],measurements['arm right length'], measurements['chest circumference'], measurements['waist circumference'],
    #                      measurements['wrist right circumference'], measurements['bicep right circumference'], measurements['forearm right circumference']))
    # else:
    #     return np.stack((measurements['height'], measurements['inside leg height'], measurements['waist circumference'], measurements['hip circumference'], 
    #                      measurements['thigh left circumference'], measurements['calf left circumference'], measurements['ankle left circumference']))
    if indicator == 'top':
        # stack them together
        return np.stack((measurements['chest circumference'], measurements['waist circumference'],
                         measurements['wrist right circumference'], measurements['bicep right circumference'], measurements['forearm right circumference']))
    else:
        return np.stack((measurements['waist circumference'], measurements['hip circumference'], 
                         measurements['thigh left circumference'], measurements['calf left circumference'], measurements['ankle left circumference']))

# def get_measurements_from_smplx(body_vertices, indicator = 'top'):
#     # body_vertices: Tensor of body vertices [num_body_verts, 3]
#     # Returns: Dictionary of anthropometric measurements
#     measurer = MeasureBody('smplx')
#     measurer.from_verts(verts=body_vertices) 
#     measurement_names = measurer.all_possible_measurements
#     measurer.measure(measurement_names) 
#     measurements = measurer.measurements
#     # if indicator == 'top':
#     #     return measurements['arm_right_length'], measurements['chest_circumference'], measurements['waist_circumference']
#     # else:
#     #     return measurements['hip_circumference'], measurements['thigh_circumference'], measurements['calf_circumference']
#     if indicator == 'top':
#         # stack them together
#         return np.stack((measurements['arm right length'], measurements['chest circumference'], measurements['waist circumference']))
#     else:
#         return np.stack((measurements['hip circumference'], measurements['thigh circumference'], measurements['calf circumference']))

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def add_labels(sampled_vertices, original_vertices, body_part_indices):
    """
    Assigns labels to sampled vertices based on the nearest original vertex associated with a specific body part.
    
    Parameters:
    - sampled_vertices: np.array, the vertices to which labels will be assigned.
    - original_vertices: np.array, the vertices from which labels are sourced.
    - body_part_indices: dict, keys are body part names and values are lists of indices of original_vertices associated with each body part.
    
    Returns:
    - List of labels for each sampled vertex, indicating the body part it is closest to.
    """
    # Prepare a reverse mapping from vertex index to body part label
    vertex_to_label = {}
    for label, indices in body_part_indices.items():
        for index in indices:
            vertex_to_label[index] = label
    
    # Find the nearest original vertex for each sampled vertex
    original_indices = find_nearest_points(sampled_vertices, original_vertices)[0]
    sampled_labels = []
    
    # Assign labels based on the nearest original vertex
    for idx in original_indices:
        idx = int(idx)
        if idx in vertex_to_label:
            sampled_labels.append(vertex_to_label[idx])
        else:
            sampled_labels.append("undefined")  # For vertices not covered by the body_part_indices
    return sampled_labels

def one_hot_emb(original_vertices, sampled_vertices, json_file):
    data_json = read_json(json_file)
    sampled_labels = add_labels(sampled_vertices, original_vertices, data_json)
    body_parts = list(data_json.keys())
    # Map each label to an index
    label_to_index = {label: idx for idx, label in enumerate(body_parts)}
    # Initialize an empty list to hold the one-hot encoded features
    one_hot_encoded_features = []

    # Generate a one-hot encoded vector for each label
    for label in sampled_labels:
        # Initialize a vector of zeros
        one_hot_vector = [0] * len(body_parts)

        # Set the appropriate element to 1
        if label in label_to_index:
            one_hot_vector[label_to_index[label]] = 1
        else:
            # Handle labels that might not be in the body_parts list, if necessary
            pass

        # Add the one-hot vector to the list of features
        one_hot_encoded_features.append(one_hot_vector)

    return one_hot_encoded_features

def one_hot_emb_smpl(json_file):
    # Load data from JSON file
    data_json = read_json(json_file)
    body_parts = list(data_json.keys())

    # Map each label to an index
    label_to_index = {label: idx for idx, label in enumerate(body_parts)}

    # Total number of categories including a category for 'other'
    num_categories = len(label_to_index)

    # Initialize an empty array for one-hot encoded features
    one_hot_encoded_features = np.zeros((6890, num_categories))


    # Create a dictionary from vertex index to part index
    vertex_to_part_index = {}
    for part, indices in data_json.items():
        for index in indices:
            vertex_to_part_index[index] = label_to_index[part]

    # Generate a one-hot encoded vector for each vertex
    for i in range(6890):
        if i in vertex_to_part_index:
            part_idx = vertex_to_part_index[i]
        else:
            part_idx = num_categories - 1  # Assign to 'other' category

        one_hot_encoded_features[i, part_idx] = 1

    return one_hot_encoded_features

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

# def build_adjacency_matrix(vertices, faces):
#     num_vertices = len(vertices)
#     adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

#     for face in faces:
#         for i in range(len(face)):
#             for j in range(i + 1, len(face)):
#                 v1 = face[i]
#                 v2 = face[j]
#                 adj_matrix[v1, v2] = 1
#                 adj_matrix[v2, v1] = 1  # because the graph is undirected

#     return adj_matrix

def build_adjacency_matrix(vertices, edges):
    """
    Build adjacency matrix for a given mesh represented by vertices and edges.

    Args:
    - num_vertices (int): Number of vertices in the mesh.
    - edges (Tensor): Tensor of edges, each edge represented by indices of its two vertices, shape (E, 2), where E is the number of edges.

    Returns:
    - Sparse Tensor representing the adjacency matrix of shape (V, V), where V is the number of vertices.
    """
    # Ensure the edges include both directions since the graph is undirected
    num_vertices = vertices.shape[0]
    directed_edges = torch.cat([edges, edges[:, [1, 0]]], dim=0)

    # Remove duplicates which may have been added
    directed_edges = directed_edges.unique(dim=0)

    # Get rows and columns indices for the sparse matrix
    rows = directed_edges[:, 0]
    cols = directed_edges[:, 1]
    values = torch.ones_like(rows).float()  # Assuming unweighted edges

    # Create a sparse matrix
    size = (num_vertices, num_vertices)
    adj_matrix = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, size)
    
    return adj_matrix

def shrink_cloth(cloth_verts, body_verts, body_faces, correspondence, distance=0.017): #0.17
    # Get normals of every body verts
    # turn body_faces into dtype int 64 as tensor
    body_faces = torch.tensor(body_faces, dtype=torch.int64)
    body_verts = torch.tensor(body_verts, dtype=torch.float32)
    body_normals = VertexNormals(body_verts, body_faces)
    cloth_corresponding_normals = body_normals[correspondence].to(cloth_verts.device)
    # Shrink the cloth vertices along the opposite direction of normals
    cloth_verts -= cloth_corresponding_normals * distance

    return cloth_verts

# def dilate_cloth(cloth_verts, body_verts, body_faces, correspondence, distance=0.005):
#     # Get normals of every body verts
#     # turn body_faces into dtype int 64 as tensor
#     body_faces = torch.tensor(body_faces, dtype=torch.int64)
#     body_verts = torch.tensor(body_verts, dtype=torch.float32)
#     body_normals = VertexNormals(body_verts, body_faces)
#     cloth_corresponding_normals = body_normals[correspondence].to(cloth_verts.device)
#     # Shrink the cloth vertices along the opposite direction of normals
#     cloth_verts += cloth_corresponding_normals * distance

#     return cloth_verts

# dilate 3d object by its own normal
def dilate_cloth(cloth_verts, cloth_faces, distance=0.008):
    # cloth_faces = torch.tensor(cloth_faces, dtype=torch.int64)
    # cloth_verts = torch.tensor(cloth_verts, dtype=torch.float32)
    cloth_faces = torch.as_tensor(cloth_faces, dtype=torch.int64)
    cloth_verts = torch.as_tensor(cloth_verts, dtype=torch.float32)

    cloth_normals = VertexNormals(cloth_verts, cloth_faces)
    cloth_verts += cloth_normals * distance

    return cloth_verts

def shrink(verts, faces, distance=0.02):
    faces = torch.tensor(faces, dtype=torch.int64)
    verts = torch.tensor(verts, dtype=torch.float32)
    cloth_normals = VertexNormals(verts, faces)
    verts -= cloth_normals * distance

    return verts

# def optimize_smpl_surface(smpl_verts, smpl_faces, original_verts):

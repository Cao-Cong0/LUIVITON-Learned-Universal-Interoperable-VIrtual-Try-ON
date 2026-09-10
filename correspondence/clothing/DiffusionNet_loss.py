import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial import cKDTree
from utils import VertexNormals, find_nearest_points, triangles_to_edges, compute_face_normals, gather, compute_RQ_curvature_feature
from sklearn.metrics import label_ranking_loss
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import json

SEGMENTATION_PATH = Path(__file__).resolve().parents[2] / "models" / "smpl_vert_segmentation.json"


def export_to_ply(vertices, file_path):
    vertices_np = vertices.cpu().detach().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices_np)

    o3d.io.write_point_cloud(file_path, pcd)


def RankingLoss(predicted_scores, gt_top_3_indices, margin):
    num_garments, num_body = predicted_scores.shape
    device = predicted_scores.device
    y_true = torch.zeros((num_garments, num_body), dtype=torch.float32, device=device)
    for i in range(num_garments):
        y_true[i, gt_top_3_indices[i]] = 1

    loss_function = nn.MultiLabelSoftMarginLoss()

    loss = loss_function(predicted_scores, y_true)
    return loss


def InnerDistanceLoss(predicted_scores, body_vertices):
    probabilities = torch.softmax(predicted_scores, dim=1)

    weighted_positions = torch.matmul(probabilities, body_vertices)

    distances = torch.norm(body_vertices.unsqueeze(1) - weighted_positions.unsqueeze(0), dim=2)
    weighted_distances = torch.matmul(probabilities, distances)
    mean_distance_loss = weighted_distances.mean()

    return mean_distance_loss


def gtDistanceLoss(predicted_scores, body_vertices, garment_gt_top_3_indices):
    weights = torch.softmax(predicted_scores, dim=1)
    weighted_positions = torch.matmul(weights, body_vertices)
    ground_truth_positions = body_vertices[garment_gt_top_3_indices].mean(dim=1)
    loss = torch.norm(weighted_positions - ground_truth_positions, dim=1).mean()
    return loss


def Part_Seg_loss(pred_coordinates, body_uvs, garment_gt_top_3_indices):
    pred = pred_coordinates*0.5
    pred = pred.cpu().detach().numpy()
    uv_tree = KDTree(body_uvs.cpu().numpy())
    _, top_1_indices = uv_tree.query(pred, k=1)

    json_file = os.environ.get("LUVITON_SMPL_SEGMENTATION", str(SEGMENTATION_PATH))

    with open(json_file, 'r') as file:
        data = json.load(file)

    index_label_map = {}
    for label, indices in data.items():
        for index in indices:
            index_label_map[index] = label

    part_labels = list(data.keys())
    num_classes = len(part_labels)

    label_to_int = {label: idx for idx, label in enumerate(part_labels)}

    ground_truth_indices = garment_gt_top_3_indices.cpu().numpy().flatten().tolist()
    predicted_indices = top_1_indices.flatten().tolist()

    ground_truth_labels = [label_to_int[index_label_map[idx]] for idx in ground_truth_indices]
    predicted_labels = [label_to_int[index_label_map[idx]] for idx in predicted_indices]

    ground_truth_tensor = torch.tensor(ground_truth_labels, dtype=torch.long)
    predicted_tensor = torch.tensor(predicted_labels, dtype=torch.long)

    ground_truth_one_hot = F.one_hot(ground_truth_tensor, num_classes=num_classes).float()
    predicted_one_hot = F.one_hot(predicted_tensor, num_classes=num_classes).float()

    loss = F.mse_loss(predicted_one_hot, ground_truth_one_hot)

    return loss


def NeighborDistanceLoss(predicted_scores, body_vertices, garment_edge_index):
    probabilities = torch.softmax(predicted_scores, dim=1)

    weighted_positions = torch.matmul(probabilities, body_vertices)

    edge_index = garment_edge_index
    source_indices = edge_index[0]
    target_indices = edge_index[1]

    source_positions = weighted_positions[source_indices]
    target_positions = weighted_positions[target_indices]

    neighbor_distances = torch.norm(source_positions - target_positions, dim=1)

    mean_distance_loss = neighbor_distances.mean()

    return mean_distance_loss


def BCEWithLogitsLoss(preds, one_hot_gt):
    loss_function = nn.BCEWithLogitsLoss()
    loss = loss_function(preds, one_hot_gt)

    return loss.item()


def GT_uv_loss(pred_coordinates, body_uvs, garment_gt_top_3_indices):
    top_3_indices_uv = body_uvs[garment_gt_top_3_indices].mean(dim=1)

    # Match tanh predictions to the centered template UV range [-0.5, 0.5].
    norm_loss = torch.norm(pred_coordinates * 0.5 - top_3_indices_uv, dim=1).sum()

    return norm_loss


def neighboring_loss(pred_coordinates, garment_edge_index):
    source_indices = garment_edge_index[0]
    target_indices = garment_edge_index[1]
    source_coordinates = pred_coordinates[source_indices]
    target_coordinates = pred_coordinates[target_indices]

    neighbor_distances = torch.norm(source_coordinates - target_coordinates, dim=1)

    mean_distance_loss = neighbor_distances.mean()

    return mean_distance_loss


def total_loss_function(pred_coordinates, body_vertices, body_uvs, garment_gt_top_3_indices, garment_edge_index, margin=None):
    device = pred_coordinates.device

    body_vertices = body_vertices.to(device)
    garment_gt_top_3_indices = garment_gt_top_3_indices.to(device)
    garment_edge_index = garment_edge_index.to(device)
    body_uvs = body_uvs.to(device)

    gt_uv_loss = GT_uv_loss(pred_coordinates, body_uvs, garment_gt_top_3_indices)

    losses = {'GT_uv_loss': gt_uv_loss}
    total_loss = gt_uv_loss

    return total_loss, losses

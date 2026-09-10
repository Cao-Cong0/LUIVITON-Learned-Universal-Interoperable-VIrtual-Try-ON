import shutil
import os
import sys
import random
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFUSION_NET_SRC = PROJECT_ROOT / "third_party" / "diffusion_net" / "src"
SHARED_REGISTRATION_ROOT = PROJECT_ROOT / "smpl_registration"
sys.path.insert(0, str(DIFFUSION_NET_SRC))
sys.path.insert(0, str(SHARED_REGISTRATION_ROOT))
import diffusion_net
from diffusion_net.utils import toNP
from utils import triangles_to_edges, simplify_mesh, get_face_connectivity_combined, make_f_area, get_face_areas, load_obj, compute_covariance_matrix, compute_RQ_curvature_feature, one_hot_emb_smpl, extract_vertices_and_uvs
import matplotlib.pyplot as plt
from tqdm import tqdm


class ClothMeshDataset(Dataset):

    def __init__(self, root_dir, train, k_eig, use_cache=True, op_cache_dir=None):
        self.train = train
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 6890

        self.verts_list = []
        self.faces_list = []
        self.labels_list = []
        self.gt_verts_list = []
        self.body_verts = []
        self.filename_list = []
        self.garment_edges_list = []
        self.garment_original_verts_list = []

        input_path = os.path.join(root_dir, "input")
        gt_path = os.path.join(root_dir, "gt")
        body_file_path = os.path.join(root_dir, "smpl_uv_free.obj")

        this_files = os.listdir(input_path)

        for f in this_files:
            print(f)
            input_file = os.path.join(input_path, f)
            gt_file = os.path.join(gt_path, f)
            verts, faces = pp3d.read_mesh(input_file)
            gt_verts, gt_faces = pp3d.read_mesh(gt_file)
            _, body_faces = pp3d.read_mesh(body_file_path)
            body_verts, body_uvs = extract_vertices_and_uvs(body_file_path)

            verts = torch.tensor(verts).float()
            original_verts = verts.clone()  # Preserve input coordinates for export.
            faces = torch.tensor(faces)
            egdes = triangles_to_edges(faces)
            edges = egdes.clone().detach() if torch.is_tensor(egdes) else torch.as_tensor(egdes)

            gt_verts = torch.tensor(gt_verts).float()
            gt_faces = torch.tensor(gt_faces)

            body_verts = torch.tensor(body_verts).float()
            body_faces = torch.tensor(body_faces)
            body_uvs = torch.tensor(body_uvs).float()

            distances = torch.cdist(gt_verts, body_verts)
            top_3_distances, top_3_indices = torch.topk(distances, 1, largest=False, dim=1)
            labels = top_3_indices

            # Center the bounding box and normalize the maximum vertex radius.
            verts = diffusion_net.geometry.normalize_positions(verts, method='bbox',scale_method='max_rad')

            file_name = os.path.splitext(f)[0]

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.labels_list.append(labels)
            self.gt_verts_list.append(gt_verts)
            self.filename_list.append(file_name)
            self.garment_edges_list.append(edges)
            self.garment_original_verts_list.append(original_verts)

        self.body_verts = body_verts
        self.body_uvs = body_uvs
        self.body_faces = body_faces

        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.labels_list[idx], self.gt_verts_list[idx], self.body_verts, self.filename_list[idx], self.garment_edges_list[idx], self.body_uvs, self.body_faces, self.garment_original_verts_list[idx]


class ClothMeshDataset_eval(Dataset):

    def __init__(self, root_dir, train, k_eig, use_cache=True, op_cache_dir=None):
        self.train = train
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 6890

        self.verts_list = []
        self.faces_list = []
        self.body_verts = []
        self.filename_list = []
        self.garment_edges_list = []
        self.garment_original_verts_list = []

        input_path = os.path.join(root_dir, "input")
        body_file_path = os.path.join(root_dir, "smpl_uv_free.obj")

        this_files = os.listdir(input_path)

        for f in this_files:
            print(f)
            input_file = os.path.join(input_path, f)
            verts, faces = pp3d.read_mesh(input_file)
            _, body_faces = pp3d.read_mesh(body_file_path)
            body_verts, body_uvs = extract_vertices_and_uvs(body_file_path)

            verts = torch.tensor(verts).float()
            original_verts = verts.clone()  # Preserve input coordinates for export.
            faces = torch.tensor(faces)
            egdes = triangles_to_edges(faces)
            edges = egdes.clone().detach() if torch.is_tensor(egdes) else torch.as_tensor(egdes)

            body_verts = torch.tensor(body_verts).float()
            body_faces = torch.tensor(body_faces)
            body_uvs = torch.tensor(body_uvs).float()

            # Center the bounding box and normalize the maximum vertex radius.
            verts = diffusion_net.geometry.normalize_positions(verts, method='bbox',scale_method='max_rad')

            file_name = os.path.splitext(f)[0]

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.filename_list.append(file_name)
            self.garment_edges_list.append(edges)
            self.garment_original_verts_list.append(original_verts)

        self.body_verts = body_verts
        self.body_uvs = body_uvs
        self.body_faces = body_faces

        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.body_verts, self.filename_list[idx], self.garment_edges_list[idx], self.body_uvs, self.body_faces, self.garment_original_verts_list[idx]


class ClothMeshDataset_single_eval(Dataset):

    def __init__(self, file_path, train, k_eig, use_cache=True, op_cache_dir=None):
        self.train = train
        self.root_dir = file_path
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir
        self.n_class = 6890

        self.verts_list = []
        self.faces_list = []
        self.body_verts = []
        self.filename_list = []
        self.garment_edges_list = []
        self.garment_original_verts_list = []

        input_file = file_path

        body_file_path = os.environ.get("LUVITON_BODY_TEMPLATE")
        if not body_file_path:
            raise ValueError("LUVITON_BODY_TEMPLATE must point to the SMPL UV template OBJ")

        verts, faces = pp3d.read_mesh(input_file)
        _, body_faces = pp3d.read_mesh(body_file_path)
        body_verts, body_uvs = extract_vertices_and_uvs(body_file_path)

        verts = torch.tensor(verts).float()
        original_verts = verts.clone()  # Preserve input coordinates for export.
        faces = torch.tensor(faces)
        egdes = triangles_to_edges(faces)
        edges = egdes.clone().detach() if torch.is_tensor(egdes) else torch.as_tensor(egdes)

        body_verts = torch.tensor(body_verts).float()
        body_faces = torch.tensor(body_faces)
        body_uvs = torch.tensor(body_uvs).float()

        # Center the bounding box and normalize the maximum vertex radius.
        verts = diffusion_net.geometry.normalize_positions(verts, method='bbox',scale_method='max_rad')
        file_name = os.path.splitext(os.path.basename(input_file))[0]

        self.verts_list.append(verts)
        self.faces_list.append(faces)
        self.filename_list.append(file_name)
        self.garment_edges_list.append(edges)
        self.garment_original_verts_list.append(original_verts)

        self.body_verts = body_verts
        self.body_uvs = body_uvs
        self.body_faces = body_faces

        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.body_verts, self.filename_list[idx], self.garment_edges_list[idx], self.body_uvs, self.body_faces, self.garment_original_verts_list[idx]


class BodyMeshDataset(Dataset):

    def __init__(self, root_dir, train, k_eig, use_cache=True, op_cache_dir=None):

        self.train = train
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 6890

        self.verts_list = []
        self.faces_list = []
        self.labels_list = []
        self.gt_verts_list = []
        self.body_verts = []
        self.filename_list = []
        self.garment_edges_list = []
        self.garment_original_verts_list = []

        input_path = os.path.join(root_dir, "input")
        gt_path = os.path.join(root_dir, "gt")
        body_file_path = os.path.join(root_dir, "smpl_uv_free.obj")

        this_files = os.listdir(input_path)

        for f in this_files:
            input_file = os.path.join(input_path, f)
            gt_file = os.path.join(gt_path, f)

            verts, faces = pp3d.read_mesh(input_file)
            gt_verts, gt_faces = pp3d.read_mesh(gt_file)
            _, body_faces = pp3d.read_mesh(body_file_path)
            body_verts, body_uvs = extract_vertices_and_uvs(body_file_path)

            verts = torch.tensor(verts).float()
            original_verts = verts.clone()  # Preserve input coordinates for export.
            faces = torch.tensor(faces)
            egdes = triangles_to_edges(faces)
            edges = egdes.clone().detach() if torch.is_tensor(egdes) else torch.as_tensor(egdes)

            gt_verts = torch.tensor(gt_verts).float()
            gt_faces = torch.tensor(gt_faces)

            body_verts = torch.tensor(body_verts).float()
            body_faces = torch.tensor(body_faces)
            body_uvs = torch.tensor(body_uvs).float()

            distances = torch.cdist(verts,gt_verts)
            top_1_distances, top_1_indices = torch.topk(distances, 1, largest=False, dim=1)
            labels = top_1_indices

            # Center the bounding box and normalize the maximum vertex radius.
            verts = diffusion_net.geometry.normalize_positions(verts, method='bbox',scale_method='max_rad')

            file_name = os.path.splitext(f)[0]

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.labels_list.append(labels)
            self.gt_verts_list.append(gt_verts)
            self.filename_list.append(file_name)
            self.garment_edges_list.append(edges)
            self.garment_original_verts_list.append(original_verts)

        self.body_verts = body_verts
        self.body_uvs = body_uvs
        self.body_faces = body_faces

        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.labels_list[idx], self.gt_verts_list[idx], self.body_verts, self.filename_list[idx], self.garment_edges_list[idx], self.body_uvs, self.body_faces, self.garment_original_verts_list[idx]


class BodyMeshDataset_eval(Dataset):
    def __init__(self, root_dir, train, k_eig, use_cache=True, op_cache_dir=None):
        self.train = train
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 6890
        self.verts_list = []
        self.faces_list = []
        self.body_verts = []
        self.filename_list = []
        self.garment_edges_list = []
        self.garment_original_verts_list = []
        input_path = os.path.join(root_dir, "input")
        body_file_path = os.path.join(root_dir, "smpl_uv_free.obj")
        this_files = os.listdir(input_path)
        for f in this_files:
            print(f)
            input_file = os.path.join(input_path, f)
            verts, faces = pp3d.read_mesh(input_file)
            _, body_faces = pp3d.read_mesh(body_file_path)
            body_verts, body_uvs = extract_vertices_and_uvs(body_file_path)
            verts = torch.tensor(verts).float()
            original_verts = verts.clone()  # Preserve input coordinates for export.
            faces = torch.tensor(faces)
            egdes = triangles_to_edges(faces)
            edges = egdes.clone().detach() if torch.is_tensor(egdes) else torch.as_tensor(egdes)
            body_verts = torch.tensor(body_verts).float()
            body_faces = torch.tensor(body_faces)
            body_uvs = torch.tensor(body_uvs).float()
            # Center the bounding box and normalize the maximum vertex radius.
            verts = diffusion_net.geometry.normalize_positions(verts, method='bbox',scale_method='max_rad')
            file_name = os.path.splitext(f)[0]
            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.filename_list.append(file_name)
            self.garment_edges_list.append(edges)
            self.garment_original_verts_list.append(original_verts)
        self.body_verts = body_verts
        self.body_uvs = body_uvs
        self.body_faces = body_faces
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
    def __len__(self):
        return len(self.verts_list)
    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.body_verts, self.filename_list[idx], self.garment_edges_list[idx], self.body_uvs, self.body_faces, self.garment_original_verts_list[idx]


class BodyMeshDataset_single_eval(Dataset):

        def __init__(self, file_path, train, k_eig, use_cache=True, op_cache_dir=None):
            self.train = train
            self.root_dir = file_path
            self.k_eig = k_eig
            self.op_cache_dir = op_cache_dir
            self.n_class = 6890

            self.verts_list = []
            self.faces_list = []
            self.body_verts = []
            self.filename_list = []
            self.garment_edges_list = []
            self.garment_original_verts_list = []

            input_file = file_path
            body_file_path = os.environ.get("LUVITON_BODY_TEMPLATE")
            if not body_file_path:
                raise ValueError("LUVITON_BODY_TEMPLATE must point to the SMPL UV template OBJ")

            verts, faces = pp3d.read_mesh(input_file)
            _, body_faces = pp3d.read_mesh(body_file_path)
            body_verts, body_uvs = extract_vertices_and_uvs(body_file_path)

            verts = torch.tensor(verts).float()
            original_verts = verts.clone()  # Preserve input coordinates for export.
            faces = torch.tensor(faces)
            egdes = triangles_to_edges(faces)
            edges = egdes.clone().detach() if torch.is_tensor(egdes) else torch.as_tensor(egdes)

            body_verts = torch.tensor(body_verts).float()
            body_faces = torch.tensor(body_faces)
            body_uvs = torch.tensor(body_uvs).float()

            # Center the bounding box and normalize the maximum vertex radius.
            verts = diffusion_net.geometry.normalize_positions(verts, method='bbox',scale_method='max_rad')

            file_name = os.path.splitext(os.path.basename(input_file))[0]

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.filename_list.append(file_name)
            self.garment_edges_list.append(edges)
            self.garment_original_verts_list.append(original_verts)

            self.body_verts = body_verts
            self.body_uvs = body_uvs
            self.body_faces = body_faces

            self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        def __len__(self):
            return len(self.verts_list)

        def __getitem__(self, idx):
            return self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx], self.body_verts, self.filename_list[idx], self.garment_edges_list[idx], self.body_uvs, self.body_faces, self.garment_original_verts_list[idx]


class BodyMeshPoseDataset(Dataset):

    def __init__(self, root_dir, train, k_eig, use_cache=True, op_cache_dir=None):

        self.train = train
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.cache_dir = os.path.join(root_dir, "cache")
        self.op_cache_dir = op_cache_dir
        self.n_class = 6890

        self.verts_list = []
        self.faces_list = []
        self.labels_list = []
        self.gt_verts_list = []
        self.body_verts = []
        self.filename_list = []
        self.input_edges_list = []
        self.input_original_verts_list = []

        self.verts_list_pose = []
        self.faces_list_pose = []
        self.labels_list_pose = []
        self.gt_verts_list_pose = []
        self.filename_list_pose = []
        self.input_edges_list_pose = []
        self.input_original_verts_list_pose = []

        input_path = os.path.join(root_dir, "input")
        input_pose_path = os.path.join(root_dir, "input_pose")
        gt_path = os.path.join(root_dir, "gt")
        body_file_path = os.path.join(root_dir, "smpl_uv_free.obj")

        this_files = os.listdir(input_path)

        for f in this_files:
            input_file = os.path.join(input_path, f)
            gt_file = os.path.join(gt_path, f)

            verts, faces = pp3d.read_mesh(input_file)
            gt_verts, gt_faces = pp3d.read_mesh(gt_file)
            _, body_faces = pp3d.read_mesh(body_file_path)
            body_verts, body_uvs = extract_vertices_and_uvs(body_file_path)

            verts = torch.tensor(verts).float()
            original_verts = verts.clone()  # Preserve input coordinates for export.
            faces = torch.tensor(faces)
            egdes = triangles_to_edges(faces)
            edges = egdes.clone().detach() if torch.is_tensor(egdes) else torch.as_tensor(egdes)

            gt_verts = torch.tensor(gt_verts).float()
            gt_faces = torch.tensor(gt_faces)

            body_verts = torch.tensor(body_verts).float()
            body_faces = torch.tensor(body_faces)
            body_uvs = torch.tensor(body_uvs).float()

            distances = torch.cdist(verts,gt_verts)
            top_1_distances, top_1_indices = torch.topk(distances, 1, largest=False, dim=1)
            labels = top_1_indices

            # Center the bounding box and normalize the maximum vertex radius.
            verts = diffusion_net.geometry.normalize_positions(verts, method='bbox',scale_method='max_rad')

            file_name = os.path.splitext(f)[0]

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.labels_list.append(labels)
            self.gt_verts_list.append(gt_verts)
            self.filename_list.append(file_name)
            self.input_edges_list.append(edges)
            self.input_original_verts_list.append(original_verts)

        self.body_verts = body_verts
        self.body_uvs = body_uvs
        self.body_faces = body_faces

        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        for i, filename in enumerate(tqdm(self.filename_list, desc="Processing files")):
            folder_path = os.path.join(input_pose_path, filename)
            all_files = os.listdir(folder_path)
            # Sample one third of the available poses, with at least one per body.
            num_files_to_select = max(1, len(all_files) // 3)
            selected_files = random.sample(all_files, num_files_to_select)

            for f in tqdm(selected_files, desc=f"Processing {filename}", leave=False):
                print(f)
                input_file = os.path.join(input_pose_path+"/"+filename, f)
                verts_pose, _ = pp3d.read_mesh(input_file)
                verts_pose = torch.tensor(verts_pose).float()
                original_verts_pose = verts_pose.clone()

                verts_pose = diffusion_net.geometry.normalize_positions(verts_pose, method='bbox',scale_method='max_rad')

                file_name_pose = os.path.splitext(f)[0]

                self.verts_list_pose.append(verts_pose)
                # Pose variants share the base mesh topology and correspondence labels.
                self.faces_list_pose.append(self.faces_list[i])
                self.labels_list_pose.append(self.labels_list[i])
                self.gt_verts_list_pose.append(self.gt_verts_list[i])
                self.filename_list_pose.append(file_name_pose)
                self.input_edges_list_pose.append(self.input_edges_list[i])
                self.input_original_verts_list_pose.append(original_verts_pose)

        self.frames_list_pose, self.massvec_list_pose, self.L_list_pose, self.evals_list_pose, self.evecs_list_pose, self.gradX_list_pose, self.gradY_list_pose = diffusion_net.geometry.get_all_operators(self.verts_list_pose, self.faces_list_pose, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

    def __len__(self):
        return len(self.verts_list_pose)

    def __getitem__(self, idx):

        return self.verts_list_pose[idx], self.faces_list_pose[idx], self.frames_list_pose[idx], self.massvec_list_pose[idx], self.L_list_pose[idx], self.evals_list_pose[idx], self.evecs_list_pose[idx], self.gradX_list_pose[idx], self.gradY_list_pose[idx], self.labels_list_pose[idx], self.gt_verts_list_pose[idx], self.body_verts, self.filename_list_pose[idx], self.input_edges_list_pose[idx], self.body_uvs, self.body_faces, self.input_original_verts_list_pose[idx]

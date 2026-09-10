import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORRESPONDENCE_ROOT = Path(__file__).resolve().parent
DIFFUSION_NET_SRC = PROJECT_ROOT / "third_party" / "diffusion_net" / "src"
SHARED_REGISTRATION_ROOT = PROJECT_ROOT / "smpl_registration"
sys.path.insert(0, str(CORRESPONDENCE_ROOT))
sys.path.insert(0, str(DIFFUSION_NET_SRC))
sys.path.insert(0, str(SHARED_REGISTRATION_ROOT))

from DiffusionNet_loss import total_loss_function
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.spatial import KDTree
from utils import save_obj, VertexNormals
import open3d as o3d

import diffusion_net
from diffusionNet_dataloader import ClothMeshDataset, ClothMeshDataset_eval, BodyMeshDataset, BodyMeshDataset_eval, BodyMeshPoseDataset, ClothMeshDataset_single_eval, BodyMeshDataset_single_eval
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
from scipy.spatial import KDTree


def visualization(pred_uv, body_uv, draw_cloth_vertices, body_vertices, cloth_faces, body_faces, epoch, filename, train=True):
    pred_uv = pred_uv.detach().cpu().numpy()
    pred_uv *= 0.5  # Match the centered template UV range [-0.5, 0.5].
    uv_tree = KDTree(body_uv)
    _, top_3_indices = uv_tree.query(pred_uv, k=3)

    draw_cloth_vertices = draw_cloth_vertices.detach().cpu().numpy()
    body_vertices = body_vertices.detach().cpu().numpy()
    body_faces = body_faces.detach().cpu().numpy() if isinstance(body_faces, torch.Tensor) else body_faces
    cloth_faces = cloth_faces.detach().cpu().numpy() if isinstance(cloth_faces, torch.Tensor) else cloth_faces

    body_colors = plt.cm.hsv(np.linspace(0, 1, len(body_vertices)))[:, :3]

    body_face_colors = body_colors[body_faces].mean(axis=1)

    garment_colors = body_colors[top_3_indices].mean(axis=1)
    cloth_face_colors = garment_colors[cloth_faces].mean(axis=1)

    base_path = str(PROJECT_ROOT / "results")

    fig_body = plt.figure()
    ax_body = fig_body.add_subplot(111, projection='3d')

    ax_body.axis('off')

    ax_body.view_init(elev=90, azim=-90)

    body_mesh = Poly3DCollection(body_vertices[body_faces], facecolors=body_face_colors, linewidths=0.02, edgecolors='k', alpha=0.7)
    ax_body.add_collection3d(body_mesh)

    ax_body.auto_scale_xyz(body_vertices[:, 0], body_vertices[:, 1], body_vertices[:, 2])

    fig_cloth = plt.figure()
    ax_cloth = fig_cloth.add_subplot(111, projection='3d')

    ax_cloth.axis('off')

    ax_cloth.view_init(elev=90, azim=-90)

    cloth_mesh = Poly3DCollection(draw_cloth_vertices[cloth_faces], facecolors=cloth_face_colors, linewidths=0.02, edgecolors='k', alpha=0.7)
    ax_cloth.add_collection3d(cloth_mesh)

    ax_cloth.auto_scale_xyz(draw_cloth_vertices[:, 0], draw_cloth_vertices[:, 1], draw_cloth_vertices[:, 2])

    if train:
        image_name_cloth = f'{base_path}/runs/DiffusionNet_vis/{filename}_epoch_{epoch+1}_cloth.png'
    else:
        image_name_cloth = f'{base_path}/runs/test/{filename}_epoch_{epoch+1}_cloth_test.png'
    fig_cloth.savefig(image_name_cloth, dpi=2000, bbox_inches='tight', pad_inches=-0.1, transparent=True)
    plt.close(fig_cloth)


def export_corresponding_mesh(pred_uv, garment_faces, body_uv, body_vertices, filename, output_dir=None):
    pred_uv = pred_uv.detach().cpu().numpy()
    pred_uv *= 0.5  # Match the centered template UV range [-0.5, 0.5].

    uv_tree = KDTree(body_uv)
    _, top_3_indices = uv_tree.query(pred_uv, k=1)

    corresponding_positions = body_vertices[top_3_indices]
    garment_faces = garment_faces.detach().cpu().numpy()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corresponding_positions)
    mesh.triangles = o3d.utility.Vector3iVector(garment_faces)
    mesh.compute_vertex_normals()

    print("Applying Laplacian smoothing to the mesh...")
    number_of_iterations = 100
    mesh_smoothed = mesh.filter_smooth_taubin(number_of_iterations)
    mesh_smoothed.compute_vertex_normals()

    output_dir = Path(output_dir or PROJECT_ROOT / "results" / "correspondence" / "mesh")
    output_dir.mkdir(parents=True, exist_ok=True)
    smoothed_file_path = output_dir / f"smoothed_{filename}.obj"
    o3d.io.write_triangle_mesh(smoothed_file_path, mesh_smoothed, write_ascii=True)
    print(f"Saved smoothed mesh to '{smoothed_file_path}'")

    original_mesh_path = output_dir / f"original_{filename}.obj"
    o3d.io.write_triangle_mesh(original_mesh_path, mesh, write_ascii=True)
    print(f"Saved original mesh to '{original_mesh_path}'")


def export_correspondence(pred_uv, garment_verts, garment_faces, body_uv, body_vertices, body_faces, filename, mesh_type, output_dir=None):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    pred_uv = pred_uv.detach().cpu().numpy()
    garment_verts = garment_verts.detach().cpu()
    garment_faces = garment_faces.detach().cpu()
    pred_uv *= 0.5  # Match the centered template UV range [-0.5, 0.5].

    uv_tree = KDTree(body_uv)
    _, top_1_indices = uv_tree.query(pred_uv, k=1)

    if output_dir is None:
        output_dir = os.path.join(project_dir, 'results', 'correspondence', mesh_type)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, f'top_1_indices_{filename}.npz'), top_1_indices=top_1_indices)

parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks')", default = 'xyz')
parser.add_argument("--mesh_type", type=str, help="what mesh category to use", default = 'cloth')
parser.add_argument("--test_file", type=str, help="the path of the testing file", default = None)
parser.add_argument("--output_dir", type=str, default=None, help="directory for correspondence NPZ outputs")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint to use for evaluation")
parser.add_argument("--body-template", type=str, required=True, help="SMPL UV template OBJ")
parser.add_argument("--device", type=str, default="cuda:0", help="torch device, e.g. cuda:0 or cpu")
args = parser.parse_args()
os.environ["LUVITON_BODY_TEMPLATE"] = str(Path(args.body_template).expanduser().resolve())

device = torch.device(args.device)
dtype = torch.float32

n_class = 2

input_features = args.input_features
k_eig = 300

train = not args.evaluate
if train:
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Training requires the optional 'wandb' package. Install it with "
            "'pip install wandb', or pass --evaluate for inference."
        ) from exc

n_epoch = 4000
lr = 5e-4
decay_every = 100
decay_rate = 0.95

augment_random_rotate = False

base_path = str(PROJECT_ROOT)
op_cache_dir = os.path.join(base_path, "results", "op_cache")

if args.mesh_type == 'cloth':
    model_save_path = os.path.join(base_path, "models", "cloth_correspondence.pth")
    pretrain_path = args.checkpoint or model_save_path

    dataset_path = os.path.join(base_path, "data/cloth/train")
    test_dataset_path = os.path.join(base_path, "data", "cloth", "evaluation")

    if args.test_file is not None:
        test_dataset = ClothMeshDataset_single_eval(args.test_file, train=False, k_eig=k_eig, use_cache=False, op_cache_dir=op_cache_dir)
    else:
        test_dataset = ClothMeshDataset_eval(test_dataset_path, train=False, k_eig=k_eig, use_cache=False, op_cache_dir=op_cache_dir)

    test_loader = DataLoader(test_dataset, batch_size=None)

if args.mesh_type == 'body':
    model_save_path = os.path.join(base_path, "models", "body_correspondence.pth")
    pretrain_path = args.checkpoint or model_save_path

    dataset_path = os.path.join(base_path, "data/body/train_pose")
    test_dataset_path = os.path.join(base_path, "data/body/Luviton_correspondence_dataset")

    if args.test_file is not None:
        test_dataset = BodyMeshDataset_single_eval(args.test_file, train=False, k_eig=k_eig, use_cache=False, op_cache_dir=op_cache_dir)
    else:
        test_dataset = BodyMeshDataset_eval(test_dataset_path, train=False, k_eig=k_eig, use_cache=False, op_cache_dir=op_cache_dir)
    test_loader = DataLoader(test_dataset, batch_size=None)

if train:
    if args.mesh_type == 'cloth':
        train_dataset = ClothMeshDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
    if args.mesh_type == 'body':
        train_dataset = BodyMeshPoseDataset(dataset_path, train=True, k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
    # Process one mesh at a time because vertex counts and topology vary.
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    wandb.init(project="diffusion-net-body", config={
        "epochs": n_epoch,
        "batch_size": None,
        "learning_rate": lr,
        "decay_rate": decay_rate,
        "input_features": input_features,
        "architecture": "DiffusionNet",
    })

    config = wandb.config
    config.optimizer = "Adam"

C_in={'xyz':3, 'hks':16}[input_features]

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_class,
                                          C_width=128,
                                          N_block=7,
                                          last_activation=lambda x : torch.tanh(x),
                                          outputs_at='vertices',
                                          dropout=True,
                                          with_gradient_rotations = True)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train_epoch(epoch, mesh_type):

    if epoch > 0 and epoch % decay_every == 0:
        global lr
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    model.train()
    optimizer.zero_grad()

    epoch_losses = {'GT_uv_loss': 0}
    total_loss = 0
    for data in tqdm(train_loader):

        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, gt_verts, body_verts, filename, garment_edges, body_uvs, body_faces, original_verts = data

        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device)

        if augment_random_rotate:
            verts = diffusion_net.utils.random_rotate_points(verts)

        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
        body_uvs = body_uvs - 0.5  # Center template UVs around zero.

        loss, losses = total_loss_function(preds, body_verts, body_uvs, labels, garment_edges)

        for key in epoch_losses.keys():
            epoch_losses[key] += losses[key]

        total_loss += loss
        wandb.log({"batch_loss": loss.item()})
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    if (epoch + 1) % 200 == 0:
        model_save_path = os.path.join(base_path, "models", "{}_mesh_corres_{}_7x128_{}_epoch.pth".format(mesh_type, input_features, epoch+1))
        torch.save(model.state_dict(), model_save_path)
        test(epoch, model_save_path)
    wandb.log({"epoch_loss": total_loss.item()})
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
    return loss.item()


def test(epoch, pretrain_path):
    print("Loading pretrained model from: " + str(pretrain_path))

    if not os.path.isfile(pretrain_path):
        raise FileNotFoundError(
            f"Correspondence checkpoint not found: {pretrain_path}. "
            "See the Assets section in README.md or pass --checkpoint."
        )
    model.load_state_dict(torch.load(pretrain_path, map_location=device))
    model.eval()

    with torch.no_grad():

        for data in tqdm(test_loader):

            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, body_verts, filename, garment_edges, body_uvs, body_faces, original_verts = data

            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)

            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)
            body_uvs = body_uvs - 0.5  # Center template UVs around zero.

            export_correspondence(preds, original_verts, faces, body_uvs, body_verts, body_faces, filename, args.mesh_type, args.output_dir)

if train:
    print("Training...")
    wandb.config.update({
        "learning_rate": lr,
        "epochs": n_epoch,
        "decay_rate": decay_rate,
        "input_features": input_features,
        "architecture": "DiffusionNet",
    })

    for epoch in range(n_epoch):

        loss = train_epoch(epoch, args.mesh_type)
    model_save_path = os.path.join(base_path, "models", "{}_mesh_corres_{}_7x128_final.pth".format(args.mesh_type, input_features))
    print(" ==> saving last model to " + model_save_path)
    torch.save(model.state_dict(), model_save_path)
    wandb.finish()

if not train:
    test(9999, pretrain_path)

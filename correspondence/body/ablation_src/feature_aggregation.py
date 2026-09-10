import torch
import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from diff3f import get_features_per_vertex_modified_ablation
from utils import convert_mesh_container_to_torch_mesh, cosine_similarity, get_vertex_colors_from_obj
from dataloaders.mesh_container import MeshContainer
from dino import init_dino
import numpy as np
import argparse

# Define constants
latent_view_size = 60
H = latent_view_size * 8
W = latent_view_size * 8
tolerance = 0.004
random_seed = 42
use_normal_map = True
expected_views = 36
expected_feature_dim = 2688

def compute_features(device, dino_model, m, diffusion_features, point_clouds, rendered_rgb_views_folder, sapiens_label_folder, aggregation):
    mesh = convert_mesh_container_to_torch_mesh(m, device=device, is_tosca=False)
    mesh_vertices = mesh.verts_list()[0]
    features = get_features_per_vertex_modified_ablation(
        device=device,
        dino_model=dino_model,
        mesh=mesh,
        mesh_vertices=mesh_vertices,
        diffusion_features=diffusion_features,
        point_clouds=point_clouds,
        rendered_rgb_views_folder=rendered_rgb_views_folder,
        sapiens_label_folder=sapiens_label_folder,
        H=H,
        W=W,
        tolerance=tolerance,
        aggregation=aggregation
    )
    return features.cpu()

def main(args):
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    # Initialize the DINO model
    dino_model = init_dino(device)

    # Paths for target
    target_file_path = os.path.join(args.target_file_folder, "textured.obj")
    target_diffusion_features_path = os.path.join(args.target_file_folder, "f_maps.npz")
    target_point_clouds_path = os.path.join(args.target_file_folder, "verts.npy")
    target_rendered_rgb_views_path = os.path.join(args.target_file_folder, "rgb_views")
    target_sapiens_label_folder = os.path.join(args.target_file_folder, "sapiens_1b")

    # Load target data
    target_diffusion_features = np.load(target_diffusion_features_path)
    target_point_clouds = torch.from_numpy(np.load(target_point_clouds_path)).to(device)
    if target_point_clouds.shape[0] != expected_views:
        raise ValueError(
            f"Expected {expected_views} target views, got {target_point_clouds.shape[0]}"
        )
    for key in ("f_map0", "f_map1", "f_map2"):
        if key not in target_diffusion_features:
            raise KeyError(f"SyncMVD feature archive is missing {key}")
        if target_diffusion_features[key].shape[0] != expected_views:
            raise ValueError(
                f"{key} has {target_diffusion_features[key].shape[0]} views; expected {expected_views}"
            )
    target_mesh = MeshContainer().load_from_file(target_file_path)
    target_mesh.vert = target_mesh.vert[:, :3]

    if os.path.exists(args.source_feature_save_path):
        print(f"Loading precomputed features from: {args.source_feature_save_path}")
        f_source = torch.load(args.source_feature_save_path, map_location="cpu")
    else:
        if not args.source_file_folder:
            raise FileNotFoundError(
                "The precomputed source feature file is missing and --source_file_folder was not provided: "
                f"{args.source_feature_save_path}"
            )

        source_file_path = os.path.join(args.source_file_folder, "textured.obj")
        source_diffusion_features_path = os.path.join(args.source_file_folder, "f_maps.npz")
        source_point_clouds_path = os.path.join(args.source_file_folder, "verts.npy")
        source_rendered_rgb_views_path = os.path.join(args.source_file_folder, "rgb_views")
        source_sapiens_label_folder = os.path.join(args.source_file_folder, "sapiens_1b")

        source_diffusion_features = np.load(source_diffusion_features_path)
        source_point_clouds = torch.from_numpy(np.load(source_point_clouds_path)).to(device)
        source_mesh = MeshContainer().load_from_file(source_file_path)
        source_mesh.vert = source_mesh.vert[:, :3]

        print("Precomputed source features not found; computing them from SyncMVD results...")
        f_source = compute_features(device, dino_model, source_mesh, source_diffusion_features, source_point_clouds, source_rendered_rgb_views_path, source_sapiens_label_folder, args.aggregation)
        os.makedirs(os.path.dirname(os.path.abspath(args.source_feature_save_path)), exist_ok=True)
        torch.save(f_source, args.source_feature_save_path)
        print(f"Features saved to: {args.source_feature_save_path}")

    if f_source.ndim != 2 or f_source.shape[1] != expected_feature_dim:
        raise ValueError(
            f"Source feature cache must have shape [vertices, {expected_feature_dim}], "
            f"got {tuple(f_source.shape)}"
        )
    
    # f_source = compute_features(device, dino_model, source_mesh, source_diffusion_features, source_point_clouds, source_rendered_rgb_views_path, source_sapiens_label_folder, args.aggregation)

    f_target = compute_features(device, dino_model, target_mesh, target_diffusion_features, target_point_clouds, target_rendered_rgb_views_path, target_sapiens_label_folder, args.aggregation)
    
    # Compute correspondence    
    s = cosine_similarity(f_source.to(device), f_target.to(device))
    s = torch.argmax(s, dim=0).cpu().numpy()
    

    # Save correspondence
    os.makedirs(os.path.dirname(args.correspondence_path), exist_ok=True)
    np.save(args.correspondence_path, s)
    print(f"Correspondence saved to {args.correspondence_path}")

    
    if args.vis_save_path is not None:
        import meshplot as mp

        mp.offline()
        if args.vis_source_mesh is None or args.vis_source_texture is None:
            raise ValueError("--vis_source_mesh and --vis_source_texture are required when --vis_save_path is set.")
        # Visualize the correspondence
        cmap_source = get_vertex_colors_from_obj(args.vis_source_mesh, args.vis_source_texture)
        
        cmap_target = cmap_source[s]
        p = mp.plot(target_mesh.vert, target_mesh.face, c=cmap_target, shading={"point_size": 0.1})
        # # mp.plot(source_mesh.vert, source_mesh.face, c=cmap_source, shading={"point_size": 0.1})
        # mesh_name = os.path.basename(os.path.dirname(args.correspondence_path))
        filename = os.path.basename(args.correspondence_path)
        # scan_index = int(filename.split('_')[1].split('.')[0])
        os.makedirs(args.vis_save_path, exist_ok=True)  # Create the directory if it doesn't exist

        # # Now save the meshplot HTML
        p.save(os.path.join(args.vis_save_path, f"{filename}.html"))

if __name__ == "__main__":
    # for target_name in ['Megan_1', 'Michelle_2', 'Ortiz_2', 'X_Bot_2', 'Mutant_3','Luffy', 'Mutant_2', 'Timmy_1', 'Timmy_1', 'SportGranny_1', 'Y_Bot_2', 'Jennifer_2', 'Abe_1']:
    parser = argparse.ArgumentParser(description="Compute and visualize mesh correspondences.")
    parser.add_argument("--source_file_folder", type=str, default=None, help="Fallback source SyncMVD results folder used when the feature cache is missing.")
    parser.add_argument("--target_file_folder", type=str, required=True, help="Path to the target SyncMVD results folder.")
    parser.add_argument("--correspondence_path", type=str, required=True, help="Path to save the computed correspondence.")
    parser.add_argument("--source_feature_save_path", type=str, required=True, help="Path to cache/load source SMPL features.")
    parser.add_argument("--vis_save_path", type=str, default=None)
    parser.add_argument("--vis_source_mesh", type=str, default=None)
    parser.add_argument("--vis_source_texture", type=str, default=None)
    parser.add_argument("--aggregation", type=int, default=0)

    args = parser.parse_args()
    main(args)

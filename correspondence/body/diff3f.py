"""Aggregate saved SyncMVD and DINO features onto mesh vertices."""

import os
import random
import time

import numpy as np
import torch
from PIL import Image
from pytorch3d.ops import ball_query
from tqdm import tqdm

from dino import get_dino_features


VERTEX_GPU_LIMIT = 35000


def arange_pixels(
    resolution=(128, 128),
    batch_size=1,
    subsample_to=None,
    invert_y_axis=False,
    margin=0,
    corner_aligned=True,
    jitter=None,
):
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    uh = 1 if corner_aligned else 1 - (1 / h)
    uw = 1 if corner_aligned else 1 - (1 / w)
    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin
        w, h = w + margin * 2, h + margin * 2

    x, y = torch.linspace(-uw, uw, w), torch.linspace(-uh, uh, h)
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
    x, y = torch.meshgrid(x, y)
    pixel_scaled = (
        torch.stack([x, y], -1)
        .permute(1, 0, 2)
        .reshape(1, -1, 2)
        .repeat(batch_size, 1, 1)
    )

    if subsample_to is not None and subsample_to > 0 and subsample_to < n_points:
        idx = np.random.choice(
            pixel_scaled.shape[1], size=(subsample_to,), replace=False
        )
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled



def get_features_per_vertex_modified_ablation(
    device,
    dino_model,
    mesh,
    diffusion_features,
    point_clouds,
    rendered_rgb_views_folder,
    sapiens_label_folder,
    H=512,
    W=512,
    tolerance=0.01,
    aggregation=0,
    mesh_vertices=None,
    return_image=True,
    bq=True
):
    t1 = time.time()
    
    # Mesh vertex setup
    if mesh_vertices is None:
        mesh_vertices = mesh.verts_list()[0]
    if len(mesh_vertices) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(mesh_vertices)), 10000)
        maximal_distance = torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
    else:
        maximal_distance = torch.cdist(mesh_vertices, mesh_vertices).max()
    ball_drop_radius = maximal_distance * tolerance
    
    if aggregation == 0:
        FEATURE_DIMS = 1280+640+768 # best aggregation
    elif aggregation == 1:
        FEATURE_DIMS = 1280 # diffusion feature 0 only
    elif aggregation == 2:
        FEATURE_DIMS = 1280 # diffusion feature 1 only
    elif aggregation == 3:
        FEATURE_DIMS = 640 # diffusion feature 2 only
    elif aggregation == 4:
        FEATURE_DIMS = 768 # dino feature only
    elif aggregation == 5:
        FEATURE_DIMS = 1280 + 768 # diffusion feature 0 and dino feature
    elif aggregation == 6:
        FEATURE_DIMS = 1280 + 640 # only diffusion features
    
        
        
        
    
        
        

    # Initialize vertex features and counts
    ft_per_vertex = torch.zeros((len(mesh_vertices), FEATURE_DIMS)).half()
    ft_per_vertex_count = torch.zeros((len(mesh_vertices), 1)).half()
    
    # Load and move diffusion features to device
    diffusion_features0 = torch.from_numpy(diffusion_features['f_map0']).to(device).half()
    diffusion_features1 = torch.from_numpy(diffusion_features['f_map1']).to(device).half()
    diffusion_features2 = torch.from_numpy(diffusion_features['f_map2']).to(device).half()
    
    # Clean up original diffusion features
    del diffusion_features
    torch.cuda.empty_cache()
    
    sapiens_label_list = [[] for _ in range(len(mesh_vertices))]
    
    # get the parts that are before 'result'
    mesh_folder = rendered_rgb_views_folder.split('result')[0]        
    depth_map_path = os.path.join(mesh_folder, 'intermediate', 'cond.jpg')
    depth_map = Image.open(depth_map_path)
    

    # Processing each view
    for i in tqdm(range(point_clouds.shape[0])):
        
        # Load the RGB view image
        rendered_rgb_view_path = rendered_rgb_views_folder + f"/{i}.jpg"
        rendered_rgb_view = Image.open(rendered_rgb_view_path)
        # Extract valid world coordinates
        point_cloud = point_clouds[i]
        indices = point_cloud[..., 3] != 0
        point_cloud = point_cloud[..., :3]
        world_coords = point_cloud[indices]
        
        # Free memory for point cloud
        del point_cloud
        torch.cuda.empty_cache()
        
        # Create grid for alignment
        grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
        
        # depth_map image is [45*512, 512]
        # slice the depth_map image to get the depth_map image for the current view
        depth_map_view = depth_map.crop((i * 512, 0, (i + 1) * 512, 512))

        # aligned_dino_features = get_featup_features(device, dino_model, rendered_rgb_view, grid)
        aligned_dino_features = get_dino_features(device, dino_model, rendered_rgb_view, grid)
        # aligned_dino_features = get_dino_features(device, dino_model, depth_map_view, grid)
        
        aligned_features_list = []
        
        # Align and normalize each diffusion feature layer sequentially
        for diffusion_feature in [diffusion_features0[i], diffusion_features1[i], diffusion_features2[i]]:
            upsampled_feature = torch.nn.Upsample(size=(H, W), mode="bilinear")(diffusion_feature.unsqueeze(0)).to(device)
            ft_dim = upsampled_feature.size(1)
            aligned_feature = torch.nn.functional.grid_sample(upsampled_feature, grid, align_corners=False).reshape(1, ft_dim, -1)
            aligned_feature = torch.nn.functional.normalize(aligned_feature, dim=1)
            aligned_features_list.append(aligned_feature.cpu())  # Move to CPU to save GPU memory
            
            # Clear memory after processing each feature
            del upsampled_feature, aligned_feature
            torch.cuda.empty_cache()

        # Concatenate the aligned features and move back to GPU
        # aligned_features = torch.hstack([
        #     aligned_features_list[0].to(device) * 0.9,
        #     aligned_features_list[1].to(device) * 0.4,
        #     aligned_features_list[2].to(device) * 0.1,
        #     aligned_dino_features * 0.2
        # ])
        
        # aligned_features = torch.hstack([
        #     aligned_features_list[0].to(device) * 1.2, #+ 
        #     aligned_features_list[1].to(device) * 0.3,
        #     # aligned_features_list[2].to(device) * 0.1,
        #     aligned_dino_features * 0.8
        # ])
        if aggregation == 0:
            aligned_features = torch.hstack([
                aligned_features_list[0].to(device) * 1.2 + aligned_features_list[1].to(device) * 0.4,
                aligned_features_list[2].to(device) * 0.1,
                aligned_dino_features * 0.8
            ])
        elif aggregation == 1:
            aligned_features = torch.hstack([
                aligned_features_list[0].to(device)
            ])
        elif aggregation == 2:
            aligned_features = torch.hstack([
                aligned_features_list[1].to(device)
            ])
        elif aggregation == 3:
            aligned_features = torch.hstack([
                aligned_features_list[2].to(device)
            ])
        elif aggregation == 4:
            aligned_features = torch.hstack([
                aligned_dino_features
            ])
        elif aggregation == 5:
            aligned_features = torch.hstack([
                aligned_features_list[0].to(device) * 1.2,
                aligned_dino_features * 0.8
            ])
        elif aggregation == 6:
            aligned_features = torch.hstack([
                aligned_features_list[0].to(device) * 1.2 + aligned_features_list[1].to(device) * 0.4,
                aligned_features_list[2].to(device) * 0.1
            ])

        
        
        # use aligned dinofeature to enhance the aligned_features, dino feature channel is 768, diffusion feature channel is 1280+1280+640
        

        # # make the depth_map_view as tensor
        # depth_map_view = torch.tensor(np.array(depth_map_view)[..., 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 512, 512]
        # depth_map_view = depth_map_view.reshape(1, 1, -1)  # [1, 1, 262144]

        # # Normalize the depth map view
        # depth_map_view = torch.nn.functional.normalize(depth_map_view, dim=2)* 10

        # # Compute uncertainty and apply it to aligned_features
        # depth_map_view = depth_map_view.clamp(min=1e-8)
        # certainty = -1 * torch.log(depth_map_view) * depth_map_view
        # certainty = certainty.expand_as(aligned_features)  # Ensure compatible shapes

        # # Adjust aligned_features based on uncertainty
        # aligned_features = aligned_features + aligned_features * certainty*20
        
        # aligned_features = torch.hstack([
        #     aligned_features_list[0].to(device)*0.0240,
        #     aligned_features_list[1].to(device)*0.0230,
        #     aligned_features_list[2].to(device)*0.0181,
        #     aligned_dino_features.to(device)*0.9349
        # ])
        
        # Normalize the concatenated features
        aligned_features = torch.nn.functional.normalize(aligned_features, dim=1)
        
        
        
        # Filter features and labels for valid pixels
        indices = indices.flatten()
        features_per_pixel = aligned_features[0, :, indices].cpu()
        
        
        
        # Map pixel features to mesh vertices
        if bq:
            queried_indices = (
                ball_query(
                    world_coords.unsqueeze(0),
                    mesh_vertices.unsqueeze(0),
                    K=100,
                    radius=ball_drop_radius,
                    return_nn=False,
                ).idx[0].cpu()
            )
            
            mask = queried_indices != -1
            repeat = mask.sum(dim=1)
            ft_per_vertex_count[queried_indices[mask]] += 1
            ft_per_vertex[queried_indices[mask]] += features_per_pixel.repeat_interleave(repeat, dim=1).T
            
            # valid_indices = queried_indices[mask]
            # valid_embeddings = sapiens_label_per_pixel.repeat_interleave(repeat, dim=0)
            # for idx, embedding in zip(valid_indices, valid_embeddings):
            #     sapiens_label_list[idx].append(embedding)
        else:
            distances = torch.cdist(world_coords, mesh_vertices, p=2)
            closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
            ft_per_vertex[closest_vertex_indices] += features_per_pixel.T
            ft_per_vertex_count[closest_vertex_indices] += 1    

        # Clean up temporary variables
        del aligned_features, features_per_pixel, queried_indices, mask, repeat, # sapiens_label_per_pixel, sapiens_label
        torch.cuda.empty_cache()

    # Average features for each vertex
    idxs = (ft_per_vertex_count != 0)[:, 0]
    ft_per_vertex[idxs, :] = ft_per_vertex[idxs, :] / ft_per_vertex_count[idxs, :]
    
    # Fill missing vertex features
    filled_indices = ft_per_vertex_count[:, 0] != 0
    missing_indices = ft_per_vertex_count[:, 0] == 0
    if missing_indices.sum() > 0:
        distances = torch.cdist(mesh_vertices[missing_indices], mesh_vertices[filled_indices], p=2)
        closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
        ft_per_vertex[missing_indices, :] = ft_per_vertex[filled_indices][closest_vertex_indices, :]

    # # Process sapiens labels with voting
    # sapiens_label_per_vertex = np.zeros((len(mesh_vertices), 1))
    # for i in range(len(mesh_vertices)):
    #     labels = [label for label in sapiens_label_list[i] if label != 0]
    #     if labels:
    #         unique, counts = np.unique(labels, return_counts=True)
    #         sapiens_label_per_vertex[i, -1] = unique[np.argmax(counts)]
    
    # # Fill missing sapiens labels
    # missing_sapiens_label = len(sapiens_label_per_vertex[sapiens_label_per_vertex == 0])
    # if missing_sapiens_label > 0:
    #     filled_indices = sapiens_label_per_vertex[:, 0] != 0
    #     missing_indices = sapiens_label_per_vertex[:, 0] == 0
    #     distances = torch.cdist(mesh_vertices[missing_indices], mesh_vertices[filled_indices], p=2)
    #     closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
    #     sapiens_label_per_vertex[missing_indices, :] = sapiens_label_per_vertex[filled_indices][closest_vertex_indices, :]
    
    # # Generate sapiens vertex label embedding and concatenate to features
    # sapiens_vertex_label_embedding_tensor = sapiens_vertex_label_embedding(sapiens_label_per_vertex, num_channels=16)
    # ft_per_vertex = torch.hstack([ft_per_vertex, sapiens_vertex_label_embedding_tensor * 0])

    # Display timing information
    t2 = (time.time() - t1) / 60
    print("Time taken in mins: ", t2)
    
    return ft_per_vertex

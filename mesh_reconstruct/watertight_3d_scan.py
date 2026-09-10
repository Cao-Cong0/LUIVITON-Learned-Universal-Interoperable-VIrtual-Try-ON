import point_cloud_utils as pcu
import trimesh
import torch
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d
import argparse
from lib.common.render import Render

class DepthRender:
    def __init__(self, mesh_path) -> None:
        self.render = Render(size=1024, device=torch.device('cuda'))
        self.scale = 1.4

        # Step 1: Load the mesh using pcu to get vertices and faces
        v, f = pcu.load_mesh_vf(mesh_path)

        # # # Step 2: Make the mesh watertight
        # resolution = 80_000#18000
        # vw, fw = pcu.make_mesh_watertight(v, f, resolution)

        # Step 3: Create a trimesh object from the watertight vertices and faces
        # mesh = trimesh.Trimesh(vw, fw)
        mesh = trimesh.Trimesh(v, f)

        # Convert the mesh vertices to tensors
        # self.vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(self.render.device)

        self.vertices = torch.tensor(np.array(mesh.vertices), dtype=torch.float32).to(self.render.device)


        # Compute the offsets to center the mesh along each axis
        x_offset = -(torch.max(self.vertices[:, 0]) + torch.min(self.vertices[:, 0])) / 2
        y_offset = -(torch.max(self.vertices[:, 1]) + torch.min(self.vertices[:, 1])) / 2
        z_offset = -(torch.max(self.vertices[:, 2]) + torch.min(self.vertices[:, 2])) / 2

        # Apply the offsets to the vertices
        self.vertices[:, 0] += x_offset
        self.vertices[:, 1] += y_offset
        self.vertices[:, 2] += z_offset

        # Store the offsets separately if needed for other operations
        self.x_off_set = x_offset
        self.y_off_set = y_offset
        self.z_off_set = z_offset

        # Scale the vertices
        self.vertices /= self.scale

        # Get the faces
        self.faces = torch.tensor(np.array(mesh.faces), dtype=torch.long).to(self.render.device)


    def render_normal(self):
            normal_maps = self.render.get_rgb_image(cam_ids=[0, 1, 2, 3])

            normal_maps_final = []
            normal_maps_vis = []
            for i, normal_map in enumerate(normal_maps):
                normal_map = normal_map.squeeze().permute(1, 2, 0)
                if i == 2:
                    normal_map = torch.flip(normal_map, [1])
                normal_map = torch.flip(normal_map, [0]).cpu().numpy()
                normal_maps_final.append(normal_map)
                normal_map = (normal_map + 1) / 2 * 255
                normal_maps_vis.append(normal_map.astype(np.uint8))

            # for i, normal_map_v in enumerate(normal_maps_vis):
            #     cv2.imwrite(f'normal_map_{i}.png', normal_map_v)

            return normal_maps_final[0], normal_maps_final[1], normal_maps_final[2], normal_maps_final[3]      
        
    # from the Tester class, we should render the depth map when we load our mesh and camera
    def render_depth(self, cam_ids=[0, 1, 2, 3]):
        self.render.load_meshes(self.vertices*torch.tensor([1.0, -1.0, -1.0]).to(self.render.device), self.faces)
        depth_maps = self.render.get_depth_map(cam_ids=cam_ids)
                        
        # flip the depth map
        depth_map0 = torch.flip(depth_maps[0], [0]) - 100
        depth_map1 = torch.flip(depth_maps[1], [0]) - 100
        
        # do the same thing for depth 2 and depth 3
        depth_map2 = torch.flip(depth_maps[2], [0]) - 100
        depth_map3 = torch.flip(depth_maps[3], [0]) - 100
        
        # so we will have the depth min as -101
        return depth_map0, depth_map1, depth_map2, depth_map3

    def depth2pcl(self, depth, normal_map=None, scaled=False, direction='front'):
        ## This is the depth to pointcloud projection function for depth rendered from pytorch3d using FoVorthogonal projection camera
        
        if scaled==False:
            mask = depth > 0
            depth = (depth/255) * 2 - 1
        else:
            mask = depth > -1
        
        normal_map = normal_map.reshape(-1, 3)

        h, w = depth.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = x / w * 2 - 1
        y = y / h * 2 - 1
        if direction == 'front':
            xyz = np.stack([x, -y, -depth], axis=-1)
            normal_map[:, 1] = -normal_map[:, 1]
            normal_map[:, 2] = -normal_map[:, 2]
            xyz = xyz.reshape(-1, 3)
        elif direction == 'back':
            xyz = np.stack([x, -y, depth], axis=-1) 
            xyz = xyz.reshape(-1, 3)
            normal_map[:, 1] = -normal_map[:, 1]
            normal_map[:, 2] = -normal_map[:, 2]
        elif direction == 'left':
            rotation_matrix = np.array([[0, 0, 1],
                                        [0, 1, 0],
                                        [-1, 0, 0]
                                        ])
            # multiply the rotation matrix to the pointcloud
            xyz = np.dot(np.stack([-x, -y, -depth], axis=-1).reshape(-1, 3), rotation_matrix.T)
            normal_map[:, 1] = -normal_map[:, 1]
            normal_map[:, 2] = -normal_map[:, 2]

        elif direction == 'right':
            # rotate the pointcloud according to the y axis by -90 degrees
            rotation_matrix = np.array([[0, 0, -1],
                                        [0, 1, 0],
                                        [1, 0, 0]
                                        ])
            xyz = np.dot(np.stack([-x, -y, -depth], axis=-1).reshape(-1, 3), rotation_matrix.T)
            normal_map[:, 1] = -normal_map[:, 1]
            normal_map[:, 2] = -normal_map[:, 2]

        mask = mask.flatten()
        xyz = xyz[mask]

        xyz = xyz * self.scale
        xyz[:, 0] -= self.x_off_set.cpu().numpy()
        xyz[:, 1] -= self.y_off_set.cpu().numpy()
        xyz[:, 2] -= self.z_off_set.cpu().numpy()

        if normal_map is not None:
            normal_map = normal_map[mask]
            # print("the norm of the normal_vector is: ", np.linalg.norm(normal_map, axis=1), 
            #       "min and max:", np.min(np.linalg.norm(normal_map, axis=1)), 
            #       np.max(np.linalg.norm(normal_map, axis=1)))
            normal_map = normal_map / np.linalg.norm(normal_map, axis=1)[:, None]

        xyz = np.concatenate((xyz, normal_map), axis=1)
        return xyz
    
    def save_ply_with_normal(self, xyz_with_normal, out_path):
        vertex_data_tuples = [tuple(vertex) for vertex in xyz_with_normal]

        # Define the data types for the .ply file
        vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]

        # Create a PlyElement for the vertex data
        vertex_element = PlyElement.describe(np.array(vertex_data_tuples, dtype=vertex_dtype), 'vertex')

        # Create a PlyData object
        ply_data = PlyData([vertex_element])

        # Save the PlyData object to a .ply file
        ply_data.write(out_path)

        print(f"Point cloud with normals saved to {out_path}")


def scan_reconstruct(input_obj_path, output_obj_path):
    render = DepthRender(mesh_path=input_obj_path)
    depth_map0, depth_map1, depth_map2, depth_map3 = render.render_depth()
    normal_map0, normal_map1, normal_map2, normal_map3 = render.render_normal()

    # get the pointclouds
    xyz0 = render.depth2pcl(depth_map0.cpu().numpy(), normal_map0, scaled=True, direction='back')
    xyz1 = render.depth2pcl(depth_map1.cpu().numpy(), normal_map1, scaled=True, direction='left')
    xyz2 = render.depth2pcl(depth_map2.cpu().numpy(), normal_map2, scaled=True, direction='front')
    xyz3 = render.depth2pcl(depth_map3.cpu().numpy(), normal_map3, scaled=True, direction='right')

    # combine all the pointclouds
    xyz = np.concatenate((xyz0, xyz1, xyz2, xyz3), axis=0)
    pcd_path = output_obj_path.replace('.obj', '.ply')
    render.save_ply_with_normal(xyz, pcd_path)

    pcd = o3d.io.read_point_cloud(pcd_path)

    # downsample point cloud
    # pcd = pcd.voxel_down_sample(voxel_size=0.016)
    pcd = pcd.voxel_down_sample(voxel_size=0.010)
    

    # print('run Poisson surface reconstruction')
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
    simplified_mesh = mesh.simplify_quadric_decimation(18000)
    print(simplified_mesh)

    # save as obj
    o3d.io.write_triangle_mesh(output_obj_path, simplified_mesh)


if __name__ == '__main__':
    # parse input and output obj paths
    parser = argparse.ArgumentParser(description='3D scan and reconstruct')
    parser.add_argument('--input_obj_path', type=str, required=True, help='input obj path')
    parser.add_argument('--output_obj_path', type=str, required=True, help='output obj path')
    args = parser.parse_args()


    scan_reconstruct(args.input_obj_path, args.output_obj_path)

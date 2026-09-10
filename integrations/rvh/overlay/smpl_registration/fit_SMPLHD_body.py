"""
fit SMPLH+offset to scans

created by Xianghui, 12 January 2022
"""
import os
import sys
from pathlib import Path


REGISTRATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REGISTRATION_ROOT))
from os.path import split, join, exists
import torch
from tqdm import tqdm
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance
import yaml
import numpy as np
from lib.mesh_laplacian import mesh_laplacian_smoothing
from smpl_registration.fit_SMPLH_body import SMPLHFitter
from utils import find_nearest_points, VertexNormals, load_obj, dilate_cloth
import smplx
from utils import save_obj
from smpl_registration.loss_weights import body_stage_two_weights





class SMPLDFitter(SMPLHFitter):
    def __init__(self, model_root, device='cuda:0', save_name='smpld', debug=False, hands=False):
        super(SMPLDFitter, self).__init__(model_root, device, save_name, debug, hands)
        self.save_name_base = 'smplhd' if self.hands else 'smpld'

    def fit(self, scans, correspondence_path, smpl_pkl, gender='female', save_path=None, rejected_path=None, transformation_path=None):
        # correspondence = np.load(correspondence_path)['top_1_indices']
        correspondence = np.load(correspondence_path)
        if rejected_path is not None:
            reject = np.load(rejected_path)
        else:
            reject = None
        if smpl_pkl is None or smpl_pkl[0] is None:
            print('SMPL not specified, fitting SMPL now')
            pose, betas, trans, scale, offsets = super(SMPLDFitter, self).fit(scans, correspondence_path, gender, save_path, rejected_path)
        else:
            # load from fitting results
            pose, betas, trans = self.load_smpl_params(smpl_pkl)

        betas, pose, trans, scale, offsets = torch.tensor(betas), torch.tensor(pose), torch.tensor(trans), torch.tensor(scale), torch.tensor(offsets)
        # Batch size
        batch_sz = len(scans)

        # init smpl
        smpl = self.init_smpl(batch_sz, gender, pose, betas, trans, offsets, scale)

        # Load scans and center them. Once smpl is registered, move it accordingly.
        th_scan_meshes = self.load_scans(scans, device=self.device)

        dilated_scan_verts = dilate_cloth(
            th_scan_meshes.verts_list()[0], th_scan_meshes.faces_list()[0], distance=0.002
        )
        dilated_scan_mesh = Meshes(
            verts=dilated_scan_verts.unsqueeze(0), faces=th_scan_meshes.faces_list()[0].unsqueeze(0)
        )

        # optimize offsets
        self.optimize_offsets(th_scan_meshes, smpl, correspondence, dilated_scan_mesh, 30, 30, reject)

        if save_path is not None:
            if not exists(save_path):
                os.makedirs(save_path)

            return self.save_outputs(save_path, scans, smpl, th_scan_meshes, save_name=self.save_name_base, transformation_path=transformation_path)
        
    
    def penetration_loss(self, outside_mesh_verts, inside_mesh_verts, inside_mesh_faces,  d_e):
        # cloth_vertices: Tensor of cloth vertices [N, 3]
        # obstacle_normals: Tensor of the normals at the nearest points on the obstacle mesh [N, 3]
        # d_e: Minimum distance of penetration as a scalar
        nearest_body_points = find_nearest_points(outside_mesh_verts, inside_mesh_verts)

        obstacle_normals = VertexNormals(inside_mesh_verts, inside_mesh_faces)  # [N, 3]
        # Calculate vector from each cloth vertex to its nearest point on the obstacle mesh
        nearest_obstacle_points_positions = inside_mesh_verts[nearest_body_points[0]]  # [N, 3]
        v_to_o = outside_mesh_verts - nearest_obstacle_points_positions  # [N, 3]

        # Calculate dot product between v_to_o and obstacle normals
        dot_product = torch.einsum('ij,ij->i', v_to_o, obstacle_normals[nearest_body_points[0]])

        # Determine if inside (dot product is negative)
        inside_mesh = dot_product < -d_e

        # Calculate penetration distance (length of v_to_o for points inside)
        penetration_distances = torch.norm(v_to_o, dim=1) * inside_mesh

        # Resolve penetration by moving cloth vertices outside along the normal
        # Adjust positions of vertices that are inside
        penetration_penalties = penetration_distances[inside_mesh]

        # Compute the mean loss over all cloth vertices
        loss = torch.mean(penetration_penalties)

        return loss
    
    # def penetration_loss(self, outside_mesh_verts, outside_mesh_faces, inside_mesh_verts,  d_e):
    #     # cloth_vertices: Tensor of cloth vertices [N, 3]
    #     # obstacle_normals: Tensor of the normals at the nearest points on the obstacle mesh [N, 3]
    #     # d_e: Minimum distance of penetration as a scalar
    #     nearest_inside_mesh_points = find_nearest_points(outside_mesh_verts, inside_mesh_verts)

    #     outside_mesh_normals = VertexNormals(outside_mesh_verts, outside_mesh_faces)  # [N, 3]
    #     # Calculate vector from each cloth vertex to its nearest point on the obstacle mesh
    #     nearest_obstacle_points_positions = inside_mesh_verts[nearest_inside_mesh_points[0]]  # [N, 3]
    #     v_to_o = outside_mesh_verts - nearest_obstacle_points_positions  # [N, 3]

    #     # Calculate dot product between v_to_o and obstacle normals
    #     dot_product = torch.einsum('ij,ij->i', v_to_o, -outside_mesh_normals[nearest_inside_mesh_points[0]])

    #     # Determine if inside (dot product is negative)
    #     inside_mesh = dot_product < -d_e

    #     # Calculate penetration distance (length of v_to_o for points inside)
    #     penetration_distances = torch.norm(v_to_o, dim=1) * inside_mesh

    #     # Resolve penetration by moving cloth vertices outside along the normal
    #     # Adjust positions of vertices that are inside
    #     penetration_penalties = penetration_distances[inside_mesh]

    #     # Compute the mean loss over all cloth vertices
    #     loss = torch.mean(penetration_penalties)

    #     return loss
        
    def inside_smpl(self, smpl, th_scan_meshes):
        # this loss aims to ensure the scan is inside the smpl
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))
        loss = self.penetration_loss(verts[0], th_scan_meshes.verts_list()[0], th_scan_meshes.faces_list()[0], 0.01)
        return loss
        

    def forward_step_offset(self, th_scan_meshes, smpl, corrspondence, dilated_scan_mesh, init_smpl_lap, reject=None):
        """
            Performs a forward step, given smpl and scan meshes.
            Then computes the losses.
        """
        # forward
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))

        # losses
        loss = dict()
        loss['b2s'] = point_mesh_face_distance(
            th_smpl_meshes, Pointclouds(points=dilated_scan_mesh.verts_list())
        )
        loss['s2b'] = point_mesh_face_distance(
            dilated_scan_mesh, Pointclouds(points=th_smpl_meshes.verts_list())
        )
        if reject is not None:
            # get the inverse of reject (ture to false, false to true)  
            reject = np.logical_not(reject)
            reject = torch.from_numpy(reject).to(self.device).unsqueeze(1)
            result = (verts[0][corrspondence] - dilated_scan_mesh.verts_list()[0]) * reject
            loss['correspondence'] = torch.norm(result, dim=1).sum() / reject.sum()
        else:
            loss['correspondence'] = torch.norm(
                verts[0][corrspondence] - dilated_scan_mesh.verts_list()[0], dim=1
            ).mean()
        # loss['inside_smpl'] = self.inside_smpl(smpl, th_scan_meshes)
        lap_new = mesh_laplacian_smoothing(th_smpl_meshes, reduction=None) # (V, 3)
        # init_lap = mesh_laplacian_smoothing(th_smpl_meshes, reduction=None) # (V, 3)
        # reference: https://github.com/NVIDIAGameWorks/kaolin/blob/v0.1/kaolin/metrics/mesh.py#L155
        loss['lap'] = torch.mean(torch.sum((lap_new - init_smpl_lap)**2, 1))
        # loss['lap'] = mesh_laplacian_smoothing(th_smpl_meshes, method='uniform')
        # loss['edge'] = mesh_edge_loss(th_smpl_meshes)
        loss['offsets'] = torch.mean(torch.mean(smpl.offsets ** 2, axis=1))
        loss['scale'] = torch.norm(smpl.scale - torch.ones_like(smpl.scale))
        return loss

    def optimize_offsets(self, th_scan_meshes, smpl, corrspondence, dilated_scan_mesh, iterations, steps_per_iter, reject=None):
        # Optimizer
        optimizer = torch.optim.Adam([smpl.offsets, smpl.pose, smpl.betas, smpl.scale], 0.005, betas=(0.9, 0.999))

        # Get loss_weights
        weight_dict = body_stage_two_weights()

        # precompute initial laplacian of the smpl meshes
        bz = smpl.offsets.shape[0]
        verts, _, _, _ = smpl()
        verts_list = [verts[i].clone().detach() for i in range(bz)]
        faces_list = [torch.as_tensor(smpl.faces, dtype=torch.int64, device=verts.device) for _ in range(bz)]
        init_smpl = Meshes(verts_list, faces_list)
        init_lap = mesh_laplacian_smoothing(init_smpl, reduction=None)

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL+D')
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_step_offset(
                    th_scan_meshes, smpl, corrspondence, dilated_scan_mesh, init_lap, reject
                )
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Lx100. Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, loss_dict[k].mean().item() * 100)
                loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(smpl, th_scan_meshes)

    def get_loss_weights(self):
        return body_stage_two_weights()
    
def load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path, 'r') as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)

    config = {
        key: _resolve_config_path(config_path.parent, value)
        if isinstance(value, str) and key.endswith("PATH")
        else value
        for key, value in config.items()
    }

    return config


def _resolve_config_path(config_dir, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()

def main(args):
    
    
    
    fitter = SMPLDFitter(args.model_root, device=args.device, debug=args.display, hands=args.hands)
    fitter.fit([args.scan_path], args.correspondence_path, [args.smpl_pkl], args.gender, args.save_path, args.rejected_path, args.transformation_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('scan_path', type=str, help='path to the 3d scans')
    parser.add_argument('correspondence_path', type=str, help='load correspondence between the body and cloth')
    # parser.add_argument('rejected_path', type=str, default=None, help='load rejected points')
    parser.add_argument('save_path', type=str, help='save path for all scans')
    parser.add_argument('-rejected_path', type=str, default=None, help='load rejected points')
    parser.add_argument('-transformation_path', type=str, default=None, help='scale and translate the mesh back')
    parser.add_argument('-gender', type=str, default='female') # can be female
    parser.add_argument('-smpl_pkl', type=str, default=None)  # In case SMPL fit is already available
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument('-hands', default=False, action='store_true', help='use SMPL+hand model or not')
    parser.add_argument("--config-path", "-c", type=Path, default="smpl_registration/config.yml",
                        help="Path to yml file with config")
    parser.add_argument("--smpl-model-root", type=Path, default=None,
                        help="Override SMPL_MODELS_PATH from the config")
    parser.add_argument("--device", default="cuda:0", help="Torch device used for registration")
    args = parser.parse_args()

    # args.scan_path = 'data/mesh_1/scan.obj'
    # args.display = True
    # args.save_path = 'data/mesh_1'
    # args.gender = 'male'
    # args.smpl_pkl = "data/mesh_1/scan_smpl.pkl"
    config = load_config(args.config_path)
    args.model_root = (
        args.smpl_model_root.expanduser().resolve()
        if args.smpl_model_root is not None
        else Path(config["SMPL_MODELS_PATH"])
    )

    main(args)

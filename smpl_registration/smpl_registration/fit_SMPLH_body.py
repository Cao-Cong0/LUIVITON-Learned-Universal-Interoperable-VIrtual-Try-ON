import os
import sys
from pathlib import Path


REGISTRATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REGISTRATION_ROOT))
import torch
import numpy as np
import yaml
from tqdm import tqdm
from os.path import exists
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from smpl_registration.base_fitter import BaseFitter
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.smpl.priors.th_hand_prior import HandPrior
from utils import VertexNormals, find_nearest_points, dilate_cloth
from lib.mesh_laplacian import mesh_laplacian_smoothing
from smpl_registration.loss_weights import body_stage_one_weights



class SMPLHFitter(BaseFitter):
    def fit(self, scans, correspondence_path, gender='male', save_path=None, rejected_path=None):
        # correspondence = np.load(correspondence_path)['top_1_indices']
        correspondence = np.load(correspondence_path)
        if rejected_path is not None:
            reject = np.load(rejected_path)
        else:
            reject = None

        # Batch size
        batch_sz = len(scans)


        # Load scans and center them. Once smpl is registered, move it accordingly.
        th_scan_meshes, centers = self.load_scans(scans, device=self.device, ret_cent=True)

        # init smpl
        smpl = self.init_smpl(batch_sz, gender, trans=centers) # add centers as initial SMPL translation

        iterations, steps_per_iter = 20, 50

        dilated_scan_verts = dilate_cloth(
            th_scan_meshes.verts_list()[0], th_scan_meshes.faces_list()[0], distance=0.008
        )
        # Optimize pose and shape
        self.optimize_pose_shape(th_scan_meshes, smpl, iterations, steps_per_iter, correspondence, dilated_scan_verts, reject)

        if save_path is not None:
            if not exists(save_path):
                os.makedirs(save_path)
            return self.save_outputs(save_path, scans, smpl, th_scan_meshes, save_name='smplh' if self.hands else 'smpl')

    def optimize_pose_shape(self, th_scan_meshes, smpl, iterations, steps_per_iter, correspondence, dilated_scan_verts, reject=None):
        # Optimizer
        optimizer = torch.optim.Adam([smpl.trans, smpl.betas, smpl.pose, smpl.scale, smpl.offsets], 0.02, betas=(0.9, 0.999))
        # optimizer = torch.optim.Adam([smpl.trans, smpl.betas, smpl.pose, smpl.offsets], 0.02, betas=(0.9, 0.999))
        # SMPLDFitter inherits this optimizer; never dispatch to its stage-two schedule.
        weight_dict = body_stage_one_weights()

        bz = smpl.offsets.shape[0]
        verts, _, _, _ = smpl()
        verts_list = [verts[i].clone().detach() for i in range(bz)]
        # faces_list = [torch.tensor(smpl.faces) for i in range(bz)]
        faces_tensor = torch.as_tensor(smpl.faces, dtype=torch.int64, device=verts.device)
        faces_list = [faces_tensor.clone() for _ in range(bz)]
        init_smpl = Meshes(verts_list, faces_list)
        init_lap = mesh_laplacian_smoothing(init_smpl, reduction=None)

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL')
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_pose_shape(
                    th_scan_meshes, smpl, correspondence, dilated_scan_verts, init_lap, reject
                )
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(smpl, th_scan_meshes)

        print('** Optimised smpl pose and shape **')

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

    def inside_smpl(self, smpl, th_scan_meshes):
        # this loss aims to ensure the scan is inside the smpl
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))
        loss = self.penetration_loss(verts[0], th_scan_meshes.verts_list()[0], th_scan_meshes.faces_list()[0], 0.01)
        return loss
    
    
    def forward_pose_shape(self, th_scan_meshes, smpl, correspondence, dilated_scan_verts, init_smpl_lap, reject=None):
        prior = get_prior(self.model_root, smpl.gender, device=self.device)
        # forward
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))

        corresponding_body_verts = verts[0][correspondence]
        cloth_verts = th_scan_meshes.verts_list()[0]


        # losses
        loss = dict()
        loss['b2s'] = point_mesh_face_distance(
            th_smpl_meshes, Pointclouds(points=th_scan_meshes.verts_list())
        )
        loss['s2b'] = point_mesh_face_distance(
            th_scan_meshes, Pointclouds(points=th_smpl_meshes.verts_list())
        )
        
        if reject is not None:
            # get the inverse of reject (ture to false, false to true)  
            reject = np.logical_not(reject)
            reject = torch.from_numpy(reject).to(self.device).unsqueeze(1)
            result = (corresponding_body_verts - dilated_scan_verts) * reject

            loss['correspondence'] = torch.norm(result, dim=1).sum() / reject.sum()
        else:
            loss['correspondence'] = torch.norm(
                corresponding_body_verts - dilated_scan_verts, dim=1
            ).mean()
        loss['offsets'] = torch.mean(torch.mean(smpl.offsets ** 2, axis=1))
        loss['betas'] = torch.mean(smpl.betas ** 2)
        loss['scale'] = torch.norm(smpl.scale - torch.ones_like(smpl.scale))
        loss['pose_pr'] = torch.mean(prior(smpl.pose))

        lap_new = mesh_laplacian_smoothing(th_smpl_meshes, reduction=None)
        loss['lap'] = torch.mean(torch.sum((lap_new - init_smpl_lap)**2, 1))


        if self.hands:
            hand_prior = HandPrior(self.model_root, type='grab')
            loss['hand'] = torch.mean(hand_prior(smpl.pose))
        return loss

    def get_loss_weights(self):
        return body_stage_one_weights()

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
    fitter = SMPLHFitter(args.model_root, debug=args.display, hands=args.hands)
    fitter.fit([args.scan_path], args.correspondence_path, args.gender, args.save_path, args.rejected_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('scan_path', type=str, help='path to the 3d scans')
    parser.add_argument('correspondence_path', type=str, help='load correspondence between the body and cloth')
    parser.add_argument('save_path', type=str, help='save path for all scans')
    parser.add_argument('-rejected_path', type=str, default=None, help='load rejected points')
    parser.add_argument('-gender', type=str, default='female')  # can be female
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument("--config-path", "-c", type=Path, default="smpl_registration/config.yml",
                        help="Path to yml file with config")
    parser.add_argument('-hands', default=False, action='store_true', help='use SMPL+hand model or not')
    args = parser.parse_args()

    # args.scan_path = 'data/mesh_1/scan.obj'
    # args.display = True
    # args.save_path = 'data/mesh_1'
    # args.gender = 'male'
    config = load_config(args.config_path)
    args.model_root = Path(config["SMPL_MODELS_PATH"])

    main(args)

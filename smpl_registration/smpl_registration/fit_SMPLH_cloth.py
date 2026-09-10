import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REGISTRATION_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import yaml
from tqdm import tqdm
from os.path import exists
from pytorch3d.structures import Meshes
from smpl_registration.base_fitter import BaseFitter
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.smpl.priors.th_hand_prior import HandPrior
from smpl_registration.loss_weights import clothing_weights
from utils import VertexNormals, find_nearest_points, load_obj, shrink_cloth



class SMPLHFitter(BaseFitter):
    def fit(self, scans, correspondence_path, gender='female', save_path=None, template_body_path=None):
        correspondence_npz = np.load(correspondence_path)
        if 'top_1_indices' in correspondence_npz:
            correspondence = correspondence_npz['top_1_indices']
        else:
            correspondence = correspondence_npz

        # Batch size
        batch_sz = len(scans)


        # Load scans and center them. Once smpl is registered, move it accordingly.
        th_scan_meshes, centers = self.load_scans(scans, device=self.device, ret_cent=True)

        # init smpl
        smpl = self.init_smpl(batch_sz, gender, trans=centers) # add centers as initial SMPL translation

        iterations, steps_per_iter = 20, 30

        if template_body_path is None:
            template_body_path = PROJECT_ROOT / 'models' / 'smpl_uv_free.obj'
        template_body_verts, template_body_faces = load_obj(str(template_body_path))
        shrink_cloth_verts = shrink_cloth(
            th_scan_meshes.verts_list()[0],
            template_body_verts,
            template_body_faces,
            correspondence,
        )

        # Optimize pose and shape
        self.optimize_pose_shape(
            th_scan_meshes,
            smpl,
            iterations,
            steps_per_iter,
            correspondence,
            shrink_cloth_verts,
        )

        if save_path is not None:
            if not exists(save_path):
                os.makedirs(save_path)
            return self.save_outputs(save_path, scans, smpl, th_scan_meshes, save_name='smplh' if self.hands else 'smpl')

    def optimize_pose_shape(self, th_scan_meshes, smpl, iterations, steps_per_iter,
                            correspondence, shrink_cloth_verts):
        # Optimizer
        optimizer = torch.optim.Adam([smpl.trans, smpl.betas, smpl.pose], 0.02, betas=(0.9, 0.999))
        # Get loss_weights
        weight_dict = self.get_loss_weights()

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL')
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_pose_shape(
                    th_scan_meshes, smpl, correspondence, shrink_cloth_verts
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

    def penetration_loss(self, cloth_verts, body_vertices, body_faces,  d_e):
        nearest_body_points = find_nearest_points(cloth_verts, body_vertices)[0]
        obstacle_normals = VertexNormals(body_vertices, body_faces)
        v_to_o = cloth_verts - body_vertices[nearest_body_points]
        signed_distances = torch.einsum(
            'ij,ij->i', v_to_o, obstacle_normals[nearest_body_points]
        )
        penetrating = signed_distances < -d_e
        if not torch.any(penetrating):
            return v_to_o.sum() * 0.0
        return torch.norm(v_to_o[penetrating], dim=1).mean()
    
    
    def forward_pose_shape(self, th_scan_meshes, smpl, correspondence,
                           shrink_cloth_verts):
        prior = get_prior(self.model_root, smpl.gender, device=self.device)

        # forward
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))

        corresponding_body_verts = verts[0][correspondence]
        cloth_verts = th_scan_meshes.verts_list()[0]


        # losses
        loss = dict()
        loss['penetration'] = self.penetration_loss(cloth_verts, verts[0], smpl.faces, 0.01)
        loss['b2c'] = torch.norm(corresponding_body_verts - shrink_cloth_verts, dim=1).mean()
        loss['betas'] = torch.mean(smpl.betas ** 2)
        loss['pose_pr'] = torch.mean(prior(smpl.pose))
        loss['pose'] = torch.mean(smpl.pose ** 2)


        if self.hands:
            hand_prior = HandPrior(self.model_root, type='grab')
            loss['hand'] = torch.mean(hand_prior(smpl.pose)) # add hand prior if smplh is used
        return loss

    def get_loss_weights(self):
        return clothing_weights()

def load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path, 'r', encoding='utf-8') as fp:
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


def _validate_model_root(model_root, gender):
    required = [
        model_root / "priors" / "body_prior.pkl",
        model_root / f"SMPL_{gender}.pkl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(
            "SMPL assets are incomplete. Set SMPL_MODELS_PATH in the config to a directory "
            f"containing these files:\n  {formatted}"
        )


def main(args):
    fitter = SMPLHFitter(
        args.model_root,
        device=args.device,
        debug=args.display,
        hands=args.hands,
    )
    fitter.fit(
        [args.scan_path],
        args.correspondence_path,
        args.gender,
        args.save_path,
        args.template_body_path,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('scan_path', type=str, help='path to the 3d scans')
    parser.add_argument('correspondence_path', type=str, help='load correspondence between the body and cloth')
    parser.add_argument('save_path', type=str, help='save path for all scans')
    parser.add_argument('-gender', type=str, default='male')  # can be female
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument("--config-path", "-c", type=Path, default=PROJECT_ROOT / "configs" / "smpl.yml",
                        help="Path to yml file with config")
    parser.add_argument("--smpl-model-root", type=Path, default=None,
                        help="Override SMPL_MODELS_PATH from the config")
    parser.add_argument("--template-body-path", type=Path, default=PROJECT_ROOT / "models" / "smpl_uv_free.obj",
                        help="Template SMPL OBJ used to shrink clothing during registration")
    parser.add_argument("--device", type=str, default="cuda:0", help="torch device, e.g. cuda:0 or cpu")
    parser.add_argument('-hands', default=False, action='store_true', help='use SMPL+hand model or not')
    args = parser.parse_args()

    # args.scan_path = 'data/mesh_1/scan.obj'
    # args.display = True
    # args.save_path = 'data/mesh_1'
    # args.gender = 'male'
    config = load_config(args.config_path)
    args.model_root = (
        args.smpl_model_root.expanduser().resolve()
        if args.smpl_model_root is not None
        else Path(config["SMPL_MODELS_PATH"])
    )
    _validate_model_root(args.model_root, args.gender)

    main(args)

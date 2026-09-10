"""
base smpl fitter class to handle data io, load smpl, output saving etc. so that they can be easily reused later
this can be inherited for fitting smplh, smph+d to scan, kinect point clouds etc.

Author: Xianghui, 12, January 2022
"""
import torch
from os.path import join, split, splitext
from pytorch3d.structures import Meshes
from pytorch3d.io import save_ply, load_ply, load_obj, save_obj
import pickle as pkl
import numpy as np
from psbody.mesh import MeshViewer, Mesh
from lib.smpl.priors.th_hand_prior import mean_hand_pose
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.smpl.wrapper_pytorch import SMPLPyTorchWrapperBatch
from lib.smpl.const import *


class BaseFitter(object):
    def __init__(self, model_root, device='cuda:0', save_name='smpl', debug=False, hands=True):
        self.model_root = model_root # root path to the smpl or smplh model
        self.debug = debug
        self.save_name = save_name # suffix of the output file
        self.device = device
        self.hands = hands
        self.save_name_base = 'smplh' if hands else 'smpl'
        if debug:
            self.mv = MeshViewer(window_width=512, window_height=512)
        if self.hands:
            print("Using SMPL-H model for registration")
        else:
            print("Using SMPL model for registration")

    def fit(self, scans, gender='female', save_path=None):
        raise NotImplemented

    def optimize_pose_shape(self, th_scan_meshes, smpl, iterations, steps_per_iter):
        """
        optimize smpl pose and shape parameters together
        Args:
            th_scan_meshes:
            smpl:
            iterations:
            steps_per_iter:

        Returns:

        """
        raise NotImplemented

    def init_smpl(self, batch_sz, gender, pose=None, betas=None, trans=None, offsets=None, scale=None, flip=False):
        """
        initialize a smpl batch model
        Args:
            batch_sz:
            gender:
            flip: rotate smpl around z-axis by 180 degree, required for kinect point clouds, which has different coordinate
            from scans

        Returns: batch smplh model

        """
        # sp = SmplPaths(gender=gender)
        # smpl_faces = sp.get_faces()
        # th_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).to(self.device)
        num_betas = 10
        prior = get_prior(self.model_root, gender=gender, device=self.device)
        total_pose_num = SMPLH_POSE_PRAMS_NUM if self.hands else SMPL_POSE_PRAMS_NUM
        pose_init = torch.zeros((batch_sz, total_pose_num), device=self.device)

        if pose is None:
            # initialize hand pose from mean
            pose_init[:, 3:SMPLH_HANDPOSE_START] = prior.mean
            if self.hands:
                hand_mean = mean_hand_pose(self.model_root)
                hand_init = torch.tensor(hand_mean, dtype=torch.float).to(self.device)
            else:
                hand_init = torch.zeros((batch_sz, SMPL_HAND_POSE_NUM), device=self.device)
            pose_init[:, SMPLH_HANDPOSE_START:] = hand_init
            if flip:
                pose_init[:, 2] = np.pi
        else:
            pose_init[:, :SMPLH_HANDPOSE_START] = pose[:, :SMPLH_HANDPOSE_START]
            if pose.shape[1] == total_pose_num:
                pose_init[:, SMPLH_HANDPOSE_START:] = pose[:, SMPLH_HANDPOSE_START:]
        beta_init = torch.zeros((batch_sz, num_betas), device=self.device) if betas is None else betas
        trans_init = torch.zeros((batch_sz, 3), device=self.device) if trans is None else trans
        offsets_init = torch.zeros((batch_sz, 6890, 3), device=self.device) if offsets is None else offsets
        scale_init = torch.ones((batch_sz, 3), device=self.device) if scale is None else scale
        betas, pose, trans, scale = beta_init, pose_init, trans_init, scale_init
        # Init SMPL, pose with mean smpl pose, as in ch.registration
        # smpl = SMPLHPyTorchWrapperBatch(self.model_root, batch_sz, betas, pose, trans,
        #                                 num_betas=num_betas, device=self.device, gender=gender).to(self.device)
        smpl = SMPLPyTorchWrapperBatch(self.model_root, batch_sz, betas, pose, trans, offsets=offsets, scale=scale,
                                        num_betas=num_betas, device=self.device,
                                       gender=gender, hands=self.hands).to(self.device)
        return smpl

    @staticmethod
    def load_smpl_params(pkl_files):
        """
        load smpl params from file
        Args:
            pkl_files:

        Returns:

        """
        pose, betas, trans = [], [], []
        for spkl in pkl_files:
            smpl_dict = pkl.load(open(spkl, 'rb'), encoding='latin-1')
            p, b, t = smpl_dict['pose'], smpl_dict['betas'], smpl_dict['trans']
            pose.append(p) # smplh only allows 10 shape parameters
            # if len(b) == 10:
            #     temp = np.zeros((300,))
            #     temp[:10] = b
            #     b = temp.astype('float32')
            betas.append(b)
            trans.append(t)
        pose, betas, trans = np.array(pose), np.array(betas), np.array(trans)
        return pose, betas, trans

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'b2s': lambda cst, it: 10. ** 2 * cst * (1 + it),
                       's2b': lambda cst, it: 10. ** 2 * cst / (1 + it),
                       'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                       'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'lap': lambda cst, it: cst / (1 + it)
                       }
        return loss_weight

    def save_outputs(self, save_path, scan_paths, smpl, th_scan_meshes, save_name='smpl', transformation_path=None):
        th_smpl_meshes = self.smpl2meshes(smpl)
        mesh_paths, names = self.get_mesh_paths(save_name, save_path, scan_paths)
        if transformation_path is not None:
            data = np.load(transformation_path, allow_pickle=True).item()
            # Get the device of the vertices
            device = th_smpl_meshes.verts_list()[0].device
            
            center = data['center']
            scale = data['scale']
            # Convert scale and center to tensors on the same device as the vertices
            scale_tensor = torch.tensor(scale, device=device)
            center_tensor = torch.tensor(center, device=device)
            
            # Apply transformations to the mesh vertices
            th_smpl_meshes.verts_list()[0] = th_smpl_meshes.verts_list()[0] / scale_tensor + center_tensor
        self.save_meshes(th_smpl_meshes, mesh_paths)
        # self.save_meshes(th_scan_meshes, [join(save_path, n) for n in names]) # save original scans
        # Save params
        self.save_smpl_params(names, save_path, smpl, save_name)
        return smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(), smpl.trans.cpu().detach().numpy(), smpl.scale.cpu().detach().numpy(), smpl.offsets.cpu().detach().numpy()

    def smpl2meshes(self, smpl):
        "convert smpl batch to pytorch3d meshes"
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))
        return th_smpl_meshes

    def get_mesh_paths(self, save_name, save_path, scan_paths):
        names = [split(s)[1] for s in scan_paths]
        # Save meshes
        mesh_paths = []
        for n in names:
            if n.endswith('.obj'):
                mesh_paths.append(join(save_path, n.replace('.obj', f'_{save_name}.obj')))
            else:
                mesh_paths.append(join(save_path, n.replace('.ply', f'_{save_name}.obj')))
        return mesh_paths, names

    # def save_smpl_params(self, names, save_path, smpl, save_name):
    #     for p, b, t, n in zip(smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(),
    #                           smpl.trans.cpu().detach().numpy(), names):
    #         smpl_dict = {'pose': p, 'betas': b, 'trans': t}
    #         sfx = splitext(n)[1]
    #         pkl_file = join(save_path, n.replace(sfx, f'_{save_name}.pkl'))
    #         pkl.dump(smpl_dict, open(pkl_file, 'wb'))
    #         print('SMPL parameters saved to', pkl_file)

    # def save_smpl_params(self, names, save_path, smpl, save_name):
    #     for p, b, t, n, o in zip(smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(),
    #                           smpl.trans.cpu().detach().numpy(), names, smpl.offsets.cpu().detach().numpy()):
    #         smpl_dict = {'pose': p, 'betas': b, 'trans': t, 'offsets': o}
    #         sfx = splitext(n)[1]
    #         pkl_file = join(save_path, n.replace(sfx, f'_{save_name}.pkl'))
    #         pkl.dump(smpl_dict, open(pkl_file, 'wb'))
    #         print('SMPL parameters saved to', pkl_file)
    
    def save_smpl_params(self, names, save_path, smpl, save_name):
        for p, b, t, n, o, s in zip(smpl.pose.cpu().detach().numpy(), smpl.betas.cpu().detach().numpy(),
                                    smpl.trans.cpu().detach().numpy(), names, smpl.offsets.cpu().detach().numpy(),
                                    smpl.scale.cpu().detach().numpy()):
            smpl_dict = {'pose': p, 'betas': b, 'trans': t, 'offsets': o, 'scale': s}
            sfx = splitext(n)[1]
            pkl_file = join(save_path, n.replace(sfx, f'_{save_name}.pkl'))
            pkl.dump(smpl_dict, open(pkl_file, 'wb'))
            print('SMPL parameters saved to', pkl_file)


    @staticmethod
    def backward_step(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    @staticmethod
    def save_meshes(meshes, save_paths):
        print('Mesh saved at', save_paths[0])
        for m, s in zip(meshes, save_paths):
            save_obj(s, m.verts_list()[0].cpu(), m.faces_list()[0].cpu())

    @staticmethod
    def load_scans(scans, device='cuda:0', ret_cent=False):
        verts, faces, centers = [], [], []
        for scan in scans:
            print('scan path ...', scan)
            if scan.endswith('.ply'):
                v, f = load_ply(scan)
            else:
                v, f, _ = load_obj(scan)
                f = f[0]  # see pytorch3d doc
            verts.append(v)
            faces.append(f)
            centers.append(torch.mean(v, 0))
        th_scan_meshes = Meshes(verts, faces).to(device)
        if ret_cent:
            return th_scan_meshes, torch.stack(centers, 0).to(device)
        return th_scan_meshes

    def viz_fitting(self, smpl, th_scan_meshes, ind=0,
                    smpl_vc=np.array([0, 1, 0]), **kwargs):
        verts, _, _, _ = smpl()
        smpl_mesh = Mesh(v=verts[ind].cpu().detach().numpy(), f=smpl.faces.cpu().numpy())
        scan_mesh = Mesh(v=th_scan_meshes.verts_list()[ind].cpu().detach().numpy(),
                         f=th_scan_meshes.faces_list()[ind].cpu().numpy(), vc=smpl_vc)
        self.mv.set_dynamic_meshes([scan_mesh, smpl_mesh])

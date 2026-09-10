'''
Takes in smpl parms and initialises a smpl object with optimizable params.
class th_SMPL currently does not take batch dim.
If code works:
    Author: Bharat
else:
    Author: Anonymous
'''
import torch
import torch.nn as nn

from lib.smpl.smplpytorch.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from .const import *


class SMPLPyTorchWrapperBatch(nn.Module):
    def __init__(self, model_root, batch_sz,
                 betas=None, pose=None,
                 trans=None, offsets=None, scale=None,
                 gender='male', num_betas=300, hands=False,
                 device='cuda:0'):
        super(SMPLPyTorchWrapperBatch, self).__init__()
        self.model_root = model_root
        self.hands = hands # use smpl-h or not
        if scale is None:
            self.scale = nn.Parameter(torch.ones(batch_sz, 3))
        else:
            assert scale.ndim == 2
            self.scale = nn.Parameter(scale)

        if betas is None:
            self.betas = nn.Parameter(torch.zeros(batch_sz, 300))
        else:
            assert betas.ndim == 2
            self.betas = nn.Parameter(betas)
        pose_param_num = SMPLH_POSE_PRAMS_NUM if hands else SMPL_POSE_PRAMS_NUM
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(batch_sz, pose_param_num))
        else:
            assert pose.ndim == 2, f'the given pose shape {pose.shape} is not a batch pose'
            assert pose.shape[1] == pose_param_num, f'given pose param shape {pose.shape} ' \
                                                    f'does not match the model selected: hands={hands}'
            self.pose = nn.Parameter(pose)
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert trans.ndim == 2
            self.trans = nn.Parameter(trans)
        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(batch_sz, 6890, 3))
        else:
            assert offsets.ndim == 3
            self.offsets = nn.Parameter(offsets)

        # self.faces = faces
        self.gender = gender

        # pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender, num_betas=num_betas,
                               model_root=str(model_root), hands=hands)
        self.faces = self.smpl.th_faces.to(device) # XH: no need to input face, it is loaded from model file

    def forward(self):
        verts, jtr, tposed, naked = self.smpl(self.pose,
                                              th_betas=self.betas,
                                              th_trans=self.trans,
                                              th_offsets=self.offsets, scale=self.scale)
        return verts, jtr, tposed, naked


class SMPLPyTorchWrapper(nn.Module):
    "XH: this one is not used, why keeping it?"
    def __init__(self, model_root, betas=None, pose=None, trans=None, offsets=None, gender='male', num_betas=300):
        super(SMPLPyTorchWrapper, self).__init__()
        if betas is None:
            self.betas = nn.Parameter(torch.zeros(300,))
        else:
            self.betas = nn.Parameter(betas)
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(72,))
        else:
            self.pose = nn.Parameter(pose)
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(3,))
        else:
            self.trans = nn.Parameter(trans)
        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(6890, 3))
        else:
            self.offsets = nn.Parameter(offsets)

        ## pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender, num_betas=num_betas,
                               model_root=str(model_root))

    def forward(self):
        verts, Jtr, tposed, naked = self.smpl(self.pose.unsqueeze(axis=0),
                                              th_betas=self.betas.unsqueeze(axis=0),
                                              th_trans=self.trans.unsqueeze(axis=0),
                                              th_offsets=self.offsets.unsqueeze(axis=0))
        return verts[0]

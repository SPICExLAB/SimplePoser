"""
    Test the system with an example IMU measurement sequence.
"""


import torch
from config import paths
import os

import articulate as art
from model import MobilePoser
from config import acc_scale
from utils import load_yaml


def load_data(seq_num):
    data = torch.load(os.path.join('data', 'TotalCapture.pt'))
    acc = data['acc'][seq_num][:, :5] / acc_scale
    ori = data['ori'][seq_num][:, :5]
    imu = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)
    return imu


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config = load_yaml('config.yaml')
net = MobilePoser.from_pretrained(config, 'best.pt').to(device)
imu = load_data(seq_num=0)
pose, joints, tran, contact = net.forward_offline(imu.unsqueeze(0))
art.ParametricModel(paths.smpl_file).view_motion([pose], [tran])
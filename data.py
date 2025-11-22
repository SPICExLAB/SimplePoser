import math
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from typing import List
import random
import lightning as L
from tqdm import tqdm
from pathlib import Path

import articulate as art
from config import combos, paths, acc_scale, vel_scale, fps, joint_set


class PoseDataset(Dataset):
    def __init__(self, cfg: dict, fold: str='train', evaluate: str=None):
        super().__init__()
        self.cfg = cfg
        self.fold = fold
        self.evaluate = evaluate
        self.combos = combos
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)

        self.data = {
            'imu_inputs': [],
            'pose_outputs': [],
            'joint_outputs': [],
            'tran_outputs': [],
            'vel_outputs': [],
            'foot_outputs': [],
        }

        self._load_data()

    def _get_data_files(self, data_folder: Path):
        return [x.name for x in data_folder.iterdir() if not x.is_dir()]

    def _load_data(self):
        # load data files
        if self.evaluate:
            data_folder = Path(paths.dipimu_dir)
        else:
            data_folder = Path(paths.amass_dir)

        data_files = self._get_data_files(data_folder)

        # process each data file
        for data_file in tqdm(data_files):
            file_data = torch.load(data_folder / data_file, map_location=torch.device('cpu'), weights_only=True)
            self._process_file(file_data)

    def _process_file(self, file_data: dict):
        accs, oris, poses, trans = file_data['acc'], file_data['ori'], file_data['pose'], file_data['tran']
        joints = file_data.get('joint', [None] * len(poses))
        foots = file_data.get('contact', [None] * len(poses))
        
        for idx, (acc, ori, pose, tran, joint, foot) in enumerate(zip(accs, oris, poses, trans, joints, foots)):
            acc = acc[:, :5] / acc_scale  # (N, 5, 3), scale the acc to be in range [-1, 1]
            ori = ori[:, :5]                          # (N, 5, 3, 3)
            pose = pose.view(-1, 24, 3, 3)            # (N, 24, 3, 3)

            pose_global, joint = self.bodymodel.forward_kinematics(pose=pose) 
            if self.cfg['use_global_pose'] and not self.evaluate:
                # use global pose for training 
                pose = pose_global

            joint = joint.view(-1, 24, 3)                # (N, 24, 3)
            tran = tran.view(-1, 3)                      # (N, 3)
            foot = foot.view(-1, 2) if foot is not None else None  # (N, 2)

            self._process_data(acc, ori, pose, joint, tran, foot)

    def _process_data(self, acc, ori, pose, joint, tran, foot):
        for _, c in self.combos.items():
            # mask: zero out sensors not in this combo
            combo_acc = torch.zeros_like(acc)
            combo_ori = torch.zeros_like(ori)
            combo_acc[:, c] = acc[:, c]
            combo_ori[:, c] = ori[:, c]
            
            # flatten to [T, 60]: 5 sensors * (3 acc + 9 ori)
            imu = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)
            
            # split long sequences into windows
            window = len(imu) if self.evaluate else self.cfg['window_length'] # note: evaluation should use the entire sequence
            
            # store pose/joint/translation sequences
            self.data['imu_inputs'].extend(torch.split(imu, window))
            self.data['pose_outputs'].extend(torch.split(pose, window))
            self.data['joint_outputs'].extend(torch.split(joint, window))
            self.data['tran_outputs'].extend(torch.split(tran, window))
            
            # compute velocities from positions
            root_vel = torch.cat([torch.zeros(1, 3), tran[1:] - tran[:-1]])
            vel = torch.cat([torch.zeros(1, 24, 3), torch.diff(joint, dim=0)])
            vel[:, 0] = root_vel
            vel = vel * (fps / vel_scale)
            
            if not self.evaluate: # not necessary nor available for some datasets
                self.data['vel_outputs'].extend(torch.split(vel, window))
                self.data['foot_outputs'].extend(torch.split(foot, window))

    def __len__(self):
        return len(self.data['imu_inputs'])            

    def __getitem__(self, idx):
        imu = self.data['imu_inputs'][idx].float()
        joint = self.data['joint_outputs'][idx].float()
        tran = self.data['tran_outputs'][idx].float()
        vel = self.data['vel_outputs'][idx].float() if self.data['vel_outputs'] else None
        contact = self.data['foot_outputs'][idx].float() if self.data['foot_outputs'] else None
        
        # convert pose rotations to 6D representation
        pose_6d = art.math.rotation_matrix_to_r6d(self.data['pose_outputs'][idx])
        n_joints = len(joint_set.full)
        pose_6d = pose_6d.reshape(-1, n_joints, 6)[:, joint_set.full].reshape(-1, 6 * n_joints)
        
        return imu, pose_6d, joint, tran, vel, contact            


def collate_fn(batch):
    """Pad variable-length sequences for batching."""
    imus, poses, joints, trans, vels, contacts = zip(*batch)
    
    # pad all sequences to max length in batch
    imus = nn.utils.rnn.pad_sequence(imus, batch_first=True)
    poses = nn.utils.rnn.pad_sequence(poses, batch_first=True)
    joints = nn.utils.rnn.pad_sequence(joints, batch_first=True)
    trans = nn.utils.rnn.pad_sequence(trans, batch_first=True)
    vels = nn.utils.rnn.pad_sequence(vels, batch_first=True)
    contacts = nn.utils.rnn.pad_sequence(contacts, batch_first=True)
    
    # track original lengths 
    lengths = torch.tensor([len(x) for x in imus])
    
    return imus, poses, joints, trans, vels, contacts, lengths


def get_dataloaders(cfg, device):
    """Get train and validation dataloaders."""
    dataset = PoseDataset(cfg, fold='train')
    
    # 90/10 split
    n_train = int(0.9 * len(dataset))
    train_data, val_data = torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(cfg['seed'])
    )
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    
    # dataloaders
    loader_args = {
        'batch_size': cfg['batch_size'],
        'collate_fn': collate_fn,
        'num_workers': cfg['num_workers'],
        'pin_memory': device.type == 'cuda'
    }
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **loader_args)
    
    return train_loader, val_loader
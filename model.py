import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning as L
import numpy as np
from torch.nn.functional import relu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import articulate as art
from config import joint_set, gravity_velocity, paths, fps, vel_scale


class RNN(torch.nn.Module):
    """
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h=None):
        lengths = [_.shape[0] for _ in x]
        x = self.dropout(relu(self.linear1(x)))
        x, h = self.rnn(pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False))
        return self.linear2(pad_packed_sequence(x, batch_first=True)[0]), h


class Joints(nn.Module):
    """
    Input: IMU
    Output: 24 joint positions
    """
    def __init__(self):
        super().__init__()
        self.rnn = RNN(joint_set.n_imu, joint_set.n_full * 3, 256)

    def forward(self, x):
        joints, _ = self.rnn(x)
        return joints # [T, 24 * 3]


class Poser(nn.Module):
    """
    Input: IMU + joints
    Output: SMPL pose (6D rotations)
    """
    def __init__(self):
        super().__init__()
        self.rnn = RNN(joint_set.n_full * 3 + joint_set.n_imu, joint_set.n_reduced * 6, 256)

    def forward(self, x):
        pose, _ = self.rnn(x)
        return pose # [T, 24 * 6]


class FootContact(nn.Module):
    """
    Input: IMU + joints
    Output: foot contact probability [left, right]
    """
    def __init__(self):
        super().__init__()
        self.rnn = RNN(joint_set.n_full * 3 + joint_set.n_imu, 2, 64)

    def forward(self, x):
        contact, _ = self.rnn(x)
        return contact # [T, 2]


class Velocity(nn.Module):
    """
    Input: IMU
    Output: root velocity
    """
    def __init__(self):
        super().__init__()
        self.rnn = RNN(joint_set.n_full * 3 + joint_set.n_imu, joint_set.n_full * 3, 256, bidirectional=False)
        self.rnn_state = None

    def forward(self, x):
        vel, _ = self.rnn(x)
        return vel # [T, 24 * 3]

    def forward_online(self, x):
        """Stateful forward for online inference."""
        vel, self.rnn_state = self.rnn(x, self.rnn_state)
        return vel

    def reset(self):
        self.rnn_state = None


class MobilePoser(nn.Module):
    """
    Input: N IMUs
    Output: SMPL pose (6D rotations) + translation
    """
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.device = cfg['device']
        
        # body model
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.device)
        self.global_to_local_pose = self.bodymodel.inverse_kinematics_R
        
        # model components
        self.pose = Poser()
        self.joints = Joints()
        self.foot_contact = FootContact()
        self.velocity = Velocity()
        
        # base joints
        self.j, _ = self.bodymodel.get_zero_pose_joint_and_vertex()
        self.feet_pos = self.j[10:12].clone()
        self.floor_y = self.j[10:12, 1].min().item()
        
        # constants
        self.gravity_velocity = torch.tensor([0, gravity_velocity, 0]).to(self.device)
        self.prob_threshold = (0.5, 0.9)
        self.num_past_frames = cfg['past_frames']
        self.num_future_frames = cfg['future_frames']
        self.num_total_frames = self.num_past_frames + self.num_future_frames
        
        # online inference state
        self.last_lfoot_pos = self.feet_pos[0].to(self.device)
        self.last_rfoot_pos = self.feet_pos[1].to(self.device)
        self.last_root_pos = torch.zeros(3).to(self.device)
        self.last_joints = torch.zeros(24, 3).to(self.device)
        self.current_root_y = 0
        self.imu = None
        self.rnn_state = None

    @classmethod
    def from_pretrained(cls, cfg, model_path):
        """Load pretrained model."""
        model = cls(cfg)
        checkpoint = torch.load(model_path, map_location=cfg['device'], weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model

    def reset(self):
        """Reset online inference state."""
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_root_pos = torch.zeros(3).to(self.device)

    def _prob_to_weight(self, p):
        """Convert contact probability to blending weight."""
        return (p.clamp(self.prob_threshold[0], self.prob_threshold[1]) - self.prob_threshold[0]) / \
               (self.prob_threshold[1] - self.prob_threshold[0])

    def _reduced_pose_to_full(self, reduced_pose):
        """Transform reduced pose to full pose."""
        B, S = reduced_pose.shape[0], reduced_pose.shape[1]
        reduced_pose = reduced_pose.view(B, S, joint_set.n_reduced, 3, 3)
        full_pose = torch.eye(3, device=reduced_pose.device).repeat(B, S, 24, 1, 1)
        full_pose[:, :, joint_set.reduced] = reduced_pose
        full_pose = full_pose.view(B, S, -1)
        return full_pose

    def _reduced_global_to_full(self, reduced_pose):
        """Convert reduced 6D pose to full 24-joint local rotations."""
        pose = art.math.r6d_to_rotation_matrix(reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        pred_pose = self._reduced_pose_to_full(pose.unsqueeze(0)).squeeze(0).view(-1, 24, 3, 3)
        if self.cfg['use_global_pose']:
            pred_pose = self.global_to_local_pose(pred_pose)
        pred_pose[:, joint_set.ignored] = torch.eye(3, device=self.device)
        pred_pose[:, 0] = pose[:, 0]
        return pred_pose

    def forward(self, imu):
        """
        Simple forward pass for training.
        Args:
            imu: [B, T, 60] IMU features
        Returns:
            pred_pose: [B, T, 24*6] 6D rotations (raw)
            pred_joints: [B, T, 24*3] joint positions  
            pred_vel: [B, T, 24*3] joint velocities
            foot_contact: [B, T, 2] foot contact logits
        """
        # predict joints from IMU
        pred_joints = self.joints(imu)  # [B, T, 24*3]
        
        # predict pose from joints + IMU
        pose_input = torch.cat([pred_joints, imu], dim=-1)
        pred_pose = self.pose(pose_input)  # [B, T, len(reduced_joint_set)*6]
        
        # predict foot contact from joints + IMU  
        tran_input = torch.cat([pred_joints, imu], dim=-1)
        foot_contact = self.foot_contact(tran_input)  # [B, T, 2]
        
        # predict velocity from joints + IMU
        pred_vel = self.velocity(tran_input)  # [B, T, 24*3]
        
        return pred_pose, pred_joints, pred_vel, foot_contact

    def predict(self, imu):
        """
        Forward pass.
        Args:
            imu: [B, T, 60] IMU features
        Returns:
            pred_pose: [B, T, 24, 3, 3] local rotations
            pred_joints: [B, T, 24*3] joint positions
            pred_vel: [B, T, 24*3] joint velocities  
            foot_contact: [B, T, 2] foot contact probabilities
        """
        B, T = imu.shape[:2]

        # forward the joint prediction model
        pred_joints = self.joints(imu) # [B, T, 24*3]
        
        # forward the pose prediction model
        pose_input = torch.cat([pred_joints, imu], dim=-1)
        pred_pose = self.pose(pose_input) # [B, T, 24*6]

        # global pose to local
        pred_pose = self._reduced_global_to_full(pred_pose) # [B*T, 24, 3, 3]
        pred_pose = pred_pose.view(B, T, 24, 3, 3) # [B, T, 24, 3, 3]
        
        # forward the foot-ground contact probability model
        tran_input = torch.cat([pred_joints, imu], dim=-1)
        foot_contact = self.foot_contact(tran_input) # [B, T, 2]
        
        # forward the foot-joint velocity model
        pred_vel = self.velocity(tran_input) # [B, T, 24*3]
        
        return pred_pose, pred_joints, pred_vel, foot_contact

    @torch.no_grad()
    def forward_offline(self, imu):
        """
        Offline inference.
        Args:
            imu: [B, T, 60] IMU features
        Returns:
            pose: [B*T, 24, 3, 3] local rotations
            joints: [B*T, 24*3] joint positions
            tran: [B*T, 3] root translations
            contact: [B*T, 2] foot contact probabilities
        """
        # forward the prediction model
        pose, joints, vel, contact = self.predict(imu)
        
        B, T = imu.shape[:2]
        pose = pose.view(B * T, 24, 3, 3)        # [B*T, 24, 3, 3]
        joints = joints.view(B * T, 24, 3)       # [B*T, 24, 3]
        contact = contact.view(B * T, 2)         # [B*T, 2]
        vel = vel.view(B * T, 24, 3)             # [B*T, 24, 3]
        
        # calculate velocity from foot-ground contact
        floor_y = self.j[10:12, 1].min().item()
        contact_vel = self.gravity_velocity + art.math.lerp(
            torch.cat([torch.zeros(1, 3).to(self.device), joints[:-1, 10] - joints[1:, 10]]),
            torch.cat([torch.zeros(1, 3).to(self.device), joints[:-1, 11] - joints[1:, 11]]),
            contact.max(dim=1).indices.view(-1, 1)
        )
        
        # velocity from network-based estimation
        root_vel = vel[:, 0]  # [T, 3]
        pred_vel = root_vel * (vel_scale / fps)
        
        # compute velocity as weighted combination of network-based and foot-contact-based
        weight = self._prob_to_weight(contact.max(dim=1).values.sigmoid()).view(-1, 1)
        velocity = art.math.lerp(pred_vel, contact_vel, weight)
        
        # remove penetration
        current_root_y = 0
        for i in range(velocity.shape[0]):
            current_foot_y = current_root_y + joints[i, 10:12, 1].min().item()
            if current_foot_y + velocity[i, 1].item() <= floor_y:
                velocity[i, 1] = floor_y - current_foot_y
            current_root_y += velocity[i, 1].item()
        
        # velocity to root position
        tran = torch.stack([velocity[:i+1].sum(dim=0) for i in range(velocity.shape[0])])
        
        return pose, joints, tran, contact

    @torch.no_grad()
    def forward_online(self, data):
        """
        Online inference with temporal smoothing.
        Args:
            data: [60] single IMU frame
        Returns:
            pose: [24, 3, 3] local rotations (single frame)
            pred_joints: [24, 3] joint positions (single frame)
            last_root_pos: [3] root translation
            contact: [2] foot contact probabilities
        """
        imu = data.repeat(self.num_total_frames, 1) if self.imu is None else torch.cat((self.imu[1:], data.view(1, -1))) # [num_total_frames, 60]
        
        # forward the pose prediction model
        pred_pose, pred_joints, pred_vel, pred_contact = self.forward(imu.unsqueeze(0))

        # extract current frame
        pose = pred_pose.view(-1, 24, 3, 3)[self.num_past_frames]   # [24, 3, 3]
        joints = pred_joints.view(-1, 24, 3)[self.num_past_frames]  # [24, 3]
        contact = pred_contact.view(-1, 2)[self.num_past_frames]    # [2]
        vel = pred_vel.view(-1, 24, 3)[self.num_past_frames]        # [24, 3]
        
        # compute translation from foot-contact probability
        lfoot_pos, rfoot_pos = joints[10], joints[11]
        if contact[0] > contact[1]:
            contact_vel = self.last_lfoot_pos - lfoot_pos + self.gravity_velocity
        else:
            contact_vel = self.last_rfoot_pos - rfoot_pos + self.gravity_velocity
        
        # velocity from network-based estimation
        root_vel = vel[0] / (fps / vel_scale)  # root joint velocity
        weight = self._prob_to_weight(contact.max())
        velocity = art.math.lerp(root_vel, contact_vel, weight)
        
        # remove penetration
        current_foot_y = self.current_root_y + min(lfoot_pos[1].item(), rfoot_pos[1].item())
        if current_foot_y + velocity[1].item() <= self.floor_y:
            velocity[1] = self.floor_y - current_foot_y
        
        # update state
        self.current_root_y += velocity[1].item()
        self.last_lfoot_pos, self.last_rfoot_pos = lfoot_pos, rfoot_pos
        self.imu = imu
        self.last_root_pos += velocity
        
        return pose, joints, self.last_root_pos.clone(), contact
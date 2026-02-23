import torch
import torch.nn as nn

import articulate as art
from model import RNN
from config import joint_set, paths
from dynaip.utils import r6d_to_local


class SubPoser(nn.Module):
    """Velocity RNN -> Pose RNN chain."""
    def __init__(self, n_input, v_output, p_output, n_hidden, n_rnn_layer, dropout, n_glb):
        super().__init__()
        self.n_glb = n_glb
        self.rnn_v = RNN(n_input - n_glb, v_output, n_hidden, n_rnn_layer, bidirectional=True, dropout=dropout)
        self.rnn_p = RNN(n_input + v_output, p_output, n_hidden, n_rnn_layer, bidirectional=True, dropout=dropout)

    def forward(self, x):
        x_local = x[:, :, :-self.n_glb] if self.n_glb > 0 else x
        v, _ = self.rnn_v(x_local)
        p, _ = self.rnn_p(torch.cat([x, v], dim=-1))
        return v, p


class DynaIP(nn.Module):
    """DynaIP body-part decomposed pose + velocity prediction."""

    def __init__(self, device='cpu'):
        super().__init__()
        n_hidden = 200
        n_rnn_layer = 2
        dropout = 0.2
        n_glb = 6

        self.sensor_names = ['LeftWrist', 'RightWrist', 'LeftPocket', 'RightPocket', 'Head']
        self.v_names = ['Pelvis', 'LeftFoot', 'RightFoot', 'Head', 'LeftWrist', 'RightWrist']
        self.p_names = ['Pelvis', 'LeftUpperLeg', 'RightUpperLeg', 'Spine1', 'LeftKnee', 'RightKnee',
                        'Spine2', 'Spine3', 'Neck', 'LeftShoulder', 'RightShoulder', 'Head',
                        'LeftUpperArm', 'RightUpperArm', 'LeftElbow', 'RightElbow']

        self.generate_indices_list()

        m = art.model.ParametricModel(paths.smpl_file, device=device)
        self.forward_kinematics = m.forward_kinematics
        self.global_to_local_pose = m.inverse_kinematics_R

        self.glb = RNN(n_input=joint_set.n_imu, n_output=n_glb, n_hidden=36, n_rnn_layer=1, dropout=dropout)
        self.posers = nn.ModuleList([
            SubPoser(n_input=24 + n_glb, v_output=6,  p_output=36, n_hidden=n_hidden, n_rnn_layer=n_rnn_layer, dropout=dropout, n_glb=n_glb),  # arms: 2 sensors, 2 vel, 6 pose
            SubPoser(n_input=36 + n_glb, v_output=12, p_output=30, n_hidden=n_hidden, n_rnn_layer=n_rnn_layer, dropout=dropout, n_glb=n_glb),  # legs: 3 sensors, 4 vel, 5 pose
            SubPoser(n_input=12 + n_glb, v_output=6,  p_output=30, n_hidden=n_hidden, n_rnn_layer=n_rnn_layer, dropout=dropout, n_glb=n_glb),  # spine: 1 sensor, 2 vel, 5 pose
        ])

    def find_indices(self, elements, lst):
        indices = []
        for element in elements:
            if element in lst:
                indices.append(lst.index(element))
        return indices

    def generate_indices_list(self):
        self.posers_config = [
            {'sensor': ['LeftWrist', 'RightWrist'],
             'velocity': ['LeftWrist', 'RightWrist'],
             'pose': ['LeftShoulder', 'LeftUpperArm', 'LeftElbow', 'RightShoulder', 'RightUpperArm', 'RightElbow']},

            {'sensor': ['LeftPocket', 'RightPocket', 'Head'],
             'velocity': ['Pelvis', 'LeftFoot', 'RightFoot', 'Head'],
             'pose': ['Pelvis', 'LeftUpperLeg', 'RightUpperLeg', 'LeftKnee', 'RightKnee']},

            {'sensor': ['Head'],
             'velocity': ['Pelvis', 'Head'],
             'pose': ['Spine1', 'Spine2', 'Spine3', 'Neck', 'Head']},
        ]

        smpl_vel_map = {'Pelvis': 0, 'LeftFoot': 10, 'RightFoot': 11, 'Head': 15,
                        'LeftWrist': 20, 'RightWrist': 21}

        self.indices = []
        self.vel_indices = []
        for i in range(len(self.posers_config)):
            temp = {'sensor_indices': self.find_indices(self.posers_config[i]['sensor'], self.sensor_names),
                    'v_indices': self.find_indices(self.posers_config[i]['velocity'], self.v_names),
                    'p_indices': self.find_indices(self.posers_config[i]['pose'], self.p_names)}
            self.indices.append(temp)
            self.vel_indices.extend(smpl_vel_map[name] for name in self.posers_config[i]['velocity'])

    def _imu_to_sensors(self, imu):
        B, T = imu.shape[:2]
        acc = imu[:, :, :15].view(B, T, 5, 3)
        ori = imu[:, :, 15:].view(B, T, 5, 9)
        return torch.cat([acc, ori], dim=-1)

    def forward(self, imu):
        """
        Args:
            imu: [B, T, 60] raw IMU features
        Returns:
            vel:  [B, T, 24] intermediate part velocities (8 joints x 3)
            pose: [B, T, 96] 6D rotations (16 reduced joints)
        """
        B, T = imu.shape[:2]
        glb, _ = self.glb(imu)
        sensors = self._imu_to_sensors(imu)

        pose = imu.new_zeros(B, T, len(self.p_names), 6)
        vel_parts = []
        for i in range(len(self.posers)):
            idx = self.indices[i]
            local = sensors[:, :, idx['sensor_indices']].flatten(2)
            v, p = self.posers[i](torch.cat([local, glb], dim=-1))
            pose[:, :, idx['p_indices']] = p.view(B, T, len(idx['p_indices']), 6)
            vel_parts.append(v)

        return torch.cat(vel_parts, dim=-1), pose.view(B, T, -1)

    @torch.no_grad()
    def predict(self, imu):
        """Convert raw 6D pose output to full 24-joint local rotations."""
        B, T = imu.shape[:2]
        _, pred_pose = self.forward(imu)
        local = r6d_to_local(pred_pose, self.global_to_local_pose)
        return local.view(B, T, 24, 3, 3)

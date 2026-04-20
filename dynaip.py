import torch
import torch.nn as nn
import torch.nn.functional as F

import articulate as art
from config import joint_set, paths
from mobileposer import RNN


def r6d_to_local(pred_pose, global_to_local_pose):
    """Reduced 6D rotations -> full 24-joint local rotation matrices."""
    pose = art.math.r6d_to_rotation_matrix(pred_pose).view(-1, joint_set.n_reduced, 3, 3)
    full = torch.eye(3, device=pose.device).expand(pose.shape[0], 24, 3, 3).clone()
    full[:, joint_set.reduced] = pose
    local = global_to_local_pose(full)
    local[:, joint_set.ignored] = torch.eye(3, device=pose.device)
    local[:, 0] = pose[:, 0]
    return local


class SubPoser(nn.Module):
    """Part-branch: sensor+glb -> velocity -> pose feature."""
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
    """Part-decomposed pose estimator with translation head (vrnet)."""
    dt = 1 / 30
    beta_velocity = 1.0
    contact_joints = [0, 10, 11, 20, 21]  # pelvis, left foot, right foot, left hand, right hand

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        device = cfg['device']

        n_hidden = 200
        n_rnn_layer = 2
        dropout = 0.2
        n_glb = 24
        n_parts = 3
        token_dim = 128

        self.sensor_names = ['LeftWrist', 'RightWrist', 'LeftPocket', 'RightPocket', 'Head']
        self.v_names = ['Pelvis', 'LeftFoot', 'RightFoot', 'Head', 'LeftWrist', 'RightWrist']
        self.p_names = ['Pelvis', 'LeftUpperLeg', 'RightUpperLeg', 'Spine1', 'LeftKnee', 'RightKnee',
                        'Spine2', 'Spine3', 'Neck', 'LeftShoulder', 'RightShoulder', 'Head',
                        'LeftUpperArm', 'RightUpperArm', 'LeftElbow', 'RightElbow']

        self._generate_indices_list()

        m = art.model.ParametricModel(paths.smpl_file, device=device)
        self.forward_kinematics = m.forward_kinematics
        self.global_to_local_pose = m.inverse_kinematics_R

        self.glb = RNN(joint_set.n_imu, n_glb, n_hidden=64, n_rnn_layer=2, bidirectional=True, dropout=dropout)

        part_feat_dim = 64
        self.posers = nn.ModuleList([
            SubPoser(n_input=24 + n_glb, v_output=6,  p_output=part_feat_dim, n_hidden=n_hidden, n_rnn_layer=n_rnn_layer, dropout=dropout, n_glb=n_glb),  # arms
            SubPoser(n_input=36 + n_glb, v_output=12, p_output=part_feat_dim, n_hidden=n_hidden, n_rnn_layer=n_rnn_layer, dropout=dropout, n_glb=n_glb),  # legs
            SubPoser(n_input=12 + n_glb, v_output=6,  p_output=part_feat_dim, n_hidden=n_hidden, n_rnn_layer=n_rnn_layer, dropout=dropout, n_glb=n_glb),  # spine
        ])

        self.part_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(part_feat_dim, token_dim), nn.LeakyReLU())
            for _ in range(n_parts)
        ])

        self.pose_head = nn.Sequential(
            nn.Linear(token_dim * n_parts, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(self.p_names) * 6)
        )

        self.vrnet = RNN(
            n_input=joint_set.n_imu + 24 * 3 + 96,
            n_output=3 + len(self.contact_joints),
            n_hidden=n_hidden,
            n_rnn_layer=n_rnn_layer,
            bidirectional=True,
            dropout=dropout
        )

    @classmethod
    def from_pretrained(cls, cfg, model_path):
        model = cls(cfg)
        checkpoint = torch.load(model_path, map_location=cfg['device'], weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model

    @staticmethod
    def _find_indices(elements, lst):
        return [lst.index(e) for e in elements if e in lst]

    def _generate_indices_list(self):
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
        for pc in self.posers_config:
            self.indices.append({
                'sensor_indices': self._find_indices(pc['sensor'], self.sensor_names),
                'v_indices': self._find_indices(pc['velocity'], self.v_names),
                'p_indices': self._find_indices(pc['pose'], self.p_names),
            })
            self.vel_indices.extend(smpl_vel_map[n] for n in pc['velocity'])

    def _pose_to_joints(self, pose_6d):
        """FK joint positions from 6D pose (no root override)."""
        B, T = pose_6d.shape[:2]
        rot = art.math.r6d_to_rotation_matrix(pose_6d).view(B * T, joint_set.n_reduced, 3, 3)
        full = torch.eye(3, device=pose_6d.device).expand(B * T, 24, 3, 3).clone()
        full[:, joint_set.reduced] = rot
        local = self.global_to_local_pose(full)
        local[:, joint_set.ignored] = torch.eye(3, device=pose_6d.device)
        _, joints = self.forward_kinematics(local)
        return joints[:, :24].contiguous().view(B, T, 24, 3)

    @staticmethod
    def _imu_to_sensors(imu):
        B, T = imu.shape[:2]
        acc = imu[:, :, :15].view(B, T, 5, 3)
        ori = imu[:, :, 15:].view(B, T, 5, 9)
        return torch.cat([acc, ori], dim=-1)

    def forward(self, imu):
        """
        Returns:
            vel:      [B, T, 24] part-joint velocities
            pose:     [B, T, 96] reduced 6D rotations
            root_vel: [B, T, 3]  root velocity
            stat:     [B, T, 5]  stationary logits
        """
        B, T = imu.shape[:2]
        glb, _ = self.glb(imu)
        sensors = self._imu_to_sensors(imu)

        vel_parts, part_features = [], []
        for i, poser in enumerate(self.posers):
            local = sensors[:, :, self.indices[i]['sensor_indices']].flatten(2)
            v, p = poser(torch.cat([local, glb], dim=-1))
            vel_parts.append(v)
            part_features.append(p)
        vel = torch.cat(vel_parts, dim=-1)

        tokens = torch.cat([proj(f) for proj, f in zip(self.part_projections, part_features)], dim=-1)
        pose = self.pose_head(tokens)

        joints = self._pose_to_joints(pose.detach()).view(B, T, -1)
        tran_out, _ = self.vrnet(torch.cat([imu, joints, pose.detach()], dim=-1))
        root_vel, stat = tran_out[:, :, :3], tran_out[:, :, 3:]
        return vel, pose, root_vel, stat

    def compute_loss(self, batch, device, finetune=False):
        """Training loss: pose (reduced 6D) + part velocity + FK joints + translation + stationarity."""
        imu, pose_6d, joints, tran, vel, contact, stationary, root_vel, lengths = batch
        imu = imu.to(device)
        pose_6d = pose_6d.to(device)
        vel = vel.to(device)
        joints = joints.to(device)
        stationary = stationary.to(device)
        root_vel = root_vel.to(device)

        B, T = pose_6d.shape[:2]
        gt_pose = pose_6d.view(B, T, 24, 6)[:, :, joint_set.reduced].view(B, T, -1)
        gt_vel = vel.view(B, T, 24, 3)[:, :, self.vel_indices].flatten(2)
        gt_joints = joints.view(B, T, 24, 3)

        pred_vel, pred_pose, pred_root_vel, pred_stat = self(imu)
        local = r6d_to_local(pred_pose, self.global_to_local_pose)
        _, pred_j = self.forward_kinematics(local)
        pred_j = pred_j.view(B, T, 24, 3)

        pose_loss = F.mse_loss(pred_vel, gt_vel) + F.mse_loss(pred_pose, gt_pose)
        joint_loss = F.mse_loss(pred_j, gt_joints)
        root_vel_loss = F.mse_loss(pred_root_vel, root_vel)
        stat_loss = F.binary_cross_entropy_with_logits(pred_stat, stationary)

        loss = pose_loss + joint_loss + root_vel_loss + stat_loss
        return loss, {
            'pose': pose_loss.item(),
            'joint': joint_loss.item(),
            'root_vel': root_vel_loss.item(),
            'stat': stat_loss.item(),
        }

    @torch.no_grad()
    def compute_val_loss(self, batch, device, finetune=False):
        return self.compute_loss(batch, device, finetune=finetune)

    @staticmethod
    def _stationary_weight(prob):
        """Soft thresholding: 0 below p=0.6, ramps to 1 at p=0.8."""
        return (prob * 5 - 3).clamp(0, 1)

    @torch.no_grad()
    def predict(self, imu):
        """
        Args:
            imu: [1, T, 60]
        Returns:
            pose:      [T, 24, 3, 3] local rotations
            tran:      [T, 3] root translation
            stat_prob: [T, 5] stationary probabilities
        """
        T = imu.shape[1]
        device = imu.device

        _, pred_pose, root_vel, stat_logits = self(imu)
        root_vel = root_vel.squeeze(0)
        stat_prob = torch.sigmoid(stat_logits.squeeze(0))

        pose = r6d_to_local(pred_pose, self.global_to_local_pose).view(T, 24, 3, 3)
        _, joints = self.forward_kinematics(pose)
        cjoint = joints[:, self.contact_joints]

        floor_y = cjoint[0, 1:3, 1].min().item()
        tran = torch.zeros(T, 3, device=device)
        last_cjoint = cjoint[0].clone()
        contact = torch.zeros(5, dtype=torch.bool, device=device)
        contact_counter = torch.zeros(5, dtype=torch.int, device=device)

        for t in range(1, T):
            stationary = stat_prob[t] > 0.7
            sw = self._stationary_weight(stat_prob[t])

            if stationary[0]:
                lleg = joints[t, 4] - joints[t, 1]
                rleg = joints[t, 5] - joints[t, 2]
                if min(torch.arccos((-lleg[1] / lleg.norm()).clamp(-1, 1)),
                       torch.arccos((-rleg[1] / rleg.norm()).clamp(-1, 1))) < torch.pi / 4:
                    stationary[0], sw[0] = False, 0.0

            cworld = tran[t-1] + cjoint[t]
            contact_new = stationary & (contact | (cworld[:, 1] < floor_y + 0.05))
            if contact_new.any():
                hdist = (cworld[:, 1] - cworld[contact_new, 1].unsqueeze(1)).abs().min(dim=0).values
                contact_new |= stationary & (hdist < 0.05)

            potential = stationary & ~contact_new
            contact_counter = torch.where(potential, contact_counter + 1, torch.zeros_like(contact_counter))
            contact_new |= contact_counter >= 5
            contact = contact_new & stationary

            sw_c = sw * contact.float()
            velocity = (sw_c @ (last_cjoint - cjoint[t]) / self.dt + self.beta_velocity * root_vel[t]) \
                       / (self.beta_velocity + sw_c.sum())
            last_cjoint = cjoint[t].clone()

            if contact[1:3].any():
                velocity[1] = 0.0

            tran[t] = tran[t-1] + velocity * self.dt

            cworld = tran[t] + cjoint[t]
            near_ground = contact & (cworld[:, 1] < floor_y + 0.15)
            if near_ground.any():
                tran[t, 1] += (floor_y - cworld[near_ground, 1].min().item()) * 0.1

            lowest = (tran[t, 1] + cjoint[t, :, 1]).min().item()
            if lowest < floor_y:
                tran[t, 1] += floor_y - lowest

        return pose, tran - tran[:1], stat_prob

import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import articulate as art
from config import joint_set, gravity_velocity, paths, fps, vel_scale
from loss import compute_vel_loss, compute_jerk_loss


class RNN(torch.nn.Module):
    """Linear -> (Bi)LSTM -> Linear."""
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super().__init__()
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
    """IMU -> 24 joint positions."""
    def __init__(self):
        super().__init__()
        self.rnn = RNN(joint_set.n_imu, joint_set.n_full * 3, 256)

    def forward(self, x):
        joints, _ = self.rnn(x)
        return joints


class Poser(nn.Module):
    """IMU + joints -> SMPL pose (6D)."""
    def __init__(self):
        super().__init__()
        self.rnn = RNN(joint_set.n_full * 3 + joint_set.n_imu, joint_set.n_reduced * 6, 256)

    def forward(self, x):
        pose, _ = self.rnn(x)
        return pose


class FootContact(nn.Module):
    """IMU + joints -> foot contact probability [left, right]."""
    def __init__(self):
        super().__init__()
        self.rnn = RNN(joint_set.n_full * 3 + joint_set.n_imu, 2, 64)

    def forward(self, x):
        contact, _ = self.rnn(x)
        return contact


class Velocity(nn.Module):
    """IMU + joints -> root velocity."""
    def __init__(self):
        super().__init__()
        self.rnn = RNN(joint_set.n_full * 3 + joint_set.n_imu, joint_set.n_full * 3, 256, bidirectional=False)
        self.rnn_state = None

    def forward(self, x):
        vel, _ = self.rnn(x)
        return vel

    def forward_online(self, x):
        vel, self.rnn_state = self.rnn(x, self.rnn_state)
        return vel

    def reset(self):
        self.rnn_state = None


class MobilePoser(nn.Module):
    """N IMUs -> SMPL pose (6D) + translation. Four heads trained independently."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg['device']

        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=self.device)
        self.global_to_local_pose = self.bodymodel.inverse_kinematics_R

        j, _ = self.bodymodel.get_zero_pose_joint_and_vertex()
        b = art.math.joint_position_to_bone_vector(j[joint_set.lower_body].unsqueeze(0),
                                                   joint_set.lower_body_parent).squeeze(0)
        bone_orientation, bone_length = art.math.normalize_tensor(b, return_norm=True)
        b = bone_orientation * bone_length
        b[:3] = 0
        self.lower_body_bone = b

        self.pose = Poser()
        self.joints = Joints()
        self.foot_contact = FootContact()
        self.velocity = Velocity()

        self.j, _ = self.bodymodel.get_zero_pose_joint_and_vertex()
        self.feet_pos = self.j[10:12].clone()
        self.floor_y = self.j[10:12, 1].min().item()

        self.gravity_velocity = torch.tensor([0, gravity_velocity, 0]).to(self.device)
        self.prob_threshold = (0.5, 0.9)
        self.num_past_frames = cfg['past_frames']
        self.num_future_frames = cfg['future_frames']
        self.num_total_frames = self.num_past_frames + self.num_future_frames

        self.last_lfoot_pos = self.feet_pos[0].to(self.device)
        self.last_rfoot_pos = self.feet_pos[1].to(self.device)
        self.last_root_pos = torch.zeros(3).to(self.device)
        self.last_joints = torch.zeros(24, 3).to(self.device)
        self.current_root_y = 0
        self.imu = None
        self.rnn_state = None

    @classmethod
    def from_pretrained(cls, cfg, model_path):
        model = cls(cfg)
        checkpoint = torch.load(model_path, map_location=cfg['device'], weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model

    def reset(self):
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_root_pos = torch.zeros(3).to(self.device)

    def _prob_to_weight(self, p):
        return (p.clamp(self.prob_threshold[0], self.prob_threshold[1]) - self.prob_threshold[0]) / \
               (self.prob_threshold[1] - self.prob_threshold[0])

    def _reduced_pose_to_full(self, reduced_pose):
        B, S = reduced_pose.shape[0], reduced_pose.shape[1]
        reduced_pose = reduced_pose.view(B, S, joint_set.n_reduced, 3, 3)
        full_pose = torch.eye(3, device=reduced_pose.device).repeat(B, S, 24, 1, 1)
        full_pose[:, :, joint_set.reduced] = reduced_pose
        full_pose = full_pose.view(B, S, -1)
        return full_pose

    def _reduced_global_to_full(self, reduced_pose):
        pose = art.math.r6d_to_rotation_matrix(reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        pred_pose = self._reduced_pose_to_full(pose.unsqueeze(0)).squeeze(0).view(-1, 24, 3, 3)
        if self.cfg['use_global_pose']:
            pred_pose = self.global_to_local_pose(pred_pose)
        pred_pose[:, joint_set.ignored] = torch.eye(3, device=self.device)
        pred_pose[:, 0] = pose[:, 0]
        return pred_pose

    def forward(self, imu):
        """Full-pipeline forward: joints -> pose, contact, velocity (raw outputs)."""
        pred_joints = self.joints(imu)
        aux = torch.cat([pred_joints, imu], dim=-1)
        pred_pose = self.pose(aux)
        foot_contact = self.foot_contact(aux)
        pred_vel = self.velocity(aux)
        return pred_pose, pred_joints, pred_vel, foot_contact

    def predict(self, imu):
        """Forward with pose decoded to local 24-joint rotation matrices."""
        B, T = imu.shape[:2]
        pred_joints = self.joints(imu)
        pred_pose = self.pose(torch.cat([pred_joints, imu], dim=-1))
        pred_pose = self._reduced_global_to_full(pred_pose).view(B, T, 24, 3, 3)
        aux = torch.cat([pred_joints, imu], dim=-1)
        foot_contact = self.foot_contact(aux)
        pred_vel = self.velocity(aux)
        return pred_pose, pred_joints, pred_vel, foot_contact

    def compute_loss(self, batch, device, finetune=False):
        """Training loss: four heads trained independently with noisy GT joints."""
        imu, pose_6d, joints, tran, vel, contact, stationary, root_vel, lengths = batch
        imu = imu.to(device)
        joints = joints.to(device)
        pose_6d = pose_6d.to(device)

        B, T = pose_6d.shape[:2]
        pose_6d = pose_6d.view(B, T, 24, 6)[:, :, joint_set.reduced].view(B, T, -1)

        pose_noise = torch.randn_like(joints) * 0.04
        contact_noise = torch.randn_like(joints) * 0.04
        vel_noise = torch.randn_like(joints) * 0.025
        noisy_joints_pose = (joints + pose_noise).view(B, T, -1)
        noisy_joints_contact = (joints + contact_noise).view(B, T, -1)
        noisy_joints_vel = (joints + vel_noise).view(B, T, -1)

        pred_joints = self.joints(imu)
        pred_pose = self.pose(torch.cat([noisy_joints_pose, imu], dim=-1))

        joints_loss = nn.MSELoss()(pred_joints, joints.view(B, T, -1))
        pose_loss = nn.MSELoss()(pred_pose, pose_6d)
        jerk = compute_jerk_loss(pred_pose) + compute_jerk_loss(pred_joints)
        loss = joints_loss + pose_loss + 1e-5 * jerk
        metrics = {'joints': joints_loss.item(), 'pose': pose_loss.item()}

        if not finetune:
            contact = contact.to(device)
            vel = vel.to(device)
            pred_contact = self.foot_contact(torch.cat([noisy_joints_contact, imu], dim=-1))
            pred_vel = self.velocity(torch.cat([noisy_joints_vel, imu], dim=-1))
            contact_loss = nn.BCEWithLogitsLoss()(pred_contact, contact)
            vel_loss = sum(compute_vel_loss(pred_vel, vel.view(B, T, -1), i) for i in [1, 3, 9])
            loss = loss + contact_loss + 0.5 * vel_loss
            metrics['contact'] = contact_loss.item()
            metrics['velocity'] = vel_loss.item()

        return loss, metrics

    @torch.no_grad()
    def compute_val_loss(self, batch, device, finetune=False):
        """Validation loss: full forward (no noise injection)."""
        imu, pose_6d, joints, tran, vel, contact, stationary, root_vel, lengths = batch
        imu = imu.to(device)
        pose_6d = pose_6d.to(device)
        joints = joints.to(device)

        B, T = pose_6d.shape[:2]
        pose_6d = pose_6d.view(B, T, 24, 6)[:, :, joint_set.reduced].view(B, T, -1)

        pred_pose, pred_joints, pred_vel, pred_contact = self(imu)
        joints_loss = nn.MSELoss()(pred_joints, joints.view(B, T, -1))
        pose_loss = nn.MSELoss()(pred_pose, pose_6d)
        loss = joints_loss + pose_loss
        metrics = {'joints': joints_loss.item(), 'pose': pose_loss.item()}

        if not finetune:
            contact = contact.to(device)
            vel = vel.to(device)
            contact_loss = nn.BCEWithLogitsLoss()(pred_contact, contact)
            vel_loss = 0.2 * sum(compute_vel_loss(pred_vel, vel.view(B, T, -1), i) for i in [1, 3, 9])
            loss = loss + contact_loss + vel_loss
            metrics['contact'] = contact_loss.item()
            metrics['velocity'] = vel_loss.item()

        return loss, metrics

    @torch.no_grad()
    def forward_offline(self, imu):
        """Offline inference: pose, joints, tran, contact for the full sequence."""
        pose, joints, vel, contact = self.predict(imu)
        B, T = imu.shape[:2]
        pose = pose.view(B * T, 24, 3, 3)
        joints = joints.view(B * T, 24, 3)
        contact = contact.view(B * T, 2)
        vel = vel.view(B * T, 24, 3)

        j = art.math.forward_kinematics(pose[:, joint_set.lower_body],
                                        self.lower_body_bone.expand(pose.shape[0], -1, -1),
                                        joint_set.lower_body_parent)[1]
        contact_vel = self.gravity_velocity + art.math.lerp(
            torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 7] - j[1:, 7])),
            torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 8] - j[1:, 8])),
            contact.max(dim=1).indices.view(-1, 1)
        )

        root_vel = vel[:, 0]
        pred_vel = root_vel * (vel_scale / fps)
        weight = self._prob_to_weight(contact.max(dim=1).values.sigmoid()).view(-1, 1)
        velocity = art.math.lerp(pred_vel, contact_vel, weight)

        floor_y = self.j[10:12, 1].min().item()
        current_root_y = 0
        for i in range(velocity.shape[0]):
            current_foot_y = current_root_y + j[i, 7:9, 1].min().item()
            if current_foot_y + velocity[i, 1].item() <= floor_y:
                velocity[i, 1] = floor_y - current_foot_y
            current_root_y += velocity[i, 1].item()

        tran = torch.stack([velocity[:i+1].sum(dim=0) for i in range(velocity.shape[0])])
        return pose, joints, tran, contact

    @torch.no_grad()
    def forward_online(self, data):
        """Online inference: single frame in, single frame out with temporal smoothing."""
        imu = data.repeat(self.num_total_frames, 1) if self.imu is None else torch.cat((self.imu[1:], data.view(1, -1)))
        pred_pose, pred_joints, pred_vel, pred_contact = self.forward(imu.unsqueeze(0))

        pose = pred_pose.view(-1, 24, 3, 3)[self.num_past_frames]
        joints = pred_joints.view(-1, 24, 3)[self.num_past_frames]
        contact = pred_contact.view(-1, 2)[self.num_past_frames]
        vel = pred_vel.view(-1, 24, 3)[self.num_past_frames]

        lfoot_pos, rfoot_pos = joints[10], joints[11]
        if contact[0] > contact[1]:
            contact_vel = self.last_lfoot_pos - lfoot_pos + self.gravity_velocity
        else:
            contact_vel = self.last_rfoot_pos - rfoot_pos + self.gravity_velocity

        root_vel = vel[0] / (fps / vel_scale)
        weight = self._prob_to_weight(contact.max())
        velocity = art.math.lerp(root_vel, contact_vel, weight)

        current_foot_y = self.current_root_y + min(lfoot_pos[1].item(), rfoot_pos[1].item())
        if current_foot_y + velocity[1].item() <= self.floor_y:
            velocity[1] = self.floor_y - current_foot_y

        self.current_root_y += velocity[1].item()
        self.last_lfoot_pos, self.last_rfoot_pos = lfoot_pos, rfoot_pos
        self.imu = imu
        self.last_root_pos += velocity
        return pose, joints, self.last_root_pos.clone(), contact

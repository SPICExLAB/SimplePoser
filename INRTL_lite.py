"""
INRTL_Lite_50Hz_TCN: Lightweight 50Hz CNN-TCN model.

Architecture:
  Input: IMU (acc + gyro) + quaternion + differentials → 20 channels
  → CNN (Conv1d 20→24→32 with BN+ReLU)
  → TCN (3 TemporalBlocks, width 32, dilations 1/2/4)
  → Linear decoders for velocity and covariance
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, dilation=2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = dilation ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout,
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class INRTL_Lite_50Hz_TCN(nn.Module):
    """
    Input:  acc [B, T, 3], gyro [B, T, 3], rot_quat [B, T, 4]
    Output: net_vel [B, T, 3], cov [B, T, 3]
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.feature_cnn = nn.Sequential(
            nn.Conv1d(20, 24, kernel_size=5, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(24, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.tcn = TemporalConvNet(
            num_inputs=32,
            num_channels=[32, 32, 32],
            kernel_size=3,
            dropout=0.2,
            dilation=2,
        )

        self.veldecoder = nn.Linear(32, 3)
        self.velcov_decoder = nn.Linear(32, 3)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def normalize_quaternion(self, q):
        return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

    def quaternion_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack([x, y, z, w], dim=-1)

    def quaternion_conjugate(self, q):
        return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)

    def forward(self, data, rot=None, rot_quat=None):
        rotation_input = rot_quat if rot_quat is not None else rot
        assert rotation_input is not None, "Rotation input required"
        B, T, _ = rotation_input.shape

        acc = data["acc"]
        gyro = data["gyro"]
        ori_quat = self.normalize_quaternion(rotation_input)

        acc_diff = torch.zeros_like(acc)
        acc_diff[:, 1:, :] = acc[:, 1:, :] - acc[:, :-1, :]
        gyro_diff = torch.zeros_like(gyro)
        gyro_diff[:, 1:, :] = gyro[:, 1:, :] - gyro[:, :-1, :]

        quat_diff = torch.zeros_like(ori_quat)
        quat_diff[:, 0, :] = torch.tensor([0., 0., 0., 1.], device=ori_quat.device, dtype=ori_quat.dtype)
        if T > 1:
            quat_prev_inv = self.quaternion_conjugate(ori_quat[:, :-1, :])
            quat_relative = self.normalize_quaternion(
                self.quaternion_multiply(ori_quat[:, 1:, :], quat_prev_inv))
            quat_diff[:, 1:, :] = quat_relative

        combined = torch.cat([
            acc, gyro, ori_quat,
            acc_diff, gyro_diff, quat_diff
        ], dim=-1)  # [B, T, 20]

        features = self.feature_cnn(combined.transpose(1, 2))   # [B, 32, T]
        tcn_out = self.tcn(features).transpose(1, 2)            # [B, T, 32]

        net_vel = self.veldecoder(tcn_out)
        cov = None
        if hasattr(self.conf, 'propcov') and self.conf.propcov:
            cov = torch.exp(self.velcov_decoder(tcn_out) - 5.0)

        return {"cov": cov, 'net_vel': net_vel}

    def get_label(self, gt_label):
        return gt_label[:, :-1, :] if gt_label.shape[1] > 1 else gt_label

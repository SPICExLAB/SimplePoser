"""
TCN_Base: MobilePoser with TCN heads (drop-in LSTM replacement).

Target size: <100 KB int8 (~88K params).
Head config:
  Joints/Poser   H=32, 3 blocks, bidirectional (RF = 29 frames ~ 0.97s @ 30Hz)
  FootContact    H=16, 3 blocks, bidirectional
  Velocity       H=32, 4 blocks, causal        (RF = 61 frames ~ 2.03s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from config import joint_set
from mobileposer import MobilePoser


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """Residual TCN block (weight-norm Conv1d x2). Causal uses chomp; non-causal uses symmetric padding."""
    def __init__(self, channels, kernel_size, dilation, dropout=0.2, causal=True):
        super().__init__()
        pad = (kernel_size - 1) * dilation

        if causal:
            self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                               padding=pad, dilation=dilation))
            self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                               padding=pad, dilation=dilation))
            self.chomp1 = Chomp1d(pad)
            self.chomp2 = Chomp1d(pad)
        else:
            half = pad // 2
            self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                               padding=half, dilation=dilation))
            self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                               padding=half, dilation=dilation))
            self.chomp1 = nn.Identity()
            self.chomp2 = nn.Identity()

        self.net = nn.Sequential(
            self.conv1, self.chomp1, nn.ReLU(), nn.Dropout(dropout),
            self.conv2, self.chomp2, nn.ReLU(), nn.Dropout(dropout),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class TCN(nn.Module):
    """Linear -> stack of TemporalBlocks -> Linear. Drop-in for RNN.forward(x, h) -> (y, h)."""
    def __init__(self, n_input, n_output, n_hidden, n_blocks=3, kernel_size=3,
                 dropout=0.2, causal=False):
        super().__init__()
        self.linear_in = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TemporalBlock(n_hidden, kernel_size, dilation=2 ** i,
                          dropout=dropout, causal=causal)
            for i in range(n_blocks)
        ])
        self.linear_out = nn.Linear(n_hidden, n_output)

    def forward(self, x, h=None):
        x = self.dropout(F.relu(self.linear_in(x)))   # [B, T, H]
        x = x.transpose(1, 2)                         # [B, H, T]
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)                         # [B, T, H]
        return self.linear_out(x), None


class Joints(nn.Module):
    def __init__(self, hidden=32, blocks=3):
        super().__init__()
        self.rnn = TCN(joint_set.n_imu, joint_set.n_full * 3,
                       hidden, n_blocks=blocks, causal=False)

    def forward(self, x):
        joints, _ = self.rnn(x)
        return joints


class Poser(nn.Module):
    def __init__(self, hidden=32, blocks=3):
        super().__init__()
        self.rnn = TCN(joint_set.n_full * 3 + joint_set.n_imu,
                       joint_set.n_reduced * 6,
                       hidden, n_blocks=blocks, causal=False)

    def forward(self, x):
        pose, _ = self.rnn(x)
        return pose


class FootContact(nn.Module):
    def __init__(self, hidden=16, blocks=3):
        super().__init__()
        self.rnn = TCN(joint_set.n_full * 3 + joint_set.n_imu, 2,
                       hidden, n_blocks=blocks, causal=False)

    def forward(self, x):
        contact, _ = self.rnn(x)
        return contact


class Velocity(nn.Module):
    def __init__(self, hidden=32, blocks=4):
        super().__init__()
        self.rnn = TCN(joint_set.n_full * 3 + joint_set.n_imu,
                       joint_set.n_full * 3,
                       hidden, n_blocks=blocks, causal=True)
        self.rnn_state = None

    def forward(self, x):
        vel, _ = self.rnn(x)
        return vel

    def forward_online(self, x):
        vel, _ = self.rnn(x)
        return vel

    def reset(self):
        self.rnn_state = None


class TCN_Base(MobilePoser):
    """MobilePoser with TCN heads. Inherits the full inference / loss pipeline unchanged."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.joints = Joints(hidden=32, blocks=3)
        self.pose = Poser(hidden=32, blocks=3)
        self.foot_contact = FootContact(hidden=16, blocks=3)
        self.velocity = Velocity(hidden=32, blocks=4)

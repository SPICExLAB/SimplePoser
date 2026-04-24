import torch
import torch.nn as nn
import torch.nn.functional as F

from config import joint_set
from mobileposer import MobilePoser


class DSConv1d(nn.Module):
    """Depthwise-separable dilated 1D conv. Causal or symmetric padding."""
    def __init__(self, channels, kernel_size, dilation, causal=False):
        super().__init__()
        self.causal = causal
        self.pad_total = (kernel_size - 1) * dilation
        self.dw = nn.Conv1d(channels, channels, kernel_size,
                            dilation=dilation, groups=channels)
        self.pw = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        if self.causal:
            x = F.pad(x, (self.pad_total, 0))
        else:
            lp = self.pad_total // 2
            x = F.pad(x, (lp, self.pad_total - lp))
        return self.pw(self.dw(x))


class TCNBlock(nn.Module):
    """Residual block: DSConv -> LN -> ReLU -> Dropout -> DSConv -> LN -> ReLU -> Dropout."""
    def __init__(self, channels, kernel_size, dilation, dropout=0.1, causal=False):
        super().__init__()
        self.conv1 = DSConv1d(channels, kernel_size, dilation, causal)
        self.conv2 = DSConv1d(channels, kernel_size, dilation, causal)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _ln(x, norm):
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.drop(F.relu(self._ln(y, self.norm1)))
        y = self.conv2(y)
        y = self.drop(F.relu(self._ln(y, self.norm2)))
        return x + y


class TCN(nn.Module):
    """Drop-in analogue of mobileposer.RNN. forward(x) -> (y, None); x, y are (B, T, C)."""
    def __init__(self, n_input, n_output, n_hidden,
                 n_blocks=4, kernel_size=3, dropout=0.1,
                 bidirectional=True):
        super().__init__()
        causal = not bidirectional
        self.in_proj = nn.Linear(n_input, n_hidden)
        self.blocks = nn.ModuleList([
            TCNBlock(n_hidden, kernel_size, dilation=2 ** i,
                     dropout=dropout, causal=causal)
            for i in range(n_blocks)
        ])
        self.out_proj = nn.Linear(n_hidden, n_output)

    def forward(self, x, h=None):
        h = self.in_proj(x).transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h.transpose(1, 2)), None


class Joints(nn.Module):
    """IMU -> 24 joint positions."""
    def __init__(self, hidden=96, blocks=4):
        super().__init__()
        self.rnn = TCN(joint_set.n_imu, joint_set.n_full * 3, hidden, n_blocks=blocks)

    def forward(self, x):
        joints, _ = self.rnn(x)
        return joints


class Poser(nn.Module):
    """IMU + joints -> reduced SMPL pose (6D)."""
    def __init__(self, hidden=96, blocks=4):
        super().__init__()
        self.rnn = TCN(joint_set.n_full * 3 + joint_set.n_imu,
                       joint_set.n_reduced * 6, hidden, n_blocks=blocks)

    def forward(self, x):
        pose, _ = self.rnn(x)
        return pose


class FootContact(nn.Module):
    """IMU + joints -> [left, right] contact probability."""
    def __init__(self, hidden=32, blocks=3):
        super().__init__()
        self.rnn = TCN(joint_set.n_full * 3 + joint_set.n_imu, 2, hidden, n_blocks=blocks)

    def forward(self, x):
        contact, _ = self.rnn(x)
        return contact


class Velocity(nn.Module):
    """IMU + joints -> per-joint velocity (causal, mirrors the unidirectional LSTM)."""
    def __init__(self, hidden=96, blocks=4):
        super().__init__()
        self.rnn = TCN(joint_set.n_full * 3 + joint_set.n_imu,
                       joint_set.n_full * 3, hidden, n_blocks=blocks,
                       bidirectional=False)
        self.rnn_state = None

    def forward(self, x):
        vel, _ = self.rnn(x)
        return vel

    def forward_online(self, x):
        vel, _ = self.rnn(x)
        return vel

    def reset(self):
        self.rnn_state = None


class TCNPoser(MobilePoser):
    """MobilePoser with TCN heads. Outer pose / translation / contact pipeline is unchanged.

    Defaults target ~300 KB at int8 (~304K params). Override via cfg:
      hidden_main / blocks_main     -- main heads (pose, joints, velocity)
      hidden_contact / blocks_contact -- foot-contact head
    Presets (hidden_main / blocks_main):
      (96, 4)  ->  ~297 KB int8, RF = 61 frames  (default)
      (96, 6)  ->  ~413 KB int8, RF = 253 frames (longer context)
      (128, 4) ->  ~488 KB int8, RF = 61 frames
      (512, 4) ->  ~6.4 MB int8, capacity-matched to MobilePoser"""
    def __init__(self, cfg, hidden_main=None, blocks_main=None,
                 hidden_contact=None, blocks_contact=None):
        super().__init__(cfg)
        hidden_main = cfg.get('hidden_main', hidden_main) if hidden_main is None else hidden_main
        blocks_main = cfg.get('blocks_main', blocks_main) if blocks_main is None else blocks_main
        hidden_contact = cfg.get('hidden_contact', hidden_contact) if hidden_contact is None else hidden_contact
        blocks_contact = cfg.get('blocks_contact', blocks_contact) if blocks_contact is None else blocks_contact
        hidden_main = hidden_main or 96
        blocks_main = blocks_main or 4
        hidden_contact = hidden_contact or 32
        blocks_contact = blocks_contact or 3
        self.joints = Joints(hidden_main, blocks_main)
        self.pose = Poser(hidden_main, blocks_main)
        self.foot_contact = FootContact(hidden_contact, blocks_contact)
        self.velocity = Velocity(hidden_main, blocks_main)

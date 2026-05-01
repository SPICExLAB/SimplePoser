"""
Build the TCN heads, export each to ONNX, quantize weights to int8, and
measure the on-disk size of the resulting quantized model files.

Runs standalone (doesn't import MobilePoser) so we can measure the TCN-only
cost — which is what's relevant for the 100 KB budget.
"""
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import parametrize
from onnxruntime.quantization import quantize_dynamic, QuantType


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.2, causal=True):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        if causal:
            self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation))
            self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation))
            self.chomp1 = Chomp1d(pad); self.chomp2 = Chomp1d(pad)
        else:
            half = pad // 2
            self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=half, dilation=dilation))
            self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=half, dilation=dilation))
            self.chomp1 = nn.Identity(); self.chomp2 = nn.Identity()
        self.net = nn.Sequential(self.conv1, self.chomp1, nn.ReLU(), nn.Dropout(dropout),
                                 self.conv2, self.chomp2, nn.ReLU(), nn.Dropout(dropout))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class TCN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_blocks=3, kernel_size=3, dropout=0.2, causal=False):
        super().__init__()
        self.linear_in = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TemporalBlock(n_hidden, kernel_size, dilation=2**i, dropout=dropout, causal=causal)
            for i in range(n_blocks)
        ])
        self.linear_out = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_in(x)))
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        return self.linear_out(x)


# ---- Full bundle module for a single ONNX export ----------------------------

class TCN_Base_Bundle(nn.Module):
    """All 4 heads wrapped so we can export/quantize the full param set together."""
    def __init__(self):
        super().__init__()
        N_IMU, N_FULL3, N_RED6 = 60, 72, 96
        AUX = N_FULL3 + N_IMU  # 132
        self.joints      = TCN(N_IMU, N_FULL3, 32, n_blocks=3, causal=False)
        self.pose        = TCN(AUX,   N_RED6,  32, n_blocks=3, causal=False)
        self.foot_contact= TCN(AUX,   2,       16, n_blocks=3, causal=False)
        self.velocity    = TCN(AUX,   N_FULL3, 32, n_blocks=4, causal=True)

    def forward(self, imu, aux):
        j = self.joints(imu)
        p = self.pose(aux)
        c = self.foot_contact(aux)
        v = self.velocity(aux)
        return j, p, c, v


def strip_weight_norm(model):
    """Fuse weight_norm (weight_g, weight_v) into a single weight parameter."""
    for module in model.modules():
        if isinstance(module, nn.Conv1d) and parametrize.is_parametrized(module, 'weight'):
            parametrize.remove_parametrizations(module, 'weight', leave_parametrized=True)
    return model


def main():
    model = TCN_Base_Bundle().eval()
    strip_weight_norm(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Fused params: {total_params:,}')

    out_dir = os.path.abspath('./_tcn_base_out')
    os.makedirs(out_dir, exist_ok=True)

    fp32_onnx = os.path.join(out_dir, 'tcn_base_fp32.onnx')
    int8_onnx = os.path.join(out_dir, 'tcn_base_int8.onnx')

    # Representative sequence length for export (dynamic axis keeps it flexible).
    T = 60
    dummy_imu = torch.randn(1, T, 60)
    dummy_aux = torch.randn(1, T, 132)

    torch.onnx.export(
        model,
        (dummy_imu, dummy_aux),
        fp32_onnx,
        input_names=['imu', 'aux'],
        output_names=['joints', 'pose', 'contact', 'velocity'],
        dynamic_axes={'imu': {1: 'T'}, 'aux': {1: 'T'},
                      'joints': {1: 'T'}, 'pose': {1: 'T'},
                      'contact': {1: 'T'}, 'velocity': {1: 'T'}},
        opset_version=17,
    )

    # Dynamic INT8 quantization: weights -> int8, activations -> fp32 at runtime.
    quantize_dynamic(
        model_input=fp32_onnx,
        model_output=int8_onnx,
        weight_type=QuantType.QInt8,
    )

    fp32_size = os.path.getsize(fp32_onnx)
    int8_size = os.path.getsize(int8_onnx)

    print(f'\nfp32 ONNX : {fp32_onnx}')
    print(f'           {fp32_size:>10,} bytes ({fp32_size/1024:.2f} KB)')
    print(f'int8 ONNX : {int8_onnx}')
    print(f'           {int8_size:>10,} bytes ({int8_size/1024:.2f} KB)')
    print(f'\nCompression ratio: {fp32_size/int8_size:.2f}x')

    # Verify dtypes inside the int8 file
    import onnx
    m = onnx.load(int8_onnx)
    dtype_counts = {}
    for init in m.graph.initializer:
        dt = onnx.TensorProto.DataType.Name(init.data_type)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
    print(f'\nInt8 ONNX tensor dtypes: {dtype_counts}')

    # Count int8 weight bytes vs metadata bytes
    import numpy as np
    int8_bytes = 0
    fp32_bytes = 0
    for init in m.graph.initializer:
        n = int(np.prod(init.dims)) if init.dims else 1
        dt = init.data_type
        if dt == onnx.TensorProto.INT8:
            int8_bytes += n
        elif dt == onnx.TensorProto.FLOAT:
            fp32_bytes += n * 4
        elif dt == onnx.TensorProto.INT32:
            fp32_bytes += n * 4
    print(f'Int8 weight bytes: {int8_bytes/1024:.2f} KB')
    print(f'FP32/INT32 tensor bytes (scales, zp, biases): {fp32_bytes/1024:.2f} KB')
    print(f'Graph overhead (rest of file): {(int8_size - int8_bytes - fp32_bytes)/1024:.2f} KB')


if __name__ == '__main__':
    main()

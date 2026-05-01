"""
Produce a raw int8 dump of the TCN_Base weights (minimum achievable deployment size):
per-channel symmetric quantization, int8 weights + fp32 scales + fp32 biases
packed into a single flat binary file with a small header.

This approximates what a hand-rolled MCU/embedded deployment format would produce
(no graph overhead, no tensor metadata, just the numbers).
"""
import os
import struct
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import parametrize

import sys
sys.path.insert(0, os.path.dirname(__file__))
from _quantize_tcn_base import TCN_Base_Bundle, strip_weight_norm


def quantize_per_channel_symmetric(weight: torch.Tensor, axis: int = 0):
    """Symmetric per-channel int8: scale = max(|w|)/127, q = round(w/scale)."""
    # move quantization axis to dim 0
    w = weight.transpose(0, axis).contiguous()
    flat = w.flatten(1)                       # [out_ch, rest]
    max_abs = flat.abs().max(dim=1).values.clamp(min=1e-12)
    scale = (max_abs / 127.0).to(torch.float32)
    q = torch.round(w / scale.view(-1, *([1] * (w.dim() - 1)))).clamp(-128, 127).to(torch.int8)
    q = q.transpose(0, axis).contiguous()
    return q, scale


def main():
    model = TCN_Base_Bundle().eval()
    strip_weight_norm(model)

    out_path = os.path.abspath('./_tcn_base_out/tcn_base_int8_raw.bin')

    named_weights = []
    for name, p in model.named_parameters():
        named_weights.append((name, p.detach()))

    with open(out_path, 'wb') as f:
        # header: uint32 num_tensors
        f.write(struct.pack('<I', len(named_weights)))
        total_int8 = 0
        total_fp32 = 0
        for name, t in named_weights:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<H', len(name_bytes)))
            f.write(name_bytes)
            # shape: num_dims (u8) + dims (u32 each)
            f.write(struct.pack('<B', t.dim()))
            for d in t.shape:
                f.write(struct.pack('<I', d))

            if t.dim() >= 2:
                # Quantize weight: per-channel symmetric along dim 0 (out_channels)
                q, scale = quantize_per_channel_symmetric(t, axis=0)
                f.write(b'Q')                                # 'Q' = quantized
                f.write(scale.numpy().tobytes())             # fp32 per-channel scales
                f.write(q.numpy().tobytes())                 # int8 weights
                total_fp32 += scale.numel() * 4
                total_int8 += q.numel()
            else:
                # biases and weight_g (if any left): keep as fp32
                f.write(b'F')
                f.write(t.to(torch.float32).numpy().tobytes())
                total_fp32 += t.numel() * 4

    file_size = os.path.getsize(out_path)
    print(f'Raw int8 binary : {out_path}')
    print(f'                  {file_size:>10,} bytes ({file_size/1024:.2f} KB)')
    print(f'  int8 weight bytes: {total_int8/1024:.2f} KB')
    print(f'  fp32 scales+biases: {total_fp32/1024:.2f} KB')
    print(f'  header/metadata:   {(file_size - total_int8 - total_fp32)/1024:.2f} KB')


if __name__ == '__main__':
    main()

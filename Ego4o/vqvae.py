"""
Part-aware VQ-VAE for HumanML3D 263-dim motion representation.

Adapted from ego4o-code-release VQVAE_limb_hml.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encdec import Encoder, Decoder
from .quantize import QuantizeEMAReset


class VQLimbHML(nn.Module):
    """
    Part-aware VQ-VAE that decomposes 263-dim HumanML3D motion into 6 body parts,
    each with its own encoder and codebook, and a shared decoder.

    Input:  (B, 263, 1, T) or (B, 263, T)
    Output: (B, 263, 1, T), commit_loss, perplexity
    """

    def __init__(self, nb_code=512, code_dim=512, output_emb_width=512,
                 down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None, mu=0.99):
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code

        # Body-part segmentation indices into 263-dim HumanML3D vector
        def values_term(i):
            i -= 1
            return ([4 + i * 3, 4 + i * 3 + 1, 4 + i * 3 + 2] +
                    [4 + 63 + i * 6 + k for k in range(6)] +
                    [4 + 63 + 126 + (i + 1) * 3 + k for k in range(3)])

        self.partSeg = [
            [0, 1, 2, 3, 4 + 63 + 126, 4 + 63 + 126 + 1, 4 + 63 + 126 + 2],  # root (7)
            [x for i in [3, 6, 9, 12, 15] for x in values_term(i)],             # head (60)
            [x for i in [13, 16, 18, 20] for x in values_term(i)],              # left arm (48)
            [x for i in [14, 17, 19, 21] for x in values_term(i)],              # right arm (48)
            [x for i in [1, 4, 7, 10] for x in values_term(i)] + [259, 260],    # left leg (50)
            [x for i in [2, 5, 8, 11] for x in values_term(i)] + [261, 262],    # right leg (50)
        ]

        # Per-part encoders
        self.limb_encoders = nn.ModuleList([
            Encoder(len(part), output_emb_width, down_t, stride_t, width, depth,
                    dilation_growth_rate, activation=activation, norm=norm)
            for part in self.partSeg
        ])

        # Per-part quantizers
        self.quantizers = nn.ModuleList([
            QuantizeEMAReset(nb_code, code_dim, mu=mu)
            for _ in self.partSeg
        ])

        # Shared decoder: takes concatenated codes from all 6 parts
        self.decoder = Decoder(263, output_emb_width * len(self.partSeg), down_t, stride_t,
                               width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def forward(self, x):
        """
        Args:
            x: (B, 263, 1, T) motion input
        Returns:
            x_out: (B, 263, 1, T) reconstruction
            commit_loss: scalar
            perplexity: scalar
        """
        has_extra_dim = (x.dim() == 4)
        if has_extra_dim:
            x = x.squeeze(2)  # (B, 263, T)

        bs, F, T = x.shape
        x_in = x.permute(0, 2, 1)  # (B, T, 263)

        # Encode each body part
        x_s = []
        for i, part in enumerate(self.partSeg):
            x_current = x_in[:, :, part].reshape(bs, T, -1)  # (B, T, part_dim)
            x_current = x_current.permute(0, 2, 1).float()   # (B, part_dim, T)
            x_feature = self.limb_encoders[i](x_current)

            # L2 normalize
            x_feature = x_feature / torch.norm(x_feature, dim=[1, 2]).unsqueeze(1).unsqueeze(1)
            x_s.append(x_feature)

        # Quantize each body part
        x_quantized = []
        loss = 0.0
        perplexity = 0.0
        for i in range(len(self.partSeg)):
            xq, l, p = self.quantizers[i](x_s[i])
            x_quantized.append(xq)
            loss += l
            perplexity += p

        x_quantized = torch.cat(x_quantized, dim=1)  # (B, 6*emb_width, T')

        # Decode
        x_decoder = self.decoder(x_quantized)  # (B, 263, T)

        if has_extra_dim:
            x_decoder = x_decoder.unsqueeze(2)  # (B, 263, 1, T)

        return x_decoder, loss, perplexity

    def get_quantized_codes(self, x, in_idx_format=True):
        """Extract codebook indices or quantized features."""
        has_extra_dim = (x.dim() == 4)
        if has_extra_dim:
            x = x.squeeze(2)

        bs, F, T = x.shape
        x_in = x.permute(0, 2, 1)

        x_s = []
        for i, part in enumerate(self.partSeg):
            x_current = x_in[:, :, part].reshape(bs, T, -1)
            x_current = x_current.permute(0, 2, 1).float()
            x_feature = self.limb_encoders[i](x_current)
            x_feature = x_feature / torch.norm(x_feature, dim=[1, 2]).unsqueeze(1).unsqueeze(1)
            x_s.append(x_feature)

        if not in_idx_format:
            x_quantized = []
            for i in range(len(self.partSeg)):
                xq, _, _ = self.quantizers[i](x_s[i])
                x_quantized.append(xq)
            return x_quantized
        else:
            x_ids = []
            for i in range(len(self.partSeg)):
                x_ids.append(self.quantizers[i].get_code_idx(x_s[i]).unsqueeze(-1))
            x_ids = torch.cat(x_ids, dim=-1)  # (B, T', 6)
            return x_ids

    def forward_decoder(self, x_id):
        """Decode from codebook indices. x_id: (B, T', 6)"""
        x_quantized = []
        for i in range(len(self.partSeg)):
            x_current = self.quantizers[i].forward_from_code_idx(x_id[:, :, i])
            x_quantized.append(x_current)
        x_quantized = torch.cat(x_quantized, dim=1)
        x_decoder = self.decoder(x_quantized)
        return x_decoder.permute(0, 2, 1)  # (B, T, 263)

    def get_code_idx(self, x):
        """Get code indices for Stage 2 training targets. Returns (B, 6, T')."""
        x_ids = self.get_quantized_codes(x, in_idx_format=True)  # (B, T', 6)
        return x_ids.permute(0, 2, 1)  # (B, 6, T')

    def get_x_quantized_from_x_ids(self, x_id):
        """
        Get quantized vectors from code IDs or one-hot vectors (Gumbel-softmax).

        Args:
            x_id: (..., 6) where last dim indexes body parts.
                  Can be integer indices or one-hot vectors.
        Returns:
            List of 6 tensors, each (B, code_dim, T')
        """
        x_quantized = []
        for i in range(len(self.partSeg)):
            x_current = self.quantizers[i].forward_from_code_idx(x_id[..., i])
            x_quantized.append(x_current)
        return x_quantized

    def forward_decoder_from_quantized_codes(self, x_quantized):
        """Decode from list of per-part quantized codes. Returns (B, 263, 1, T)."""
        x_quantized = torch.cat(x_quantized, dim=1)
        x_decoder = self.decoder(x_quantized)
        return x_decoder.unsqueeze(2)

"""
IMU Transformer Encoder for Stage 2: maps IMU sensor data to VQ-VAE code predictions.

Adapted from ego4o-code-release TransformerAutoencoder_withCodes_hml_G2_noTraj,
with CLIP/text/image components removed for IMU-only operation.

Input:  [B, T, sensor_num, input_dim]  (e.g. [B, 128, 6, 9])
Output: [B, sensor_num, Tt, nb_code]   code logits per body part
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)


class IMUTransformerEncoder(nn.Module):
    """
    Transformer encoder: IMU → VQ-VAE code logits.

    Architecture (matching ego4o):
        1. Temporal tokenization: group every codes_realLength frames
        2. Linear projection to model_dim1
        3. Positional encoding
        4. Main transformer encoder (num_encoder_layers)
        5. Code prediction head: linear down + transformer + project to nb_code
    """

    def __init__(self, input_dim=9, sensor_num=6, nb_code=512,
                 model_dim1=512, model_dim2=256, codes_realLength=4,
                 nhead=4, num_encoder_layers=4, num_decoder_layers=3,
                 dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.sensor_num = sensor_num
        self.nb_code = nb_code
        self.codes_realLength = codes_realLength
        self.model_dim1 = model_dim1
        self.model_dim2 = model_dim2

        # Input projection: codes_realLength * input_dim → model_dim1
        self.linear_in = nn.Linear(codes_realLength * input_dim, model_dim1)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim1, dropout)

        # Main transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim1, nhead=nhead,
            dim_feedforward=model_dim1 * 4, dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Code prediction head
        self.linear_mid_codeIdx = nn.Linear(model_dim1, model_dim2)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim2, nhead=nhead,
            dim_feedforward=model_dim2 * 4, dropout=dropout,
            batch_first=False
        )
        self.transformer_codeIdx = nn.TransformerEncoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.project_codeIdx = nn.Linear(model_dim2, nb_code)

    def forward(self, x):
        """
        Args:
            x: [B, T, sensor_num, input_dim]  e.g. [B, 128, 6, 9]
        Returns:
            pre_codes: [B, sensor_num, Tt, nb_code]  e.g. [B, 6, 32, 512]
        """
        bs, T, NJoints, input_dim = x.shape
        Tt = T // self.codes_realLength  # temporal tokens

        # Temporal tokenization: interleave sensor × time tokens
        x = x.permute(2, 1, 0, 3)                          # [NJ, T, B, dim]
        x = x.reshape(NJoints, Tt, self.codes_realLength, bs, input_dim)
        x = x.permute(1, 0, 3, 2, 4)                       # [Tt, NJ, B, crl, dim]
        x = x.reshape(Tt * NJoints, bs, self.codes_realLength * input_dim)
        # → [Tt*NJ, B, crl*dim]

        # Project + positional encoding
        x = self.linear_in(x)       # [Tt*NJ, B, model_dim1]
        x = self.pos_encoder(x)

        # Main transformer encoder
        result = self.transformer_encoder(x)  # [Tt*NJ, B, model_dim1]

        # Code prediction head
        pre_codes = self.transformer_codeIdx(self.linear_mid_codeIdx(result))
        pre_codes = self.project_codeIdx(pre_codes)  # [Tt*NJ, B, nb_code]

        # Reshape to [B, sensor_num, Tt, nb_code]
        pre_codes = pre_codes.reshape(Tt, NJoints, bs, -1)
        pre_codes = pre_codes.permute(2, 1, 0, 3)  # [B, NJ, Tt, nb_code]

        return pre_codes

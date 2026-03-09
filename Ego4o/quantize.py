import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu=0.99):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # EMA update
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        # Reset unused codes with random vectors
        self.codebook = usage * code_update + (1 - usage) * code_rand

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        k_w = self.codebook.t()
        distance = (torch.sum(x ** 2, dim=-1, keepdim=True)
                    - 2 * torch.matmul(x, k_w)
                    + torch.sum(k_w ** 2, dim=0, keepdim=True))
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        return F.embedding(code_idx, self.codebook)

    def dequantize_from_one_hot(self, code_onehot):
        """Decode from one-hot vectors (e.g. from Gumbel-softmax)."""
        return torch.matmul(code_onehot, self.codebook)

    def forward(self, x):
        N, width, T = x.shape
        x = self.preprocess(x)

        if self.training and not self.init:
            self.init_codebook(x)

        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        commit_loss = F.mse_loss(x, x_d.detach())

        # Straight-through estimator
        x_d = x + (x_d - x).detach()

        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        return x_d, commit_loss, perplexity

    def get_code_idx(self, x):
        N, width, T = x.shape
        x = self.preprocess(x)
        if self.training and not self.init:
            self.init_codebook(x)
        code_idx = self.quantize(x).view(N, T).contiguous()
        return code_idx

    def forward_from_code_idx(self, x_idx):
        if len(x_idx.shape) == 3:
            N, T, F = x_idx.shape
            code_idx = x_idx.view(N * T, F)
            x_d = torch.matmul(code_idx, self.codebook)
            x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        else:
            N, T = x_idx.shape
            code_idx = x_idx.view(N * T)
            x_d = self.dequantize(code_idx)
            x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        return x_d

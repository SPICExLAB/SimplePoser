import torch
import torch.nn as nn
import torch.nn.functional as F

from articulate.math.general import normalize_tensor


def activation_layer(act_name) -> nn.Module:
    """
    Get activation layer by name.

    Args:
        act_name: Name of activation.
    Return:
        act_layer: Activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_name.lower() == 'leakyrelu':
            act_layer = nn.LeakyReLU(negative_slope=1e-2)
        elif act_name.lower() == 'elu':
            act_layer = nn.ELU()
        elif act_name.lower() == 'tanh':
            act_layer = nn.Tanh()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


def linear_interpolation_batch(vector1, vector2, target_length):
    if vector1.size(-1) != vector2.size(-1):
        raise ValueError("Vector dimension mismatch.")

    interpolation_steps = target_length - 1
    step_size = (vector2 - vector1) / interpolation_steps

    interpolated_data = [vector1]

    for i in range(1, interpolation_steps):
        interpolated_point = vector1 + i * step_size
        interpolated_data.append(interpolated_point)

    interpolated_data.append(vector2)
    interpolated_data = torch.stack(interpolated_data, dim=1)

    return interpolated_data    


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss    


class DNN(nn.Module):
    """

    Args:
        n_input: dim of input
        n_hiddens: dim of hidden layers. e.g. [64, 128, 64]
        n_output: dim of output
        act_func: 'relu' | 'tanh' | 'LeakyReLu' | 'sigmoid'
        dropout: dropout rate, default:None

    Returns:
        nn.Module, a DNN module
    """
    def __init__(self, n_input, n_hiddens, n_output, act_func='relu', dropout=None):
        super(DNN, self).__init__()
        
        channel_list = [n_input] + n_hiddens 
        layers = []

        for i in range(len(channel_list) - 1):
            mini_layer = [nn.Linear(in_features=channel_list[i], out_features=channel_list[i+1]), activation_layer(act_func)]
            if dropout is not None:
                mini_layer += [nn.Dropout(dropout)]
            layers += mini_layer

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features=channel_list[-1], out_features=n_output)

    
    def forward(self, x):
        return self.output_layer(self.network(x))


class DiscrepancyNomalizer(nn.Module):
    def __init__(self, n_feature):
        super(DiscrepancyNomalizer, self).__init__()
        self.mmd = MMDLoss()
        self.BN = nn.BatchNorm1d(num_features=n_feature)
    
    def forward(self, x1, x2):
        dis = x2 - x1
        mean = torch.mean(dis, dim=0)
        dis = self.BN(dis)
        norm_noise = torch.randn(size=dis.shape).float().to(dis.device)
        return self.mmd(dis, norm_noise) + torch.pow(mean, 2).mean()

    def get_std(self):
        return torch.sqrt(self.BN.running_var.data.float().squeeze(0))

    def get_mean(self):
        return self.BN.running_mean.data.float().squeeze(0)        

    def semo_recon(self, x1, eta=1, mask=None):
        # x1 shape: [batch, seq_len, dim]
        if mask is not None:
            std = torch.sqrt(self.BN.running_var.data).float().squeeze(0).to(x1.device)
            mask = mask.to(x1.device)
            std = std * mask
            return x1 + std

        # BN statistics
        std = torch.sqrt(self.BN.running_var.data).float().squeeze(0).to(x1.device)

        # ------------------------------------------------------------
        # Frame-wise jitter (zero-mean random noise)
        # ------------------------------------------------------------
        norm_noise = torch.randn(size=x1.shape).float().to(x1.device)
        scale = 0.05  # global magnitude scale tuned to match real device sliding

        recon_semo = norm_noise * std * scale

        # ------------------------------------------------------------
        # Temporal Coherence Scheme (TCS)
        # Smooth drift from one random bias to another
        # ------------------------------------------------------------
        bias_shift_1 = torch.randn(size=x1[:, 0, :].shape).float().to(x1.device)  # [B, D]
        bias_shift_2 = torch.randn(size=x1[:, 0, :].shape).float().to(x1.device)  # [B, D]
        bias_shift = linear_interpolation_batch(bias_shift_1, bias_shift_2, target_length=x1.shape[1])  # [B, T, D]

        recon_semo = recon_semo + bias_shift * std * eta * scale

        return x1 + recon_semo



class SemoAE(nn.Module):
    def __init__(self, feat_dim, encode_dim):
        super(SemoAE, self).__init__()

        self.feat_dim = feat_dim
        self.encode_dim = encode_dim
        
        act_func = 'tanh'
        self.encoder = DNN(n_input=feat_dim, n_hiddens=[128, 64], n_output=encode_dim, act_func=act_func)
        self.decoder = DNN(n_input=encode_dim, n_hiddens=[64, 128], n_output=feat_dim, act_func=act_func)
        self.dis_normalizer = DiscrepancyNomalizer(n_feature=encode_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, normalize_r6d=True):
        x = self.decoder(z)
        if normalize_r6d:
            x = self._normalize_r6d(x)
        return x

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def _r6d_norm(self, x, rot_num=5):
        """Gram-Schmidt orthonormalization for R6D rotations."""
        x_shape = x.shape
        x = x.reshape(x_shape[0], x_shape[1], rot_num, 6)
        result = []
        for i in range(rot_num):
            column0 = normalize_tensor(x[:, :, i, 0:3])
            column1 = normalize_tensor(x[:, :, i, 3:6] - (column0 * x[:, :, i, 3:6]).sum(dim=-1, keepdim=True) * column0)
            result.append(torch.cat([column0, column1], dim=-1))
        x = torch.cat(result, dim=-1)
        return x
    
    @torch.no_grad()
    def add_secondary_motion(self, x, eta=1):
        """
        Add secondary motion to IMU sequence for data augmentation.
        
        Args: 
            x: [B, T, dim] clean IMU sequence
            eta: noise scale (higher = more noise)
        
        Returns: 
            [B, T, dim] augmented IMU sequence
        """
        self.eval()
        B, T, D = x.shape

        # encode and add secondary motion
        x = self.encoder(x.view(-1, D)).view(B, T, -1)
        x = self.dis_normalizer.semo_recon(x, eta=eta)
        x = self.decoder(x)
        acc, rot = x[:, :, :45][:, :, :-30], x[:, :, -30:]
        rot = self._r6d_norm(rot)

        x = torch.cat([acc, rot], dim=-1)
        return x
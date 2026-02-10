import yaml
import torch
import numpy as np


def load_yaml(file_path: str):
    """Load YAML file.
    Args:
        file_path: Path to the YAML file.
    Returns:
        cfg: Dictionary containing the YAML file contents.
    """
    with open(file_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_vposer(vposer_ckpt: str, device: torch.device):
    """
    Load VPoser model from checkpoint directory.

    Args:
        vposer_ckpt: Path to VPoser checkpoint directory
        device: torch device

    Returns:
        vposer: VPoser model in eval mode with frozen gradients
    """
    from pathlib import Path
    from human_body_prior.train.vposer_smpl import VPoser

    ckpt_path = Path(vposer_ckpt) / 'snapshots' / 'TR00_E096.pt'
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    vposer = VPoser(num_neurons=512, latentD=32, data_shape=[1, 21, 3])
    vposer.load_state_dict(state_dict)
    vposer.to(device).eval()
    vposer.requires_grad_(False)
    return vposer


def rotation_matrix_to_axis_angle_differentiable(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Differentiable rotation matrix to axis-angle using Rodrigues formula.
    Args:
        rotmat: [..., 3, 3] rotation matrices
    Returns:
        [..., 3] axis-angle vectors
    """
    shape = rotmat.shape[:-2]
    R = rotmat.reshape(-1, 3, 3)

    # angle from trace
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))

    # axis from skew-symmetric part of R
    skew = torch.stack([R[:, 2, 1] - R[:, 1, 2],
                        R[:, 0, 2] - R[:, 2, 0],
                        R[:, 1, 0] - R[:, 0, 1]], dim=-1)

    # scale: axis_angle = skew * angle / (2 * sin(angle)), with small-angle fallback
    sin_a = torch.sin(angle).unsqueeze(-1)
    axis_angle = torch.where(sin_a.abs() > 1e-7,
                             skew * angle.unsqueeze(-1) / (2 * sin_a),
                             skew / 2)

    return axis_angle.reshape(*shape, 3)


def r6d_to_axis_angle(r6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to axis-angle.

    Args:
        r6d: [*, 6] 6D rotation vectors

    Returns:
        Axis-angle tensor of shape [*, 3]
    """
    from articulate.math.angular import r6d_to_rotation_matrix, rotation_matrix_to_axis_angle

    batch_shape = r6d.shape[:-1]
    rotmat = r6d_to_rotation_matrix(r6d.reshape(-1, 6))
    aa = rotation_matrix_to_axis_angle(rotmat)
    return aa.reshape(*batch_shape, 3)
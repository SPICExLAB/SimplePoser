import torch
import yaml
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


def normalize_imu(acc: torch.Tensor, ori: torch.Tensor):
    """
    Normalize IMU data to root IMU. Root IMU reamins unchanged.  

    Args:
        acc: [B, T, 3] Acceleration data
        ori: [B, T, 3, 3] Orientation data
    Returns:
        normalized_acc: [B, T, 3] Normalized acceleration data
        normalized_ori: [B, T, 3, 3] Normalized orientation data
    """
    root_acc, root_ori = acc[:, 3:4], ori[:, 3:4]
    acc = (acc - root_acc).bmm(root_ori.squeeze(1))
    ori = root_ori.transpose(2, 3).matmul(ori)
    acc[:, 3], ori[:, 3] = root_acc.squeeze(1), root_ori.squeeze(1)
    return acc, ori
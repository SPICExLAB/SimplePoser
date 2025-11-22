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
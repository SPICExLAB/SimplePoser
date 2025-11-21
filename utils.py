import yaml


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
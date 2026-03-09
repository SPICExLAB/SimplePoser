"""
VQ-VAE dataset: loads GT motion .pt files and converts to HumanML3D 263-dim.

The VQ-VAE trains on ground-truth motion only (no IMU data needed).
Pipeline per sequence:
  1. Load pose (T, 24, 3, 3) and tran (T, 3) from .pt files
  2. FK → joint positions (T, 24, 3), add translation for world-space
  3. Take first 22 joints (drop l_hand, r_hand)
  4. Downsample to target FPS (20 Hz for HumanML3D)
  5. process_file() → (T-1, 263) HumanML3D representation
  6. Window into fixed-length segments
  7. Return as (263, 1, T) tensors for VQ-VAE input
"""

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

import articulate as art
from config import paths, datasets
from Ego4o.humanml import process_file, compute_target_offsets


class VQVAEDataset(Dataset):
    def __init__(self, cfg, fold='train'):
        super().__init__()
        self.cfg = cfg
        self.fold = fold
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)
        self.tgt_offsets = compute_target_offsets()

        self.windows = []
        self._load_data()

    def _load_data(self):
        data_folder = Path(paths.data_dir) / self.cfg['dataset']

        if self.cfg.get('data_file'):
            data_files = [self.cfg['data_file']]
        elif (data_folder / 'train.pt').exists():
            data_files = ['test.pt'] if self.fold == 'test' else ['train.pt']
        else:
            data_files = [x.name for x in data_folder.iterdir() if not x.is_dir()]

        print(f"[VQ-VAE] Loading {data_files} from {data_folder}")

        for data_file in tqdm(data_files, desc="Loading files"):
            file_data = torch.load(
                data_folder / data_file,
                map_location='cpu',
                weights_only=True
            )
            dataset_name = Path(data_file).stem
            source_fps = datasets.get(dataset_name, {}).get('fps', self.cfg['target_fps'])
            step = max(1, round(source_fps / self.cfg['target_fps']))
            print(f"  Dataset: {dataset_name} | Source FPS: {source_fps} | "
                  f"Target FPS: {self.cfg['target_fps']} | Step: {step}")
            self._process_file(file_data, step)

    def _process_file(self, file_data, step):
        poses = file_data['pose']
        trans = file_data['tran']

        pbar = tqdm(
            enumerate(zip(poses, trans)),
            total=len(poses),
            desc=f"Converting to HumanML3D",
            leave=False
        )

        window = self.cfg['window_length']
        feet_thre = self.cfg.get('feet_thre', 0.002)
        n_success = 0
        n_fail = 0

        for idx, (pose, tran) in pbar:
            try:
                data_263 = self._pose_to_humanml(pose, tran, step, feet_thre)
                if data_263 is None or len(data_263) < window:
                    n_fail += 1
                    continue

                # Window into fixed-length segments
                data_tensor = torch.from_numpy(data_263).float()  # (T, 263)
                chunks = torch.split(data_tensor, window)
                for chunk in chunks:
                    if len(chunk) == window:
                        self.windows.append(chunk)
                n_success += 1
            except Exception:
                n_fail += 1
                continue

            pbar.set_postfix({'ok': n_success, 'fail': n_fail, 'windows': len(self.windows)})

        print(f"  Sequences: {n_success} success, {n_fail} failed | Windows: {len(self.windows)}")

    def _pose_to_humanml(self, pose, tran, step, feet_thre):
        """Convert SMPL pose/tran to HumanML3D 263-dim representation."""
        pose = pose.view(-1, 24, 3, 3)
        tran = tran.view(-1, 3)

        # FK to get joint positions
        _, joints = self.bodymodel.forward_kinematics(pose=pose)
        joints = joints.view(-1, 24, 3)

        # World-space positions
        positions = joints + tran.unsqueeze(1)  # (T, 24, 3)

        # Take first 22 joints (drop l_hand=22, r_hand=23)
        positions = positions[:, :22]  # (T, 22, 3)

        # Downsample
        positions = positions[::step]

        # Need at least a few frames
        if len(positions) < 4:
            return None

        # Convert to numpy for process_file
        positions_np = positions.numpy()

        # process_file returns (T-1, 263)
        data, _, _, _ = process_file(positions_np, feet_thre, self.tgt_offsets)

        return data

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # (T, 263) → (263, 1, T) for VQ-VAE input
        x = self.windows[idx]  # (T, 263)
        x = x.permute(1, 0).unsqueeze(1)  # (263, 1, T)
        return x


def get_dataloaders(cfg, device):
    """Get train and validation dataloaders for VQ-VAE."""
    dataset = VQVAEDataset(cfg, fold='train')

    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_data, val_data = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg['seed'])
    )
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    loader_args = {
        'batch_size': cfg['batch_size'],
        'num_workers': cfg['num_workers'],
        'pin_memory': device.type == 'cuda',
    }
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **loader_args)

    return train_loader, val_loader

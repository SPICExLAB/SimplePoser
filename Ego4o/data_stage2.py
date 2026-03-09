"""
Stage 2 dataset: loads paired IMU + GT motion for training the IMU encoder.

Provides both:
  - IMU data: (T, 6, 9) — 5 real sensors (acc + rot6d) + 1 zero-padded
  - HumanML3D motion: (263, 1, T) — for frozen VQ-VAE target generation

Both are temporally aligned and windowed into fixed-length segments.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

import articulate as art
from config import paths, datasets, acc_scale
from Ego4o.humanml import process_file, compute_target_offsets


class Stage2Dataset(Dataset):
    def __init__(self, cfg, fold='train'):
        super().__init__()
        self.cfg = cfg
        self.fold = fold
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)
        self.tgt_offsets = compute_target_offsets()

        self.windows = []  # list of (imu_chunk, motion_chunk) tuples
        self._load_data()

    def _load_data(self):
        data_folder = Path(paths.data_dir) / self.cfg['dataset']

        if self.cfg.get('data_file'):
            data_files = [self.cfg['data_file']]
        elif (data_folder / 'train.pt').exists():
            data_files = ['test.pt'] if self.fold == 'test' else ['train.pt']
        else:
            data_files = [x.name for x in data_folder.iterdir() if not x.is_dir()]

        print(f"[Stage2] Loading {data_files} from {data_folder}")

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
        accs = file_data['acc']
        oris = file_data['ori']
        poses = file_data['pose']
        trans = file_data['tran']

        window = self.cfg['window_length']
        feet_thre = self.cfg.get('feet_thre', 0.002)
        n_success = 0
        n_fail = 0

        pbar = tqdm(
            enumerate(zip(accs, oris, poses, trans)),
            total=len(accs),
            desc="Processing sequences",
            leave=False
        )

        for idx, (acc, ori, pose, tran) in pbar:
            try:
                result = self._process_sequence(acc, ori, pose, tran, step, feet_thre)
                if result is None:
                    n_fail += 1
                    continue

                imu_aligned, motion_263 = result
                if len(motion_263) < window:
                    n_fail += 1
                    continue

                # Window into fixed-length segments
                motion_chunks = torch.split(motion_263, window)
                imu_chunks = torch.split(imu_aligned, window)
                for m_chunk, i_chunk in zip(motion_chunks, imu_chunks):
                    if len(m_chunk) == window and len(i_chunk) == window:
                        self.windows.append((i_chunk, m_chunk))
                n_success += 1
            except Exception:
                n_fail += 1
                continue

            pbar.set_postfix({'ok': n_success, 'fail': n_fail, 'windows': len(self.windows)})

        print(f"  Sequences: {n_success} success, {n_fail} failed | Windows: {len(self.windows)}")

    def _process_sequence(self, acc, ori, pose, tran, step, feet_thre):
        """Process one sequence into aligned (IMU, motion_263) tensors."""
        # --- IMU processing ---
        acc = acc[:, :5]  # (T, 5, 3)
        ori = ori[:, :5]  # (T, 5, 3, 3) or (T, 5, 9)

        # Downsample
        acc = acc[::step] / acc_scale
        ori = ori[::step]

        # Convert orientation to rot6d
        ori_flat = ori.reshape(-1, 3, 3) if ori.dim() >= 3 else ori.reshape(-1, 9).reshape(-1, 3, 3)
        ori_r6d = art.math.rotation_matrix_to_r6d(ori_flat.float())  # (-1, 6)
        ori_r6d = ori_r6d.reshape(len(acc), 5, 6)

        # Concatenate acc + rot6d per sensor
        imu = torch.cat([acc.float(), ori_r6d], dim=-1)  # (T, 5, 9)

        # Zero-pad to 6 sensors
        imu_padded = F.pad(imu, (0, 0, 0, 1))  # (T, 6, 9)

        # --- Motion processing ---
        pose = pose.view(-1, 24, 3, 3)
        tran = tran.view(-1, 3)

        # FK to get world-space joint positions
        _, joints = self.bodymodel.forward_kinematics(pose=pose)
        joints = joints.view(-1, 24, 3)
        positions = joints + tran.unsqueeze(1)  # (T, 24, 3)
        positions = positions[:, :22]  # (T, 22, 3)

        # Downsample
        positions = positions[::step]

        if len(positions) < 4:
            return None

        # Convert to HumanML3D 263-dim (returns T-1 frames)
        data_263, _, _, _ = process_file(positions.numpy(), feet_thre, self.tgt_offsets)
        if data_263 is None:
            return None

        motion = torch.from_numpy(data_263).float()  # (T-1, 263)

        # Align: drop first IMU frame so both are T-1 frames
        imu_aligned = imu_padded[1:len(motion) + 1]  # (T-1, 6, 9)

        if len(imu_aligned) != len(motion):
            return None

        return imu_aligned, motion

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        imu, motion = self.windows[idx]
        # imu: (T, 6, 9) — ready for encoder
        # motion: (T, 263) → (263, 1, T) for VQ-VAE
        motion_vqvae = motion.permute(1, 0).unsqueeze(1)  # (263, 1, T)
        return imu.float(), motion_vqvae.float()


def get_dataloaders(cfg, device):
    """Get train and validation dataloaders for Stage 2."""
    dataset = Stage2Dataset(cfg, fold='train')

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

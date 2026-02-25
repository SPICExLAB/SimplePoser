import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

import articulate as art
from config import combos, paths, acc_scale, vel_scale, fps, joint_set, datasets


class PoseDataset(Dataset):
    contact_joints = [0, 10, 11, 20, 21]  # pelvis, left foot, right foot, left hand, right hand

    def __init__(self, cfg: dict, fold: str='train', evaluate: str=None):
        super().__init__()
        self.cfg = cfg
        self.fold = fold
        self.evaluate = evaluate
        self.combos = combos
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)

        self.data = {
            'imu_inputs': [],
            'pose_outputs': [],
            'joint_outputs': [],
            'tran_outputs': [],
            'vel_outputs': [],
            'foot_outputs': [],
            'stationary_outputs': [],
            'root_vel_outputs': [],
        }

        self._load_data()

    def _get_data_files(self, data_folder: Path):
        return [x.name for x in data_folder.iterdir() if not x.is_dir()]

    def _load_data(self):
        data_folder = Path(paths.data_dir) / self.cfg['dataset']

        if self.cfg.get('data_file'):
            data_files = [self.cfg['data_file']]
        elif (data_folder / 'train.pt').exists():
            data_files = ['test.pt'] if self.evaluate else ['train.pt']
        else:
            data_files = self._get_data_files(data_folder)

        print(f"Loading {data_files} from {data_folder}")

        for data_file in tqdm(data_files):
            file_data = torch.load(data_folder / data_file, map_location=torch.device('cpu'), weights_only=True)
            dataset_name = Path(data_file).stem
            source_fps = datasets.get(dataset_name, {}).get('fps', self.cfg['target_fps'])
            step = max(1, round(source_fps / self.cfg['target_fps']))
            self._process_file(dataset_name, file_data, step)

    def _compute_stationary_labels(self, joint, tran, dt):
        """
        Compute stationary labels and root velocity.

        Args:
            joint: [T, 24, 3] root-relative joint positions
            tran:  [T, 3] root translation
            dt:    frame interval in seconds
        Returns:
            stationary: [T, 5] binary stationary labels
            root_vel:   [T, 3] root velocity in m/s
        """
        # contact joint positions
        contact_pos = joint[:, self.contact_joints] + tran.unsqueeze(1)  # [T, 5, 3]

        # velocity in m/s via finite difference
        contact_vel = torch.zeros_like(contact_pos)
        contact_vel[1:] = (contact_pos[1:] - contact_pos[:-1]) / dt
        speed = contact_vel.norm(dim=-1)  # [T, 5]

        # stationary = low speed
        threshold = self.cfg.get('stationary_threshold', 0.15)
        stationary = (speed < threshold).float()

        # root velocity in m/s
        root_vel = torch.zeros_like(tran)
        root_vel[1:] = (tran[1:] - tran[:-1]) / dt

        return stationary, root_vel

    def _process_file(self, dataset_name: str, file_data: dict, step: int = 1):
        accs, oris, poses, trans = file_data['acc'], file_data['ori'], file_data['pose'], file_data['tran']
        joints = file_data.get('joint', [None] * len(poses))
        foots = file_data.get('contact', [None] * len(poses))

        pbar = tqdm(
            enumerate(zip(accs, oris, poses, trans, joints, foots)),
            total=len(accs),
            desc=f"Processing {len(accs)} sequences",
            leave=False
        )

        for idx, (acc, ori, pose, tran, joint, foot) in pbar:
            pbar.set_postfix({'seq': idx, 'frames': len(pose)})

            if self.cfg['add_noise'] and not self.evaluate and dataset_name not in ['IMUPoser', 'DIP_IMU']:
                from imu_synthesis import syn_imu_from_smpl
                acc, gyro, ori = syn_imu_from_smpl(pose, tran)
                acc = acc.cpu()
                gyro = gyro.cpu()
                ori = ori.cpu()

            acc = acc[:, :5]
            ori = ori[:, :5]
            pose = pose.view(-1, 24, 3, 3)

            pose_global, joint = self.bodymodel.forward_kinematics(pose=pose)
            if self.cfg['use_global_pose'] and not self.evaluate:
                pose = pose_global

            joint = joint.view(-1, 24, 3)
            tran = tran.view(-1, 3)
            foot = foot.view(-1, 2) if foot is not None else None

            # downsample
            acc = acc[::step] / acc_scale
            ori = ori[::step]
            pose = pose[::step]
            tran = tran[::step]
            joint = joint[::step]
            foot = foot[::step] if foot is not None else None

            self._process_data(acc, ori, pose, joint, tran, foot)

    def _process_data(self, acc, ori, pose, joint, tran, foot):
        use_r6d = self.cfg['use_r6d_input']
        dt = 1.0 / self.cfg['target_fps']

        # compute stationary labels
        stationary, root_vel = self._compute_stationary_labels(joint, tran, dt)

        for _, c in self.combos.items():
            combo_acc = torch.zeros_like(acc)
            combo_ori = torch.zeros_like(ori)
            combo_acc[:, c] = acc[:, c]
            combo_ori[:, c] = ori[:, c]

            if use_r6d:
                combo_ori = art.math.rotation_matrix_to_r6d(combo_ori).view(-1, 5, 6)

            imu = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)
            window = len(imu) if self.evaluate else self.cfg['window_length']

            self.data['imu_inputs'].extend(torch.split(imu, window))
            self.data['pose_outputs'].extend(torch.split(pose, window))
            self.data['joint_outputs'].extend(torch.split(joint, window))
            self.data['tran_outputs'].extend(torch.split(tran, window))

            # per-joint velocities
            vel = torch.cat([torch.zeros(1, 24, 3), torch.diff(joint, dim=0)])
            vel[:, 1:] = vel[:, 1:] - vel[:, :1]
            vel[:, 0] = torch.cat([torch.zeros(1, 3), tran[1:] - tran[:-1]])
            vel = vel * (fps / vel_scale)

            if not self.evaluate:
                self.data['vel_outputs'].extend(torch.split(vel, window))
                self.data['foot_outputs'].extend(
                    torch.split(foot, window) if foot is not None
                    else [None] * len(torch.split(imu, window))
                )
                self.data['stationary_outputs'].extend(
                    torch.split(stationary, window) if stationary is not None
                    else [None] * len(torch.split(imu, window))
                )
                self.data['root_vel_outputs'].extend(
                    torch.split(root_vel, window) if root_vel is not None
                    else [None] * len(torch.split(imu, window))
                )

    def __len__(self):
        return len(self.data['imu_inputs'])

    def __getitem__(self, idx):
        imu = self.data['imu_inputs'][idx].float()
        joint = self.data['joint_outputs'][idx].float()
        tran = self.data['tran_outputs'][idx].float()

        pose_6d = art.math.rotation_matrix_to_r6d(self.data['pose_outputs'][idx])
        n_joints = len(joint_set.full)
        pose_6d = pose_6d.reshape(-1, n_joints, 6)[:, joint_set.full].reshape(-1, 6 * n_joints)

        vel = self.data['vel_outputs'][idx].float() if self.data['vel_outputs'] and self.data['vel_outputs'][idx] is not None else torch.zeros(imu.shape[0], 24, 3)
        contact = self.data['foot_outputs'][idx].float() if self.data['foot_outputs'] and self.data['foot_outputs'][idx] is not None else torch.zeros(imu.shape[0], 2)
        stationary = self.data['stationary_outputs'][idx].float() if self.data['stationary_outputs'] and self.data['stationary_outputs'][idx] is not None else torch.zeros(imu.shape[0], 5)
        root_vel = self.data['root_vel_outputs'][idx].float() if self.data['root_vel_outputs'] and self.data['root_vel_outputs'][idx] is not None else torch.zeros(imu.shape[0], 3)

        return imu, pose_6d, joint, tran, vel, contact, stationary, root_vel


def collate_fn(batch):
    """Pad variable-length sequences for batching."""
    imus, poses, joints, trans, vels, contacts, stationaries, root_vels = zip(*batch)

    imus = nn.utils.rnn.pad_sequence(imus, batch_first=True)
    poses = nn.utils.rnn.pad_sequence(poses, batch_first=True)
    joints = nn.utils.rnn.pad_sequence(joints, batch_first=True)
    trans = nn.utils.rnn.pad_sequence(trans, batch_first=True)
    vels = nn.utils.rnn.pad_sequence(vels, batch_first=True)
    contacts = nn.utils.rnn.pad_sequence(contacts, batch_first=True)
    stationaries = nn.utils.rnn.pad_sequence(stationaries, batch_first=True)
    root_vels = nn.utils.rnn.pad_sequence(root_vels, batch_first=True)

    lengths = torch.tensor([len(x) for x in imus])

    return imus, poses, joints, trans, vels, contacts, stationaries, root_vels, lengths


def get_dataloaders(cfg, device):
    """Get train and validation dataloaders."""
    dataset = PoseDataset(cfg, fold='train')

    n_train = int(0.9 * len(dataset))
    train_data, val_data = torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(cfg['seed'])
    )
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    loader_args = {
        'batch_size': cfg['batch_size'],
        'collate_fn': collate_fn,
        'num_workers': cfg['num_workers'],
        'pin_memory': device.type == 'cuda'
    }
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **loader_args)

    return train_loader, val_loader
import torch
from torch.utils.data import Dataset

import articulate as art
from config import acc_scale


class SynPairedIMUData(Dataset):
    def __init__(self, data_path: str, normalize=False):
        super().__init__()

        # load the data from the data_path
        data = torch.load(data_path, weights_only=False)
        syn_acc = torch.cat(data['syn_acc'], dim=0)   # [N, 6, 3]
        syn_ori = torch.cat(data['syn_ori'], dim=0)   # [N, 6, 3, 3]
        real_acc = torch.cat(data['acc'], dim=0)      # [N, 5, 3]
        real_ori = torch.cat(data['ori'], dim=0)      # [N, 5, 3, 3]

        # ignore root data
        syn_acc = syn_acc[:, :5] # [N, 5, 3]
        syn_ori = syn_ori[:, :5] # [N, 5, 3, 3]

        self.normalize = normalize
        self.syn_imu = self._process_data(syn_acc, syn_ori)
        self.real_imu = self._process_data(real_acc, real_ori)
    
    def _process_data(self, acc, ori):
        """
        Some preprocessing for the data. Scale normalization and convert to 6D. 
        
        Args:
            acc: [N, 5, 3]
            ori: [N, 5, 3, 3]
        Returns:
            imu: [N, 36]
        """
        if self.normalize:
            # normalize data to root (3 -> right pocket)
            root_acc = acc[:, 3:4] # [N, 1, 3]
            root_ori = ori[:, 3]   # [N, 3, 3]
           
            # relative acc
            acc_diff = (acc - root_acc).unsqueeze(-1)          # (N, 5, 3, 1)
            acc = (root_ori[:, None] @ acc_diff).squeeze(-1)   # (N, 5, 3)

            # relative ori
            ori = root_ori.transpose(-1, -2)[:, None] @ ori    # (N, 5, 3, 3)

        acc = acc / acc_scale
        ori = art.math.rotation_matrix_to_r6d(ori).view(-1, 5, 6)
        imu = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)
        return imu

    def __len__(self):
        return len(self.syn_imu)

    def __getitem__(self, idx):
        return self.syn_imu[idx], self.real_imu[idx]


if __name__ == '__main__':
    # test
    data_path = '/data/projects/Pose/dataset_work/IMUPoser/train.pt'
    dataset = SynPairedIMUData(data_path)
    
    print(f"\nDataset size: {len(dataset)}")
    
    syn, real = dataset[0]
    print(f"Sample shape: syn={syn.shape}, real={real.shape}")
    
    # analyze secondary motion in world frame
    diff = (dataset.real_imu - dataset.syn_imu)
    print(f"\nSecondary motion (world frame):")
    print(f"  Acc mean: {diff[:, :12].mean():.4f}")
    print(f"  Acc std: {diff[:, :12].std():.4f}")
    print(f"  Ori mean: {diff[:, 12:].mean():.4f}")
    print(f"  Ori std: {diff[:, 12:].std():.4f}")
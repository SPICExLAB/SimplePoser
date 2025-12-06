# vis_paired_imu.py
import os
import torch
import matplotlib.pyplot as plt

from semoae.semoae import SemoAE
from semoae.dataset import SynPairedIMUData


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def plot_and_save(x, curves, labels, title, ylabel, out_path, hline0=False):
    plt.figure(figsize=(14, 4))
    for c, lab in zip(curves, labels):
        plt.plot(x, c, label=lab, linewidth=1.8)
    if hline0:
        plt.axhline(0, color='black', linewidth=1)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# --------------------------------------------------------
# Visualization
# --------------------------------------------------------

def visualize(clean, aug, real, imu=0, axis=0, out_dir="vis"):
    mkdir(out_dir)

    idx = imu * 3 + axis
    cs, as_, rs = clean[:, idx], aug[:, idx], real[:, idx]
    T = len(cs)
    x = torch.arange(T)

    # 1) full signals
    plot_and_save(
        x,
        [cs, as_, rs],
        ["clean synthetic", "augmented", "real IMU"],
        f"IMU {imu}, axis {axis} — clean vs aug vs real",
        "Value",
        f"{out_dir}/clean_aug_real_IMU{imu}_axis{axis}.png"
    )

    # 2) noise only
    nr, na = rs - cs, as_ - cs
    plot_and_save(
        x,
        [nr, na],
        ["real - clean", "aug - clean"],
        f"Noise comparison IMU {imu}, axis {axis}",
        "Noise magnitude",
        f"{out_dir}/noise_IMU{imu}_axis{axis}.png",
        hline0=True
    )

    # 3) stats
    print("-------------------------------------------------")
    print(f"IMU {imu}, axis {axis}")
    print(f" std(real-clean) = {nr.std():.6f}")
    print(f" std(aug-clean)  = {na.std():.6f}")
    print(f" mean(real-clean)= {nr.mean():.6f}")
    print(f" mean(aug-clean) = {na.mean():.6f}")
    print("-------------------------------------------------")


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():
    ckpt_path = "semoae/checkpoints/semoae_best.pt"
    data_path = "/data/projects/Pose/dataset_work/IMUPoser/train.pt"
    eta = 0.5
    seq_len = 150
    imu_to_plot, axis_to_plot = 3, 0

    # load data
    dataset = SynPairedIMUData(data_path, normalize=False)

    # build sequence
    clean_seq = torch.stack([dataset[i][0] for i in range(seq_len)])
    real_seq  = torch.stack([dataset[i][1] for i in range(seq_len)])

    # load model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = SemoAE(feat_dim=45, encode_dim=32)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # generate augmentation
    with torch.no_grad():
        aug_seq = model.add_secondary_motion(clean_seq.unsqueeze(0), eta)[0]

    # visualize
    visualize(clean_seq, aug_seq, real_seq,
              imu=imu_to_plot, axis=axis_to_plot, out_dir="vis")


if __name__ == "__main__":
    main()

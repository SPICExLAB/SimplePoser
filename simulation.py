import numpy as np
import torch
import articulate as art


def add_gaussian_noise(x, sigma=0.025):
    """Add Gaussian noise to a tensor."""
    return x + torch.randn_like(x) * sigma


def simulation_MODA(imu_rot, imu_acc, imu_num=5, acc_noise=0.025, random_global_yaw=True):
    """
    MODA: Motion-Drift Augmentation with gravity leakage correction.

    Based on: Wu et al., "MODA: Motion-Drift Augmentation for Inertial
    Human Motion Analysis", CVPR 2025.

    Args:
        imu_rot: [B, T, imu_num, 3, 3] rotation matrices
        imu_acc: [B, T, imu_num, 3, 1] accelerations
        imu_num: number of IMU sensors
        acc_noise: std of Gaussian noise added to acceleration
        random_global_yaw: whether to apply random yaw rotation
    Returns:
        imu_rot: [B, T, imu_num, 3, 3] augmented rotations
        imu_acc: [B, T, imu_num, 3, 1] augmented accelerations
        offset:  [B, T, imu_num, 6] drift offset in r6d
    """
    device = imu_rot.device
    B, T = imu_rot.shape[0], imu_rot.shape[1]
    GA = torch.FloatTensor([[0, -9.80665, 0]]).to(device)

    # acc noise
    imu_acc = add_gaussian_noise(imu_acc, sigma=acc_noise)

    # rotational drift (random walk on SO(3))
    sigma = np.pi / 32
    delta_euler = torch.randn(B, T, imu_num, 3, device=device) * sigma

    delta_rot = art.math.euler_angle_to_rotation_matrix(
        delta_euler.reshape(-1, 3), seq="YZX"
    ).reshape(B, T, imu_num, 3, 3)

    # cumulative product along time
    offset_rot = torch.zeros_like(delta_rot)
    offset_rot[:, 0] = delta_rot[:, 0]
    for t in range(1, T):
        offset_rot[:, t] = offset_rot[:, t - 1].matmul(delta_rot[:, t])

    # random global yaw
    if random_global_yaw:
        global_yaw_euler = torch.zeros(B, 1, 3, device=device)
        global_yaw_euler[:, :, 1] = global_yaw_euler[:, :, 1].uniform_(-np.pi, np.pi)
        global_yaw_rot = art.math.euler_angle_to_rotation_matrix(
            global_yaw_euler.reshape(-1, 3)
        ).reshape(B, 1, 1, 3, 3).expand(-1, T, imu_num, -1, -1)

        imu_rot = global_yaw_rot.matmul(imu_rot)
        if imu_acc is not None:
            imu_acc = global_yaw_rot.matmul(imu_acc)

    # apply rotational drift to orientation
    imu_rot = imu_rot.matmul(offset_rot)

    # gravity leakage correction for acceleration
    if imu_acc is not None:
        GA_expanded = GA.reshape(1, 1, 1, 3, 1).expand_as(imu_acc)
        imu_acc = offset_rot.matmul(imu_acc) + \
                  (torch.eye(3, device=device) - offset_rot).matmul(GA_expanded)

    offset = art.math.rotation_matrix_to_r6d(
        offset_rot.reshape(-1, 3, 3)
    ).reshape(B, T, imu_num, 6)

    return imu_rot, imu_acc, offset

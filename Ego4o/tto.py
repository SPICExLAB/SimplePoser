"""
Test-Time Optimization (TTO) for Ego4o Stage 3.

Refines VQ-VAE quantized codes at inference time by comparing decoded
motion (as limb orientations) against observed IMU orientations.

Uses LBFGS optimizer with SmoothL1Loss, following the official ego4o
imuposer_encoder.py implementation.
"""

import torch
import torch.nn as nn

from Ego4o.motion_utils import rotation_6d_to_matrix, recover_from_ric


def get_limb_orientation_from_imu(imu):
    """
    Extract limb direction vectors and availability masks from IMU data.

    Our sensor ordering (from imu_synthesis.py j_imu[:5] + zero-pad):
        0: right wrist (SMPL 18)
        1: left wrist  (SMPL 19)
        2: left hip    (SMPL 1)
        3: right hip   (SMPL 2)
        4: head        (SMPL 15) — always disabled
        5: zero-padded — auto-masked

    Args:
        imu: (B, T, 6, 9) where dim 9 = [acc(3), rot6d(6)]
    Returns:
        dict with keys 'left_arm', 'right_arm', 'left_leg', 'right_leg'
        each value is (orientation: (B, T, 3), availability: (B, T))
    """
    imu_rot6d = imu[..., 3:9].detach()  # (B, T, 6, 6)
    imu_rot_matrix = rotation_6d_to_matrix(imu_rot6d)  # (B, T, 6, 3, 3)

    # Detect valid sensors via rotation matrix determinant
    det = torch.linalg.det(imu_rot_matrix)  # (B, T, 6)
    sensor_available = det > 0.1

    # Disable head (sensor 4) — always masked
    sensor_available[:, :, 4] = False

    # Extract limb orientations using our sensor indices
    # Arms: first column of rotation matrix (bone direction along x-axis)
    left_arm_ori = imu_rot_matrix[:, :, 1, :3, 0]         # sensor 1 = left wrist
    left_arm_avail = sensor_available[:, :, 1].float()

    right_arm_ori = -1 * imu_rot_matrix[:, :, 0, :3, 0]   # sensor 0 = right wrist, negated
    right_arm_avail = sensor_available[:, :, 0].float()

    # Legs: second column of rotation matrix (bone direction along y-axis)
    left_leg_ori = -1 * imu_rot_matrix[:, :, 2, :3, 1]    # sensor 2 = left hip, negated
    left_leg_avail = sensor_available[:, :, 2].float()

    right_leg_ori = -1 * imu_rot_matrix[:, :, 3, :3, 1]   # sensor 3 = right hip, negated
    right_leg_avail = sensor_available[:, :, 3].float()

    return {
        'left_arm': (left_arm_ori, left_arm_avail),
        'right_arm': (right_arm_ori, right_arm_avail),
        'left_leg': (left_leg_ori, left_leg_avail),
        'right_leg': (right_leg_ori, right_leg_avail),
    }


def get_limb_orientation_from_joints(joint_positions):
    """
    Extract normalized limb direction vectors from joint positions.

    HumanML3D 22-joint indices:
        left_arm:  joint 20 (left hand) - joint 18 (left forearm)
        right_arm: joint 21 (right hand) - joint 19 (right forearm)
        left_leg:  joint 4 (left knee)  - joint 1 (left hip)
        right_leg: joint 5 (right knee) - joint 2 (right hip)

    Args:
        joint_positions: (B, T, 22, 3)
    Returns:
        dict with keys 'left_arm', 'right_arm', 'left_leg', 'right_leg'
        each value is (B, T, 3) normalized direction vector
    """
    left_arm = joint_positions[:, :, 20] - joint_positions[:, :, 18]
    left_arm = left_arm / torch.norm(left_arm, dim=-1, keepdim=True)

    right_arm = joint_positions[:, :, 21] - joint_positions[:, :, 19]
    right_arm = right_arm / torch.norm(right_arm, dim=-1, keepdim=True)

    left_leg = joint_positions[:, :, 4] - joint_positions[:, :, 1]
    left_leg = left_leg / torch.norm(left_leg, dim=-1, keepdim=True)

    right_leg = joint_positions[:, :, 5] - joint_positions[:, :, 2]
    right_leg = right_leg / torch.norm(right_leg, dim=-1, keepdim=True)

    return {
        'left_arm': left_arm,
        'right_arm': right_arm,
        'left_leg': left_leg,
        'right_leg': right_leg,
    }


def optimize_codes(vqvae, x_quantized_init, imu,
                   lr=0.5, max_iter=50, history_size=20, acc_weight=0.0):
    """
    LBFGS optimization of VQ-VAE quantized codes against observed IMU.

    Args:
        vqvae: Frozen VQLimbHML
        x_quantized_init: list of 6 tensors, each (B, code_dim, T')
        imu: (B, T, 6, 9) raw IMU data
        lr: LBFGS learning rate
        max_iter: LBFGS max iterations per step
        history_size: LBFGS history size
        acc_weight: weight for acceleration loss (0 = disabled)
    Returns:
        list of 6 optimized quantized code tensors
    """
    # Extract GT limb orientations from IMU
    limb_ori_gt = get_limb_orientation_from_imu(imu)

    # Detach codes and make them optimizable
    free_vars = []
    for code in x_quantized_init:
        v = code.detach().clone()
        v.requires_grad = True
        free_vars.append(v)

    optimizer = torch.optim.LBFGS(
        free_vars,
        lr=lr,
        max_iter=max_iter,
        tolerance_change=1e-6,
        max_eval=None,
        history_size=history_size,
        line_search_fn='strong_wolfe',
    )

    data_loss = nn.SmoothL1Loss(reduction='mean')

    def closure():
        optimizer.zero_grad()

        # Decode current codes → motion
        motion_hml = vqvae.forward_decoder_from_quantized_codes(free_vars)
        # (B, 263, 1, T) → (B, T, 263)
        motion_data = motion_hml.squeeze(2).permute(0, 2, 1)

        # Convert to joint positions
        joint_positions = recover_from_ric(motion_data, 22)  # (B, T, 22, 3)

        # Extract predicted limb orientations
        limb_ori_pred = get_limb_orientation_from_joints(joint_positions)

        # Compute masked orientation loss
        loss = torch.tensor(0.0, device=imu.device, requires_grad=True)
        for limb_name in ('left_arm', 'right_arm', 'left_leg', 'right_leg'):
            ori_gt, avail = limb_ori_gt[limb_name]
            ori_pred = limb_ori_pred[limb_name]
            mask = avail[..., None]  # (B, T, 1)
            loss = loss + data_loss(ori_pred * mask, ori_gt * mask)

        loss.backward(retain_graph=True)
        return loss

    optimizer.step(closure)

    return free_vars

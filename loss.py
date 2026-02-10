import torch
import torch.nn as nn

from utils import rotation_matrix_to_axis_angle_differentiable


def compute_vel_loss(pred_vel: torch.Tensor, gt_vel: torch.Tensor, n: int=1) -> torch.Tensor:
    """
    Compute velocity loss.
    Args:
        pred_vel: [B, T, ...] predicted velocity
        gt_vel: [B, T, ...] ground truth velocity
        n: int number of frames to skip
    Returns:
        loss: [B] velocity loss
    """
    mse = nn.MSELoss()
    T = pred_vel.shape[1]
    loss = 0.0

    B, T, _ = pred_vel.shape
    pred_vel = pred_vel.view(B, T, -1)
    gt_vel = gt_vel.view(B, T, -1)

    for m in range(0, T//n):
        end = min(n*m+n, T)
        loss += mse(pred_vel[:, m*n:end, :], gt_vel[:, m*n:end, :])

    return loss


def compute_jerk_loss(pred_motion):
    """
    Compute jerk loss.
    Args:
        pred_motion: [B, T, D] predicted motion
    Returns:
        jerk_loss: [B] jerk loss
    """
    jerk = pred_motion[:, 3:, :] - 3*pred_motion[:, 2:-1, :] + 3*pred_motion[:, 1:-2, :] - pred_motion[:, :-3, :]
    l1_norm = torch.norm(jerk, p=1, dim=2)
    return l1_norm.sum(dim=1).mean()


def compute_loss(pred_pose, pred_joints, pred_vel, pred_contact, 
                 gt_pose, gt_joints, gt_vel, gt_contact):
    B, T = pred_pose.shape[:2]
    
    # mse losses
    mse = nn.MSELoss()
    pose_loss = mse(pred_pose, gt_pose)
    joints_loss = mse(pred_joints, gt_joints.view(B, T, -1))
    pose_jerk_loss = compute_jerk_loss(pred_pose)
    joints_jerk_loss = compute_jerk_loss(pred_joints)
    vel_loss = sum(compute_vel_loss(pred_vel, gt_vel, i) for i in [1, 3, 9])
    
    # binary cross entropy for contacts
    bce = nn.BCEWithLogitsLoss()
    contact_loss = bce(pred_contact, gt_contact)
    
    total_loss = pose_loss + 1e-5 * pose_jerk_loss + 1e-5 * joints_jerk_loss + joints_loss + 0.5 * vel_loss + contact_loss
    
    return total_loss, {
        'pose': pose_loss.item(),
        'joints': joints_loss.item(),
        'velocity': vel_loss.item(),
        'contact': contact_loss.item()
    }


def compute_vposer_loss(vposer, pred_pose_6d: torch.Tensor, reduced_indices: list,
                        stride: int = 1, global_to_local_fn=None) -> torch.Tensor:
    """
    Compute VPoser prior loss: ||z||^2 where z = vposer.encode(pose).mean,
    encouraging the latent code to stay close to the prior N(0, I).

    Args:
        vposer: Loaded VPoser model
        pred_pose_6d: [B, T, n_reduced*6] predicted 6D rotations
        reduced_indices: list of joint indices in reduced set
        stride: frame stride for subsampling (default 1 = all frames)
        global_to_local_fn: function to convert global rotations to local (if using global pose)

    Returns:
        Scalar tensor, mean squared norm of latent codes
    """
    from articulate.math import r6d_to_rotation_matrix

    B = pred_pose_6d.shape[0]

    # subsample frames
    pred_pose_6d = pred_pose_6d[:, ::stride, :].contiguous()
    T = pred_pose_6d.shape[1]

    # convert 6D to rotation matrices
    pred_rotmat = r6d_to_rotation_matrix(pred_pose_6d.view(-1, 6)).view(B, T, -1, 3, 3)

    # scatter reduced joints into full 24-joint pose (identity for missing joints)
    full_rotmat = torch.eye(3, device=pred_pose_6d.device).expand(B, T, 24, 3, 3).clone()
    full_rotmat[:, :, reduced_indices] = pred_rotmat

    # convert global to local if needed
    if global_to_local_fn is not None:
        full_rotmat = global_to_local_fn(full_rotmat.view(-1, 24, 3, 3)).view(B, T, 24, 3, 3)

    # convert to axis-angle (use differentiable)
    full_pose_aa = rotation_matrix_to_axis_angle_differentiable(full_rotmat).view(B, T, 24, 3)

    # VPoser expects body joints 1-21 (exclude root and hands 22-23)
    body_pose_aa = full_pose_aa[:, :, 1:22].contiguous().view(B * T, 63)

    z = vposer.encode(body_pose_aa).mean
    return (z ** 2).sum(dim=-1).mean()
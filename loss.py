import torch
import torch.nn as nn


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
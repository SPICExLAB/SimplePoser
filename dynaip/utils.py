import torch
import articulate as art
from config import joint_set


def r6d_to_local(pred_pose, global_to_local_pose):
    """Convert predicted 6D rotations to full 24-joint local rotation matrices.
    Args:
        pred_pose: [N, 96] or [B, T, 96] 6D rotations for reduced joints
        global_to_local_pose: IK function (global rotations -> local rotations)
    Returns:
        local: [N, 24, 3, 3] local rotation matrices
    """
    pose = art.math.r6d_to_rotation_matrix(pred_pose).view(-1, joint_set.n_reduced, 3, 3)
    full = torch.eye(3, device=pose.device).expand(pose.shape[0], 24, 3, 3).clone()
    full[:, joint_set.reduced] = pose
    local = global_to_local_pose(full)
    local[:, joint_set.ignored] = torch.eye(3, device=pose.device)
    local[:, 0] = pose[:, 0]
    return local

"""
Inverse HumanML3D conversion: 263-dim representation → 3D joint positions.

Ported from ego4o-code-release motion_process.py (recover_from_ric, recover_root_rot_pos).
Also includes rotation_6d_to_matrix for TTO sensor processing.
"""

import torch
import torch.nn.functional as F

from Ego4o.humanml import qrot, qinv


def rotation_6d_to_matrix(d6):
    """
    Convert 6D rotation representation to 3x3 rotation matrix via Gram-Schmidt.

    Args:
        d6: (..., 6) first two columns of rotation matrix
    Returns:
        (..., 3, 3) rotation matrix
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def recover_root_rot_pos(data):
    """
    Recover root rotation quaternion and position from HumanML3D 263-dim.

    data layout:
        [0]:   root angular velocity (y-axis, as arcsin value)
        [1:3]: root linear velocity (XZ plane)
        [3]:   root height (Y)

    Args:
        data: (..., 263)
    Returns:
        r_rot_quat: (..., 4) root rotation quaternions
        r_pos: (..., 3) root world positions
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)

    # Integrate rotation velocity → absolute angle (shifted by 1 frame)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    # Y-axis rotation quaternion: (cos θ, 0, sin θ, 0)
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    # Integrate linear velocity → root position
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]

    # Rotate XZ velocity by inverse root rotation, then cumsum
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)

    # Set Y from root height
    r_pos[..., 1] = data[..., 3]

    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num=22):
    """
    Convert HumanML3D 263-dim representation to world-space joint positions.

    Args:
        data: (..., 263) HumanML3D representation
        joints_num: number of joints (22 for HumanML3D)
    Returns:
        positions: (..., joints_num, 3) world-space joint positions
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    # Extract rotation-invariant joint positions (21 joints × 3)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Rotate from root-local to world frame
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions,
    )

    # Add root XZ offset
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Prepend root position
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

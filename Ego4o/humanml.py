"""
HumanML3D 263-dim motion representation.

Consolidated from ego4o-code-release:
  - quaternion_original.py (quaternion math)
  - skeleton.py (Skeleton class with IK/FK)
  - motion_representation.py (process_file: positions → 263-dim)

The 263-dim representation contains:
  - Root data (4D): angular velocity, linear velocity XZ, root height
  - RIC data (63D): rotation-invariant joint positions (21 joints × 3)
  - Rotation data (126D): 6D continuous rotation (21 joints × 6)
  - Velocity data (66D): local-frame joint velocity (22 joints × 3)
  - Foot contact (4D): left/right foot binary contact (2 joints × 2)
"""

import numpy as np
import torch
import scipy.ndimage.filters as filters

import articulate as art
from config import paths


# ============================================================================
# Quaternion math (from quaternion_original.py)
# ============================================================================

def qinv(q):
    assert q.shape[-1] == 4
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qinv_np(q):
    return qinv(torch.from_numpy(q).float()).numpy()


def qnormalize(q):
    return q / torch.norm(q, dim=-1, keepdim=True)


def qmul(q, r):
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    original_shape = q.shape
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous().float()
    r = torch.from_numpy(r).contiguous().float()
    return qmul(q, r).numpy()


def qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()


def qfix(q):
    """Enforce quaternion continuity across the time dimension."""
    assert len(q.shape) == 3
    assert q.shape[-1] == 4
    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def qbetween(v0, v1):
    """Find quaternion that rotates v0 to v1."""
    assert v0.shape[-1] == 3
    assert v1.shape[-1] == 3
    v = torch.cross(v0, v1)
    w = torch.sqrt((v0 ** 2).sum(dim=-1, keepdim=True) * (v1 ** 2).sum(dim=-1, keepdim=True)) + \
        (v0 * v1).sum(dim=-1, keepdim=True)
    return qnormalize(torch.cat([w, v], dim=-1))


def qbetween_np(v0, v1):
    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    return qbetween(v0, v1).numpy()


def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_to_matrix_np(quaternions):
    q = torch.from_numpy(quaternions).contiguous().float()
    return quaternion_to_matrix(q).numpy()


def quaternion_to_cont6d_np(quaternions):
    rotation_mat = quaternion_to_matrix_np(quaternions)
    cont_6d = np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1)
    return cont_6d


def cont6d_to_matrix(cont6d):
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    return torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)


def cont6d_to_matrix_np(cont6d):
    q = torch.from_numpy(cont6d).contiguous().float()
    return cont6d_to_matrix(q).numpy()


# ============================================================================
# Skeleton (from skeleton.py)
# ============================================================================

class Skeleton:
    def __init__(self, offset, kinematic_tree, device):
        self.device = device
        self._raw_offset_np = offset.numpy()
        self._raw_offset = offset.clone().detach().to(device).float()
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j - 1]

    def set_offset(self, offsets):
        self._offset = offsets.clone().detach().to(self.device).float()

    def get_offsets_joints(self, joints):
        """Compute bone offsets from joint positions. joints: (22, 3)"""
        assert len(joints.shape) == 2
        _offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]
        self._offset = _offsets.detach()
        return _offsets

    def get_offsets_joints_batch(self, joints):
        """Compute bone offsets from batched joint positions. joints: (B, 22, 3)"""
        assert len(joints.shape) == 3
        _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[:, i] = torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None] * _offsets[:, i]
        self._offset = _offsets.detach()
        return _offsets

    def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False):
        """Inverse kinematics: positions → quaternion params. joints: (T, 22, 3)"""
        assert len(face_joint_idx) == 4
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[:, np.newaxis]

        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
        forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        quat_params = np.zeros(joints.shape[:-1] + (4,))
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat

        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                u = self._raw_offset_np[chain[j + 1]][np.newaxis, ...].repeat(len(joints), axis=0)
                v = joints[:, chain[j + 1]] - joints[:, chain[j]]
                v = v / np.sqrt((v ** 2).sum(axis=-1))[:, np.newaxis]
                rot_u_v = qbetween_np(u, v)
                R_loc = qmul_np(qinv_np(R), rot_u_v)
                quat_params[:, chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)
        return quat_params

    def forward_kinematics_np(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                R = qmul_np(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
        return joints


# ============================================================================
# T2M skeleton constants
# ============================================================================

t2m_raw_offsets = np.array([
    [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
    [0, -1, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0],
    [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, 0, 0],
    [0, 0, 1], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
    [0, -1, 0], [0, -1, 0],
])

t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],   # right leg
    [0, 1, 4, 7, 10],   # left leg
    [0, 3, 6, 9, 12, 15],  # spine → head
    [9, 14, 17, 19, 21],   # right arm
    [9, 13, 16, 18, 20],   # left arm
]

# Joint indices for face direction detection
face_joint_indx = [2, 1, 17, 16]  # r_hip, l_hip, sdr_r, sdr_l

# Foot joint indices for contact detection
fid_r, fid_l = [8, 11], [7, 10]

# Leg bone indices for uniform skeleton scaling
l_idx1, l_idx2 = 5, 8


# ============================================================================
# Target skeleton offsets (computed from SMPL T-pose instead of HumanML3D file)
# ============================================================================

def compute_target_offsets(smpl_file=None):
    """Compute reference skeleton offsets from SMPL T-pose (first 22 joints)."""
    if smpl_file is None:
        smpl_file = paths.smpl_file
    body_model = art.model.ParametricModel(smpl_file)
    j, _ = body_model.get_zero_pose_joint_and_vertex()
    t2m_joints = j[:22]  # (22, 3)

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    skel = Skeleton(n_raw_offsets, t2m_kinematic_chain, 'cpu')
    tgt_offsets = skel.get_offsets_joints(t2m_joints)
    return tgt_offsets


# ============================================================================
# HumanML3D conversion: process_file (from motion_representation.py)
# ============================================================================

def uniform_skeleton(positions, target_offset):
    """Normalize skeleton proportions to match reference skeleton."""
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    src_skel = Skeleton(n_raw_offsets, t2m_kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    # Scale ratio based on leg length
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()
    scale_rt = tgt_leg_len / src_leg_len

    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    # IK → FK with target skeleton
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def process_file(positions, feet_thre, tgt_offsets, uniform=True):
    """
    Convert 22-joint positions to HumanML3D 263-dim representation.

    Args:
        positions: (T, 22, 3) numpy array of world-space joint positions
        feet_thre: foot contact velocity threshold
        tgt_offsets: target skeleton offsets for uniform skeleton normalization
        uniform: whether to apply uniform skeleton normalization

    Returns:
        data: (T-1, 263) numpy array
        global_positions: (T, 22, 3) processed global positions
        positions: (T, 22, 3) rotation-invariant positions
        l_velocity: (T-1, 2) root linear velocity XZ
    """
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)

    if uniform:
        positions = uniform_skeleton(positions, tgt_offsets)

    # Put on floor
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # XZ at origin
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # All initially face Z+
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    global_positions = positions.copy()

    # Foot contact detection
    def foot_detect(positions, thres):
        velfactor = np.array([thres, thres])
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    # Continuous 6D rotation + root rotation/velocity
    r_rot = None

    def get_rifke(positions):
        """Rotation-invariant positions."""
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, t2m_kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()

        # Root linear velocity
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)

        # Root angular velocity
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    # Root height
    root_y = positions[:, 0, 1:2]

    # Root rotation velocity (y-axis) and linear velocity (xz plane)
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)  # (T-1, 4)

    # Joint rotation (6D continuous) — exclude root
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)  # (T, 126)

    # Rotation-invariant joint positions — exclude root
    ric_data = positions[:, 1:].reshape(len(positions), -1)  # (T, 63)

    # Joint velocity in local frame
    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1]
    )
    local_vel = local_vel.reshape(len(local_vel), -1)  # (T-1, 66)

    # Concatenate: (T-1, 263)
    data = np.concatenate([
        root_data,           # (T-1, 4)
        ric_data[:-1],       # (T-1, 63)
        rot_data[:-1],       # (T-1, 126)
        local_vel,           # (T-1, 66)
        feet_l, feet_r,      # (T-1, 2) + (T-1, 2)
    ], axis=-1)

    return data, global_positions, positions, l_velocity

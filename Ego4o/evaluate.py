"""
Evaluation script for Ego4o IMU-based pose estimation.

Extracts SMPL rotation matrices from predicted HumanML3D 263-dim output
and evaluates using the same FullMotionEvaluator as other models.

Usage:
    python -m Ego4o.evaluate --config configs/ego4o_predict.yaml
    python -m Ego4o.evaluate --config configs/ego4o_predict.yaml --tto
"""

import torch
from argparse import ArgumentParser
from tqdm import tqdm

from config import paths, joint_set, fps
import articulate as art
from Ego4o.predict import load_models, encode_and_decode, predict_with_tto
from Ego4o.data_stage2 import Stage2Dataset
from Ego4o.humanml import cont6d_to_matrix
from Ego4o.motion_utils import recover_root_rot_pos
from utils import load_yaml, set_seed


def hml263_to_rotations(data):
    """
    Extract SMPL-compatible rotation matrices from HumanML3D 263-dim.

    The 263-dim layout stores:
      - [0]:     root angular velocity (Y-axis, as arcsin)
      - [67:193]: 6D continuous rotations for joints 1-21 (21 × 6)

    These rotations come from IK on a SMPL-proportioned skeleton (via
    uniform_skeleton in process_file), so they're compatible with SMPL FK.

    Args:
        data: (T, 263) HumanML3D representation
    Returns:
        rotations: (T, 24, 3, 3) local rotation matrices
    """
    T = data.shape[0]
    device = data.device

    # Root rotation: integrate angular velocity → Y-axis rotation matrix
    r_rot_quat, _ = recover_root_rot_pos(data)  # (T, 4) quaternion
    # Convert quaternion (w, x, y, z) to rotation matrix
    r, i, j, k = r_rot_quat.unbind(-1)
    two_s = 2.0 / (r_rot_quat * r_rot_quat).sum(-1)
    root_rot = torch.stack([
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ], dim=-1).reshape(T, 3, 3)

    # Joint rotations 1-21: extract 6D → rotation matrices
    rot_6d = data[:, 67:193].reshape(T, 21, 6)
    joint_rots = cont6d_to_matrix(rot_6d)  # (T, 21, 3, 3)

    # Combine: root + 21 joints + 2 identity (hands)
    rotations = torch.eye(3, device=device).reshape(1, 1, 3, 3).expand(T, 24, -1, -1).clone()
    rotations[:, 0] = root_rot
    rotations[:, 1:22] = joint_rots

    return rotations


class PoseEvaluator:
    """Same evaluator as dynaip/evaluate.py — wraps FullMotionEvaluator."""

    def __init__(self, device='cpu'):
        self._eval_fn = art.FullMotionEvaluator(
            paths.smpl_file,
            joint_mask=torch.tensor([2, 5, 16, 20], device=device),
            fps=fps,
            device=device,
        )

    def eval(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)

        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[9], errs[3], errs[9], errs[0]*100, errs[7]*100, errs[1]*100, errs[4] / 100])

    @staticmethod
    def print(errors):
        for i, name in enumerate([
            'SIP Error (deg)', 'Angular Error (deg)', 'Masked Angular Error (deg)',
            'Positional Error (cm)', 'Masked Positional Error (cm)', 'Mesh Error (cm)',
            'Jitter Error (100m/s^3)'
        ]):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))


def evaluate_pose(encoder, vqvae, dataset, cfg):
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    evaluator = PoseEvaluator(device=device)

    use_tto = cfg.get('use_tto', False)
    tto_cfg = cfg.get('tto', {})

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.get('num_workers', 0),
    )

    errs_no_tto = []
    errs_tto = []

    print(f"\nEvaluating on {len(dataset)} windows | TTO: {use_tto}")

    for imu, motion_gt_hml in tqdm(loader, desc='Evaluating'):
        imu = imu.to(device)
        motion_gt_hml = motion_gt_hml.to(device)

        # GT: 263-dim → SMPL rotations
        gt_263 = motion_gt_hml.squeeze(0).squeeze(1).permute(1, 0)  # (T, 263)
        pose_t = hml263_to_rotations(gt_263)  # (T, 24, 3, 3)

        # Prediction without TTO
        with torch.no_grad():
            _, motion_pred = encode_and_decode(encoder, vqvae, imu)
        pred_263 = motion_pred.squeeze(0).squeeze(1).permute(1, 0)  # (T, 263)
        pose_p = hml263_to_rotations(pred_263)
        errs_no_tto.append(evaluator.eval(pose_p, pose_t))

        # Prediction with TTO (needs gradients for LBFGS)
        if use_tto:
            motion_pred_tto = predict_with_tto(encoder, vqvae, imu, tto_cfg)
            pred_263_tto = motion_pred_tto.squeeze(0).squeeze(1).permute(1, 0)
            pose_p_tto = hml263_to_rotations(pred_263_tto)
            errs_tto.append(evaluator.eval(pose_p_tto, pose_t))

    # Print results
    print('\n============== Ego4o (no TTO) =================')
    evaluator.print(torch.stack(errs_no_tto).mean(dim=0))

    if use_tto:
        print('\n============== Ego4o (with TTO) ================')
        evaluator.print(torch.stack(errs_tto).mean(dim=0))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--tto', action='store_true', help='Enable TTO')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.tto:
        cfg['use_tto'] = True

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['seed'])

    vqvae, encoder = load_models(cfg, device)
    dataset = Stage2Dataset(cfg, fold='test')

    evaluate_pose(encoder, vqvae, dataset, cfg)

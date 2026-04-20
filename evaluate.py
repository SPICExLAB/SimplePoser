import os
import torch
import tqdm
from argparse import ArgumentParser

import articulate as art
from config import paths, joint_set, fps
from data import PoseDataset
from utils import load_yaml
from mobileposer import MobilePoser
from dynaip import DynaIP


MODEL_REGISTRY = {
    'mobileposer': MobilePoser,
    'dynaip': DynaIP,
}


class PoseEvaluator:
    def __init__(self, device='cpu'):
        self._eval_fn = art.FullMotionEvaluator(
            paths.smpl_file, joint_mask=torch.tensor([2, 5, 16, 20], device=device),
            fps=fps, device=device,
        )

    def eval(self, pose_p, pose_t, tran_p=None, tran_t=None):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)

        if tran_p is not None and tran_t is not None:
            tran_p = tran_p.clone().view(-1, 3)
            tran_t = tran_t.clone().view(-1, 3)
            errs = self._eval_fn(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)
            return torch.stack([errs[9], errs[3], errs[9], errs[0]*100, errs[7]*100, errs[1]*100, errs[4] / 100, errs[6]])
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[9], errs[3], errs[9], errs[0]*100, errs[7]*100, errs[1]*100, errs[4] / 100])

    @staticmethod
    def print(errors, with_tran=True):
        names = ['SIP Error (deg)', 'Angular Error (deg)', 'Masked Angular Error (deg)',
                 'Positional Error (cm)', 'Masked Positional Error (cm)', 'Mesh Error (cm)',
                 'Jitter Error (100m/s^3)']
        if with_tran:
            names.append('Distance Error (cm)')
        for i, name in enumerate(names):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))


def _predict(model, imu):
    """Model-specific inference. Returns (pose [T,24,3,3], tran [T,3] or None)."""
    if isinstance(model, MobilePoser):
        pose, _, tran, _ = model.forward_offline(imu.unsqueeze(0))
        return pose, tran
    if isinstance(model, DynaIP):
        pose, tran, _ = model.predict(imu.unsqueeze(0))
        return pose, tran
    raise ValueError(f"Unsupported model type: {type(model).__name__}")


@torch.no_grad()
def evaluate_pose(model, dataset, cfg):
    device = cfg['device']
    evaluator = PoseEvaluator(device=device)
    errs = []

    model.eval()
    for imu, pose_6d, joint, tran_t, vel, contact, stationary, root_vel in tqdm.tqdm(dataset):
        imu = imu.to(device)
        pose_t = art.math.r6d_to_rotation_matrix(pose_6d.to(device))
        tran_t = tran_t.to(device)

        pose_p, tran_p = _predict(model, imu)
        errs.append(evaluator.eval(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t))

    print('============== average =================')
    evaluator.print(torch.stack(errs).mean(dim=0), with_tran=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--config', type=str, default='evaluate.yaml')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = cfg['device']

    ModelClass = MODEL_REGISTRY[cfg.get('model', 'dynaip')]
    model = ModelClass(cfg).to(device)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    dataset = PoseDataset(cfg, fold='test', evaluate=True)
    evaluate_pose(model, dataset, cfg)

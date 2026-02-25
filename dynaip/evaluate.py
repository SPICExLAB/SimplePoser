import torch
import tqdm
from argparse import ArgumentParser

from config import paths, joint_set, fps
import articulate as art
from data import PoseDataset
from dynaip import MODEL_REGISTRY
from utils import load_yaml


class PoseEvaluator:
    def __init__(self, device='cpu'):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([2, 5, 16, 20], device=device), fps=fps, device=device)

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


@torch.no_grad()
def evaluate_pose(model, dataset, cfg):
    device = cfg['device']
    evaluator = PoseEvaluator(device=device)
    errs = []

    model.eval()
    for imu, pose_6d, joint, tran, vel, contact in tqdm.tqdm(dataset):
        imu = imu.to(device)
        pose_t = art.math.r6d_to_rotation_matrix(pose_6d.to(device))
        pose_p, _, _ = model.predict(imu.unsqueeze(0))
        errs.append(evaluator.eval(pose_p.squeeze(0), pose_t))

    print('============== average =================')
    evaluator.print(torch.stack(errs).mean(dim=0))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = cfg['device']

    ModelClass = MODEL_REGISTRY[cfg.get('model', 'dynaip')]
    model = ModelClass(device=device).to(device)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    dataset = PoseDataset(cfg, fold='test', evaluate=True)
    evaluate_pose(model, dataset, cfg)

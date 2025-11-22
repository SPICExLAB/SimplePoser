import os
import numpy as np
import torch
from argparse import ArgumentParser
import tqdm

from config import paths, joint_set, fps
import articulate as art
from data import PoseDataset
from model import MobilePoser
from utils import load_yaml


class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([2, 5, 16, 20]), fps=fps)

    def eval(self, pose_p, pose_t, tran_p=None, tran_t=None):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        tran_p = tran_p.clone().view(-1, 3)
        tran_t = tran_t.clone().view(-1, 3)

        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)

        errs = self._eval_fn(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)
        return torch.stack([errs[9], errs[3], errs[9], errs[0]*100, errs[7]*100, errs[1]*100, errs[4] / 100, errs[6]])

    @staticmethod
    def print(errors):
        for i, name in enumerate([
            'SIP Error (deg)', 'Angular Error (deg)', 'Masked Angular Error (deg)',
            'Positional Error (cm)', 'Masked Positional Error (cm)', 'Mesh Error (cm)',
            'Jitter Error (100m/s^3)', 'Distance Error (cm)'
        ]):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))


@torch.no_grad()
def evaluate_pose(model, dataset, num_future_frame=5):
    # specify device
    device = cfg['device']

    # load data
    xs, ys = zip(*[
        (imu.to(device), (pose_6d.to(device), tran.to(device)))
        for imu, pose_6d, joint, tran, vel, contact in dataset
    ])

    xs = xs[:10]
    ys = ys[:10]

    # setup Pose Evaluator
    evaluator = PoseEvaluator()

    # track errors
    offline_errs, online_errs = [], []

    model.eval()
    with torch.no_grad():
        for x, y in tqdm.tqdm(list(zip(xs, ys))):
            model.reset()

            # offline
            pose_p_offline, joint_p_offline, tran_p_offline, _ = model.forward_offline(x.unsqueeze(0))

            pose_t_6d, tran_t = y
            pose_t = art.math.r6d_to_rotation_matrix(pose_t_6d)

            # online
            if os.getenv("ONLINE"):
                online_results = [model.forward_online(f) for f in torch.cat((x, x[-1].repeat(num_future_frame, 1)))]
                pose_p_online, joint_p_online, tran_p_online, _ = [torch.stack(_)[num_future_frame:] for _ in zip(*online_results)]

            # store errors
            offline_errs.append(evaluator.eval(pose_p_offline, pose_t, tran_p=tran_p_offline, tran_t=tran_t))

            if os.getenv("ONLINE"):
                online_errs.append(
                    evaluator.eval(pose_p_online, pose_t, tran_p=tran_p_online, tran_t=tran_t)
                )

    # print joint errors
    print('============== offline ================')
    evaluator.print(torch.stack(offline_errs).mean(dim=0))

    if os.getenv("ONLINE"):
        print('============== online ================')
        evaluator.print(torch.stack(online_errs).mean(dim=0))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='test')
    args = parser.parse_args()

    # load cfg
    cfg = load_yaml('config.yaml')

    # load model
    model = MobilePoser.from_pretrained(cfg, args.weights).to(cfg['device'])

    # load dataset
    dataset = PoseDataset(cfg, fold='test', evaluate=True)

    # evaluate
    evaluate_pose(model, dataset)

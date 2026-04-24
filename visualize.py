"""
    Test the system with an example IMU measurement sequence.
"""

import os
import torch
from argparse import ArgumentParser

import articulate as art
from config import paths, combos, acc_scale
from utils import load_yaml
from evaluate import MODEL_REGISTRY


def load_data(seq_num, combo):
    data = torch.load(os.path.join('data', 'TotalCapture.pt'))
    acc = data['acc'][seq_num][:, :5] / acc_scale
    ori = data['ori'][seq_num][:, :5]

    c = combos[combo]
    combo_acc = torch.zeros_like(acc)
    combo_ori = torch.zeros_like(ori)
    combo_acc[:, c] = acc[:, c]
    combo_ori[:, c] = ori[:, c]

    imu = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)
    return imu


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', default='tcnposer', choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--config', default='tcnposer_rf253.yaml')
    parser.add_argument('--weights', default='checkpoints/tcnposer_h96_b6/best.pt')
    parser.add_argument('--seq', type=int, default=0)
    parser.add_argument('--combo', default='rp')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = load_yaml(args.config)
    cfg['device'] = str(device)

    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(cfg).to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()

    imu = load_data(seq_num=args.seq, combo=args.combo).to(device)
    with torch.no_grad():
        if args.model == 'dynaip':
            pose, tran, _ = model.predict(imu.unsqueeze(0))
        else:
            pose, joints, tran, contact = model.forward_offline(imu.unsqueeze(0))

    art.ParametricModel(paths.smpl_file).view_motion([pose], [tran])


if __name__ == '__main__':
    main()

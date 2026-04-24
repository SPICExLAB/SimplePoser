"""Side-by-side eval: TCNPoser vs DynaIP on the evaluate.yaml test set."""
import torch
import tqdm
from argparse import ArgumentParser

import articulate as art
from config import paths, joint_set, fps
from data import PoseDataset
from utils import load_yaml
from evaluate import PoseEvaluator, _predict, MODEL_REGISTRY


METRIC_NAMES = [
    'SIP Error (deg)',
    'Angular Error (deg)',
    'Masked Angular Error (deg)',
    'Positional Error (cm)',
    'Masked Positional Error (cm)',
    'Mesh Error (cm)',
    'Jitter Error (100m/s^3)',
    'Distance Error (cm)',
]


def _load_model(model_key, weights, cfg, overrides=None):
    cfg = dict(cfg)
    cfg['model'] = model_key
    if overrides:
        cfg.update(overrides)
    model = MODEL_REGISTRY[model_key](cfg).to(cfg['device'])
    ckpt = torch.load(weights, map_location=cfg['device'], weights_only=True)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()
    return model


@torch.no_grad()
def _eval_model(model, dataset, device):
    evaluator = PoseEvaluator(device=device)
    errs = []
    for imu, pose_6d, joint, tran_t, vel, contact, stationary, root_vel in tqdm.tqdm(dataset, leave=False):
        imu = imu.to(device)
        pose_t = art.math.r6d_to_rotation_matrix(pose_6d.to(device))
        tran_t = tran_t.to(device)
        pose_p, tran_p = _predict(model, imu)
        errs.append(evaluator.eval(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t))
    return torch.stack(errs).mean(dim=0)


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default='evaluate.yaml')
    parser.add_argument('--dataset', default=None, help='Override cfg["dataset"] (e.g. IMUPoser, DIP_IMU)')
    parser.add_argument('--tcn-weights', default='checkpoints/tcnposer_h96_b6/best.pt')
    parser.add_argument('--tcn-hidden', type=int, default=96)
    parser.add_argument('--tcn-blocks', type=int, default=6)
    parser.add_argument('--tcn-label', default=None)
    parser.add_argument('--dynaip-weights', default='checkpoints/dynaip_imu2scene_jesse/best.pt')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.dataset is not None:
        cfg['dataset'] = args.dataset
    device = cfg['device']

    print(f"Test dataset: {cfg['dataset']}")
    dataset = PoseDataset(cfg, fold='test', evaluate=True)
    print(f"Test sequences: {len(dataset)}\n")

    tcn_label = args.tcn_label or f'TCNPoser (H={args.tcn_hidden}, {args.tcn_blocks} blocks)'
    runs = [
        (tcn_label,         'tcnposer', args.tcn_weights,
         {'hidden_main': args.tcn_hidden, 'blocks_main': args.tcn_blocks}),
        ('DynaIP (jesse)',  'dynaip',   args.dynaip_weights, None),
    ]

    results = {}
    for label, key, weights, overrides in runs:
        print(f"=== {label} ===")
        print(f"    weights: {weights}")
        model = _load_model(key, weights, cfg, overrides=overrides)
        results[label] = _eval_model(model, dataset, device)
        del model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    # Side-by-side table: mean (+/- std) per metric, per model
    labels = list(results.keys())
    col_w = 24
    print('\n' + '=' * (40 + col_w * len(labels)))
    print(f'{"Metric":<38}' + ''.join(f'{lab:>{col_w}}' for lab in labels))
    print('-' * (40 + col_w * len(labels)))
    for i, name in enumerate(METRIC_NAMES):
        row = f'{name:<38}'
        for lab in labels:
            mean, std = results[lab][i, 0].item(), results[lab][i, 1].item()
            row += f'{mean:>10.2f} (+/- {std:>5.2f})'.rjust(col_w)
        print(row)
    print('=' * (40 + col_w * len(labels)))


if __name__ == '__main__':
    main()

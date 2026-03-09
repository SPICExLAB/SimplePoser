"""
Inference pipeline for Ego4o IMU-based pose estimation.

Runs Stage 2 encoder + optional Stage 3 TTO, evaluates on test set.

Usage:
    python -m Ego4o.predict --config configs/ego4o_predict.yaml [--tto] [--save]
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

from Ego4o.encoder import IMUTransformerEncoder
from Ego4o.vqvae import VQLimbHML
from Ego4o.data_stage2 import Stage2Dataset
from Ego4o.motion_utils import recover_from_ric
from Ego4o.tto import optimize_codes
from utils import load_yaml, set_seed


def load_models(cfg, device):
    """Load frozen VQ-VAE and trained encoder from checkpoints."""
    # VQ-VAE
    vqvae = VQLimbHML(
        nb_code=cfg['nb_code'],
        code_dim=cfg['code_dim'],
        output_emb_width=cfg['output_emb_width'],
        down_t=cfg['down_t'],
        stride_t=cfg['stride_t'],
        width=cfg['width'],
        depth=cfg['depth'],
        dilation_growth_rate=cfg['dilation_growth_rate'],
        mu=cfg['mu'],
    ).to(device)

    ckpt = torch.load(cfg['vqvae_checkpoint'], map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        vqvae.load_state_dict(ckpt['model_state_dict'])
    else:
        vqvae.load_state_dict(ckpt)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    # Encoder
    encoder = IMUTransformerEncoder(
        input_dim=cfg['input_dim'],
        sensor_num=cfg['sensor_num'],
        nb_code=cfg['nb_code'],
        model_dim1=cfg['model_dim1'],
        model_dim2=cfg['model_dim2'],
        codes_realLength=cfg['codes_realLength'],
        nhead=cfg['nhead'],
        num_encoder_layers=cfg['num_encoder_layers'],
        num_decoder_layers=cfg['num_decoder_layers'],
        dropout=cfg['dropout'],
    ).to(device)

    ckpt = torch.load(cfg['encoder_checkpoint'], map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        encoder.load_state_dict(ckpt['model_state_dict'])
    else:
        encoder.load_state_dict(ckpt)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print(f"Loaded VQ-VAE from {cfg['vqvae_checkpoint']}")
    print(f"Loaded Encoder from {cfg['encoder_checkpoint']}")

    return vqvae, encoder


def encode_and_decode(encoder, vqvae, imu):
    """
    Run encoder → Gumbel-softmax → VQ-VAE decode.

    Args:
        encoder: Frozen IMUTransformerEncoder
        vqvae: Frozen VQLimbHML
        imu: (B, T, 6, 9)
    Returns:
        x_quantized: list of 6 tensors (for TTO)
        motion_hml: (B, 263, 1, T) decoded motion
    """
    pre_codes = encoder(imu)  # (B, 6, Tt, nb_code)
    codes_gumbel = F.gumbel_softmax(pre_codes, tau=1e-3, eps=1e-10, hard=True, dim=-1)
    x_quantized = vqvae.get_x_quantized_from_x_ids(
        codes_gumbel.permute(0, 2, 3, 1).contiguous()
    )
    motion_hml = vqvae.forward_decoder_from_quantized_codes(x_quantized)
    return x_quantized, motion_hml


def decode_to_positions(motion_hml):
    """
    Convert HumanML3D output to joint positions.

    Args:
        motion_hml: (B, 263, 1, T)
    Returns:
        positions: (B, T, 22, 3) world-space joint positions
    """
    data = motion_hml.squeeze(2).permute(0, 2, 1)  # (B, T, 263)
    return recover_from_ric(data, 22)


def compute_mpjpe(pred, gt):
    """
    Mean Per-Joint Position Error.

    Args:
        pred: (B, T, 22, 3)
        gt: (B, T, 22, 3)
    Returns:
        scalar MPJPE in meters
    """
    return torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean()


@torch.no_grad()
def predict_no_tto(encoder, vqvae, imu):
    """Run inference without TTO. Returns (B, 263, 1, T)."""
    _, motion_hml = encode_and_decode(encoder, vqvae, imu)
    return motion_hml


def predict_with_tto(encoder, vqvae, imu, tto_cfg):
    """
    Run inference with TTO.

    Args:
        encoder: Frozen encoder
        vqvae: Frozen VQ-VAE
        imu: (B, T, 6, 9)
        tto_cfg: dict with TTO hyperparameters
    Returns:
        motion_hml: (B, 263, 1, T) optimized decoded motion
    """
    with torch.no_grad():
        x_quantized, _ = encode_and_decode(encoder, vqvae, imu)

    optimized_codes = optimize_codes(
        vqvae=vqvae,
        x_quantized_init=x_quantized,
        imu=imu,
        lr=tto_cfg.get('lr', 0.5),
        max_iter=tto_cfg.get('max_iter', 50),
        history_size=tto_cfg.get('history_size', 20),
        acc_weight=tto_cfg.get('acc_weight', 0.0),
    )
    motion_hml = vqvae.forward_decoder_from_quantized_codes(optimized_codes)
    return motion_hml


def evaluate(cfg):
    """Main evaluation loop."""
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['seed'])

    vqvae, encoder = load_models(cfg, device)

    # Load test dataset
    dataset = Stage2Dataset(cfg, fold='test')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.get('eval_batch_size', 1),
        shuffle=False,
        num_workers=cfg.get('num_workers', 0),
    )

    use_tto = cfg.get('use_tto', False)
    tto_cfg = cfg.get('tto', {})

    mpjpe_no_tto = []
    mpjpe_tto = []

    print(f"\nEvaluating on {len(dataset)} windows | TTO: {use_tto}")

    for imu, motion_gt_hml in tqdm(loader, desc='Evaluating'):
        imu = imu.to(device)
        motion_gt_hml = motion_gt_hml.to(device)

        # GT joint positions
        gt_positions = decode_to_positions(motion_gt_hml)

        # Prediction without TTO
        motion_pred = predict_no_tto(encoder, vqvae, imu)
        pred_positions = decode_to_positions(motion_pred)
        mpjpe_no_tto.append(compute_mpjpe(pred_positions, gt_positions).item())

        # Prediction with TTO
        if use_tto:
            motion_pred_tto = predict_with_tto(encoder, vqvae, imu, tto_cfg)
            pred_positions_tto = decode_to_positions(motion_pred_tto)
            mpjpe_tto.append(compute_mpjpe(pred_positions_tto, gt_positions).item())

    # Results
    avg_no_tto = np.mean(mpjpe_no_tto) * 100  # m → cm
    print(f"\nMPJPE (no TTO): {avg_no_tto:.2f} cm")

    if use_tto:
        avg_tto = np.mean(mpjpe_tto) * 100
        print(f"MPJPE (with TTO): {avg_tto:.2f} cm")
        print(f"Improvement: {avg_no_tto - avg_tto:.2f} cm")

    # Save results
    if cfg.get('save_results'):
        output_dir = Path(cfg['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {
            'mpjpe_no_tto': mpjpe_no_tto,
            'mpjpe_tto': mpjpe_tto if use_tto else [],
        }
        save_path = output_dir / 'results.pt'
        torch.save(results, save_path)
        print(f"Results saved to {save_path}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--tto', action='store_true', help='Enable TTO')
    parser.add_argument('--save', action='store_true', help='Save results')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.tto:
        cfg['use_tto'] = True
    if args.save:
        cfg['save_results'] = True

    evaluate(cfg)

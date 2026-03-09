"""
Training loop for Stage 2: IMU Encoder → VQ-VAE code prediction.

Uses a frozen VQ-VAE to generate training targets (GT motion → code indices)
and a reconstruction pathway (Gumbel-softmax → VQ-VAE decode → MSE loss).

Usage:
    python -m Ego4o.train_stage2 --config configs/ego4o_encoder.yaml [--wandb]
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

from Ego4o.encoder import IMUTransformerEncoder
from Ego4o.vqvae import VQLimbHML
from Ego4o.data_stage2 import get_dataloaders
from utils import load_yaml, set_seed


def load_vqvae(cfg, device):
    """Load and freeze the pretrained VQ-VAE."""
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

    print(f"Loaded frozen VQ-VAE from {cfg['vqvae_checkpoint']}")
    return vqvae


def compute_loss(encoder, vqvae, imu, motion_hml, recon_loss_weight):
    """
    Compute Stage 2 loss.

    Args:
        encoder: IMUTransformerEncoder (trainable)
        vqvae: VQLimbHML (frozen)
        imu: (B, T, 6, 9)
        motion_hml: (B, 263, 1, T)
        recon_loss_weight: weight for reconstruction loss

    Returns:
        loss: scalar
        metrics: dict
    """
    # 1. Target codes from frozen VQ-VAE
    with torch.no_grad():
        target_codes = vqvae.get_code_idx(motion_hml)  # (B, 6, T')

    # 2. Encoder prediction
    pre_codes = encoder(imu)  # (B, 6, Tt, nb_code)

    # 3. Cross-entropy loss: predicted logits vs target code indices
    latent_loss = F.cross_entropy(
        pre_codes.permute(0, 3, 1, 2),  # (B, nb_code, 6, Tt)
        target_codes                      # (B, 6, Tt)
    )

    # 4. Reconstruction loss via Gumbel-softmax → frozen VQ-VAE decode
    codes_gumbel = F.gumbel_softmax(pre_codes, tau=1, eps=1e-10, hard=True, dim=-1)
    # (B, 6, Tt, nb_code) → (B, Tt, nb_code, 6) for get_x_quantized_from_x_ids
    x_quantized = vqvae.get_x_quantized_from_x_ids(
        codes_gumbel.permute(0, 2, 3, 1).contiguous()
    )
    sample = vqvae.forward_decoder_from_quantized_codes(x_quantized)  # (B, 263, 1, T)
    recon_loss = F.mse_loss(sample, motion_hml)

    # 5. Total loss
    loss = latent_loss + recon_loss_weight * recon_loss

    # Code accuracy
    with torch.no_grad():
        pred_codes = pre_codes.argmax(dim=-1)  # (B, 6, Tt)
        accuracy = (pred_codes == target_codes).float().mean().item()

    return loss, {
        'latent': latent_loss.item(),
        'recon': recon_loss.item(),
        'accuracy': accuracy,
    }


@torch.no_grad()
def evaluate(encoder, vqvae, val_loader, device, recon_loss_weight):
    encoder.eval()
    total_loss = 0.0
    total_metrics = {'latent': 0, 'recon': 0, 'accuracy': 0}

    for imu, motion_hml in val_loader:
        imu = imu.to(device)
        motion_hml = motion_hml.to(device)
        loss, metrics = compute_loss(encoder, vqvae, imu, motion_hml, recon_loss_weight)
        total_loss += loss.item()
        for k in total_metrics:
            total_metrics[k] += metrics[k]

    n = len(val_loader)
    return total_loss / n, {k: v / n for k, v in total_metrics.items()}


def train():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['seed'])

    if args.wandb:
        import wandb
        wandb.init(
            project=cfg.get('wandb_project', 'SimplePoser'),
            name=cfg.get('wandb_run_name', None),
            group=cfg.get('wandb_group', None),
            config=cfg,
        )

    print(f"Device: {device}")
    print(f"Epochs: {cfg['num_epochs']} | Batch: {cfg['batch_size']} | LR: {cfg['learning_rate']}")

    # Frozen VQ-VAE
    vqvae = load_vqvae(cfg, device)

    # Trainable encoder
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
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"VQ-VAE params:  {sum(p.numel() for p in vqvae.parameters()):,} (frozen)\n")

    # Data
    train_loader, val_loader = get_dataloaders(cfg, device)

    # Optimizer
    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'])

    recon_loss_weight = cfg.get('recon_loss_weight', 1.0)

    output_dir = Path(cfg['output_dir']) / cfg['wandb_run_name']
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(cfg['num_epochs']):
        encoder.train()
        train_loss = 0.0
        train_metrics = {'latent': 0, 'recon': 0, 'accuracy': 0}

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg["num_epochs"]}')
        for imu, motion_hml in pbar:
            imu = imu.to(device)
            motion_hml = motion_hml.to(device)

            loss, metrics = compute_loss(encoder, vqvae, imu, motion_hml, recon_loss_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            for k in train_metrics:
                train_metrics[k] += metrics[k]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'latent': f'{metrics["latent"]:.4f}',
                'acc': f'{metrics["accuracy"]:.3f}',
            })

        scheduler.step()
        n = len(train_loader)
        train_loss /= n
        train_metrics = {k: v / n for k, v in train_metrics.items()}
        val_loss, val_metrics = evaluate(encoder, vqvae, val_loader, device, recon_loss_weight)

        print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")
        print(f"  Train - latent: {train_metrics['latent']:.4f} "
              f"recon: {train_metrics['recon']:.4f} "
              f"accuracy: {train_metrics['accuracy']:.3f}")
        print(f"  Val   - latent: {val_metrics['latent']:.4f} "
              f"recon: {val_metrics['recon']:.4f} "
              f"accuracy: {val_metrics['accuracy']:.3f}")

        if args.wandb:
            wandb.log({
                'train/loss': train_loss,
                'train/latent': train_metrics['latent'],
                'train/recon': train_metrics['recon'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_loss,
                'val/latent': val_metrics['latent'],
                'val/recon': val_metrics['recon'],
                'val/accuracy': val_metrics['accuracy'],
                'epoch': epoch + 1,
            })

        torch.save({
            'epoch': epoch,
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': cfg
        }, output_dir / 'latest.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), output_dir / 'best.pt')
            print(f"  -> Best model saved!")

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    train()

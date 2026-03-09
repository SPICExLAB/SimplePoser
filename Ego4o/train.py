"""
Training loop for Part-aware VQ-VAE on HumanML3D 263-dim motion.

Usage:
    python -m Ego4o.train --config configs/ego4o_vqvae.yaml [--wandb]
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

from Ego4o import MODEL_REGISTRY
from Ego4o.data import get_dataloaders
from utils import load_yaml, set_seed


def compute_loss(model, x, commit_weight):
    """
    Forward pass + loss computation.

    Args:
        model: VQLimbHML model
        x: (B, 263, 1, T) input motion
        commit_weight: weight for commitment loss

    Returns:
        loss: scalar total loss
        metrics: dict with recon_loss, commit_loss, perplexity
    """
    x_recon, commit_loss, perplexity = model(x)
    recon_loss = F.mse_loss(x_recon, x)
    loss = recon_loss + commit_weight * commit_loss

    return loss, {
        'recon': recon_loss.item(),
        'commit': commit_loss.item(),
        'perplexity': perplexity.item(),
    }


@torch.no_grad()
def evaluate(model, val_loader, device, commit_weight):
    model.eval()
    total_loss = 0.0
    total_metrics = {'recon': 0, 'commit': 0, 'perplexity': 0}

    for x in val_loader:
        x = x.to(device)
        loss, metrics = compute_loss(model, x, commit_weight)
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

    # Model
    model = MODEL_REGISTRY['vqvae_limb_hml'](
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
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Data
    train_loader, val_loader = get_dataloaders(cfg, device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'])

    commit_weight = cfg.get('commit_weight', 0.001)

    output_dir = Path(cfg['output_dir']) / cfg['wandb_run_name']
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(cfg['num_epochs']):
        model.train()
        train_loss = 0.0
        train_metrics = {'recon': 0, 'commit': 0, 'perplexity': 0}

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg["num_epochs"]}')
        for x in pbar:
            x = x.to(device)
            loss, metrics = compute_loss(model, x, commit_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            for k in train_metrics:
                train_metrics[k] += metrics[k]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{metrics["recon"]:.4f}',
                'ppl': f'{metrics["perplexity"]:.1f}',
            })

        scheduler.step()
        n = len(train_loader)
        train_loss /= n
        train_metrics = {k: v / n for k, v in train_metrics.items()}
        val_loss, val_metrics = evaluate(model, val_loader, device, commit_weight)

        print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")
        print(f"  Train - recon: {train_metrics['recon']:.4f} "
              f"commit: {train_metrics['commit']:.4f} "
              f"perplexity: {train_metrics['perplexity']:.1f}")
        print(f"  Val   - recon: {val_metrics['recon']:.4f} "
              f"commit: {val_metrics['commit']:.4f} "
              f"perplexity: {val_metrics['perplexity']:.1f}")

        if args.wandb:
            wandb.log({
                'train/loss': train_loss,
                'train/recon': train_metrics['recon'],
                'train/commit': train_metrics['commit'],
                'train/perplexity': train_metrics['perplexity'],
                'val/loss': val_loss,
                'val/recon': val_metrics['recon'],
                'val/commit': val_metrics['commit'],
                'val/perplexity': val_metrics['perplexity'],
                'epoch': epoch + 1,
            })

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': cfg
        }, output_dir / 'latest.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best.pt')
            print(f"  -> Best model saved!")

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    train()

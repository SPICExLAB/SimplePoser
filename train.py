from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from data import get_dataloaders
from utils import load_yaml, set_seed
from mobileposer import MobilePoser
from dynaip import DynaIP


MODEL_REGISTRY = {
    'mobileposer': MobilePoser,
    'dynaip': DynaIP,
}


@torch.no_grad()
def evaluate(model, val_loader, device, finetune=False):
    model.eval()
    total_loss = 0.0
    total_metrics = defaultdict(float)
    for batch in tqdm(val_loader, desc='Validating', leave=False):
        loss, metrics = model.compute_val_loss(batch, device, finetune=finetune)
        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] += v
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
            project=cfg.get('wandb_project', 'simpleposer'),
            name=cfg.get('wandb_run_name'),
            group=cfg.get('wandb_group'),
            tags=cfg.get('wandb_tags', []),
            config=cfg,
        )

    print(f"Device: {device}")
    print(f"Model: {cfg.get('model', 'dynaip')}")
    print(f"Epochs: {cfg['num_epochs']} | Batch: {cfg['batch_size']} | LR: {cfg['learning_rate']}")

    ModelClass = MODEL_REGISTRY[cfg.get('model', 'dynaip')]
    model = ModelClass(cfg).to(device)

    finetune = cfg.get('pretrained') is not None
    if finetune:
        model.load_state_dict(torch.load(cfg['pretrained'], map_location=device, weights_only=True))
        print(f"Loaded pretrained: {cfg['pretrained']}")

    train_loader, val_loader = get_dataloaders(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg['num_epochs'] // 2))

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    output_dir = Path(cfg['output_dir']) / cfg['wandb_run_name']
    if finetune:
        output_dir = output_dir / 'finetune'
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(cfg['num_epochs']):
        model.train()
        train_loss = 0.0
        train_metrics = defaultdict(float)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg["num_epochs"]}')
        for batch in pbar:
            loss, metrics = model.compute_loss(batch, device, finetune=finetune)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            for k, v in metrics.items():
                train_metrics[k] += v
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        n = len(train_loader)
        train_loss /= n
        train_metrics = {k: v / n for k, v in train_metrics.items()}
        val_loss, val_metrics = evaluate(model, val_loader, device, finetune=finetune)

        print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")
        print(f"  Train: {', '.join(f'{k} {v:.4f}' for k, v in train_metrics.items())}")
        print(f"  Val:   {', '.join(f'{k} {v:.4f}' for k, v in val_metrics.items())}")

        if args.wandb:
            log = {'train/loss': train_loss, 'val/loss': val_loss, 'epoch': epoch + 1}
            for k, v in train_metrics.items():
                log[f'train/{k}'] = v
            for k, v in val_metrics.items():
                log[f'val/{k}'] = v
            wandb.log(log)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': cfg,
        }, output_dir / 'latest.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best.pt')
            print(f"  -> Best model saved!")

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    train()

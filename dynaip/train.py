import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

from dynaip import MODEL_REGISTRY
from dynaip.utils import r6d_to_local
from data import get_dataloaders
from utils import load_yaml, set_seed
from config import joint_set


def pred_joints_fk(model, pred_pose):
    """Convert predicted 6D pose to joint positions via FK."""
    local = r6d_to_local(pred_pose, model.global_to_local_pose)
    _, joints = model.forward_kinematics(local)
    return joints


def compute_loss(model, imu, pose_6d, joints, vel, stationary, root_vel, device):
    """Compute combined pose + translation loss."""
    B, T = pose_6d.shape[:2]

    gt_pose = pose_6d.view(B, T, 24, 6)[:, :, joint_set.reduced].view(B, T, -1)
    gt_vel = vel.view(B, T, 24, 3)[:, :, model.vel_indices].flatten(2)
    gt_joints = joints.view(B, T, 24, 3)

    pred_vel, pred_pose, pred_root_vel, pred_stat = model(imu)
    pred_j = pred_joints_fk(model, pred_pose).view(B, T, 24, 3)

    # pose losses
    pose_loss = F.mse_loss(pred_vel, gt_vel) + F.mse_loss(pred_pose, gt_pose)
    joint_loss = F.mse_loss(pred_j, gt_joints)

    # translation losses
    root_vel_loss = F.mse_loss(pred_root_vel, root_vel)
    stat_loss = F.binary_cross_entropy_with_logits(pred_stat, stationary)

    loss = pose_loss + joint_loss + root_vel_loss + stat_loss

    return loss, {
        'pose': pose_loss.item(),
        'joint': joint_loss.item(),
        'root_vel': root_vel_loss.item(),
        'stat': stat_loss.item(),
    }


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_metrics = {'pose': 0, 'joint': 0, 'root_vel': 0, 'stat': 0}

    for imu, pose_6d, joints, tran, vel, contact, stationary, root_vel, lengths in val_loader:
        imu = imu.to(device)
        pose_6d = pose_6d.to(device)
        vel = vel.to(device)
        joints = joints.to(device)
        stationary = stationary.to(device)
        root_vel = root_vel.to(device)

        loss, metrics = compute_loss(model, imu, pose_6d, joints, vel, stationary, root_vel, device)
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
            project=cfg.get('wandb_project', 'simpleposer'),
            name=cfg.get('wandb_run_name', None),
            group=cfg.get('wandb_group', None),
            tags=cfg.get('wandb_tags', []),
            config=cfg,
        )

    print(f"Device: {device}")
    print(f"Epochs: {cfg['num_epochs']} | Batch: {cfg['batch_size']} | LR: {cfg['learning_rate']}")

    ModelClass = MODEL_REGISTRY[cfg.get('model', 'dynaip')]
    model = ModelClass(device=device).to(device)

    finetune = cfg.get('pretrained') is not None
    if finetune:
        model.load_state_dict(torch.load(cfg['pretrained'], map_location=device, weights_only=True))
        print(f"Loaded pretrained: {cfg['pretrained']}")

    train_loader, val_loader = get_dataloaders(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'] // 2)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    output_dir = Path(cfg['output_dir']) / cfg['wandb_run_name']
    if finetune:
        output_dir = output_dir / 'finetune'
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(cfg['num_epochs']):
        model.train()
        train_loss = 0.0
        train_metrics = {'pose': 0, 'joint': 0, 'root_vel': 0, 'stat': 0}

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg["num_epochs"]}')
        for imu, pose_6d, joints, tran, vel, contact, stationary, root_vel, lengths in pbar:
            imu = imu.to(device)
            pose_6d = pose_6d.to(device)
            vel = vel.to(device)
            joints = joints.to(device)
            stationary = stationary.to(device)
            root_vel = root_vel.to(device)

            loss, metrics = compute_loss(model, imu, pose_6d, joints, vel, stationary, root_vel, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            for k in train_metrics:
                train_metrics[k] += metrics[k]
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'stat': f'{metrics["stat"]:.4f}'})

        scheduler.step()
        n = len(train_loader)
        train_loss /= n
        train_metrics = {k: v / n for k, v in train_metrics.items()}
        val_loss, val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")
        print(f"  Train - pose: {train_metrics['pose']:.4f} joint: {train_metrics['joint']:.4f} "
              f"root_vel: {train_metrics['root_vel']:.4f} stat: {train_metrics['stat']:.4f}")
        print(f"  Val   - pose: {val_metrics['pose']:.4f} joint: {val_metrics['joint']:.4f} "
              f"root_vel: {val_metrics['root_vel']:.4f} stat: {val_metrics['stat']:.4f}")

        if args.wandb:
            wandb.log({
                'train/loss': train_loss,
                'train/pose': train_metrics['pose'],
                'train/joint': train_metrics['joint'],
                'train/root_vel': train_metrics['root_vel'],
                'train/stat': train_metrics['stat'],
                'val/loss': val_loss,
                'val/pose': val_metrics['pose'],
                'val/joint': val_metrics['joint'],
                'val/root_vel': val_metrics['root_vel'],
                'val/stat': val_metrics['stat'],
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
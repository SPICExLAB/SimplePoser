import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

from model import MobilePoser
from data import get_dataloaders
from utils import load_yaml, set_seed
from loss import compute_vel_loss, compute_jerk_loss
from config import joint_set


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    
    for imu, pose_6d, joints, tran, vel, contact, lengths in tqdm(val_loader, desc='Validating'):
        imu = imu.to(device)
        pose_6d = pose_6d.to(device)
        joints = joints.to(device)
        vel = vel.to(device)
        contact = contact.to(device)
        
        B, T = pose_6d.shape[:2]
        pose_6d = pose_6d.view(B, T, 24, 6)[:, :, joint_set.reduced].view(B, T, -1)
        
        # forward the model
        pred_pose, pred_joints, pred_vel, pred_contact = model(imu)
        
        # compute loss
        joints_loss = nn.MSELoss()(pred_joints, joints.view(B, T, -1))
        pose_loss = nn.MSELoss()(pred_pose, pose_6d)
        contact_loss = nn.BCEWithLogitsLoss()(pred_contact, contact)
        vel_loss = sum(compute_vel_loss(pred_vel, vel.view(B, T, -1), i) for i in [1, 3, 9])
        
        loss = joints_loss + pose_loss + contact_loss + 0.2 * vel_loss
        total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['seed'])
    output_dir = Path(cfg['output_dir']) / Path(cfg['wandb_run_name'])
    output_dir.mkdir(exist_ok=True)

    # setup wandb
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

    # init model and data
    model = MobilePoser(cfg).to(device)
    train_loader, val_loader = get_dataloaders(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    best_val_loss = float('inf')

    for epoch in range(cfg['num_epochs']):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg["num_epochs"]}')
        for imu, pose_6d, joints, tran, vel, contact, lengths in pbar:
            # move everything to device
            imu = imu.to(device)         # [B, T, 60]
            joints = joints.to(device)   # [B, T, 24, 3]
            vel = vel.to(device)         # [B, T, 24, 3]
            contact = contact.to(device) # [B, T, 2]
            pose_6d = pose_6d.to(device) # [B, T, 24*6]
            
            B, T = pose_6d.shape[:2]
            pose_6d = pose_6d.view(B, T, 24, 6)[:, :, joint_set.reduced].view(B, T, -1)
            
            # add noise to GT joints for downstream modules
            pose_noise = torch.randn_like(joints) * 0.04
            contact_noise = torch.randn_like(joints) * 0.04
            vel_noise = torch.randn_like(joints) * 0.025
            
            noisy_joints_pose = (joints + pose_noise).view(B, T, -1)
            noisy_joints_contact = (joints + contact_noise).view(B, T, -1)
            noisy_joints_vel = (joints + vel_noise).view(B, T, -1)
            
            # train each module independently like MobilePoser 
            pred_joints = model.joints(imu)
            pred_pose = model.pose(torch.cat([noisy_joints_pose, imu], dim=-1))
            pred_contact = model.foot_contact(torch.cat([noisy_joints_contact, imu], dim=-1))
            pred_vel = model.velocity(torch.cat([noisy_joints_vel, imu], dim=-1))
            
            # compute losses
            joints_loss = nn.MSELoss()(pred_joints, joints.view(B, T, -1))
            pose_loss = nn.MSELoss()(pred_pose, pose_6d)
            contact_loss = nn.BCEWithLogitsLoss()(pred_contact, contact)
            vel_loss = sum(compute_vel_loss(pred_vel, vel.view(B, T, -1), i) for i in [1, 3, 9])
            pose_jerk_loss = compute_jerk_loss(pred_pose)
            joints_jerk_loss = compute_jerk_loss(pred_joints)

            # compute total loss
            loss = 0.0
            loss += joints_loss
            loss += pose_loss
            loss += contact_loss
            loss += 0.5 * vel_loss
            loss += 1e-5 * pose_jerk_loss
            loss += 1e-5 * joints_jerk_loss

            # backward 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")

        if args.wandb:
            wandb.log({
                'train/loss': train_loss,
                'val/loss': val_loss,
                'epoch': epoch+1,
            })
        
        # save checkpoint
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
            print(f"  → Best model saved!")
    
    print(f"\nDone! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    train()
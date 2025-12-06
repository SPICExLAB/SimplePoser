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
def evaluate(model, val_loader, device, finetune=False):
    model.eval()
    total_loss = 0.0
    
    for imu, pose_6d, joints, tran, vel, contact, lengths in tqdm(val_loader, desc='Validating'):
        imu = imu.to(device)
        pose_6d = pose_6d.to(device)
        joints = joints.to(device)
        
        B, T = pose_6d.shape[:2]
        pose_6d = pose_6d.view(B, T, 24, 6)[:, :, joint_set.reduced].view(B, T, -1)
        
        pred_pose, pred_joints, pred_vel, pred_contact = model(imu)
        
        joints_loss = nn.MSELoss()(pred_joints, joints.view(B, T, -1))
        pose_loss = nn.MSELoss()(pred_pose, pose_6d)
        loss = joints_loss + pose_loss
        
        if not finetune:
            vel = vel.to(device)
            contact = contact.to(device)
            loss += nn.BCEWithLogitsLoss()(pred_contact, contact)
            loss += 0.2 * sum(compute_vel_loss(pred_vel, vel.view(B, T, -1), i) for i in [1, 3, 9])
        
        total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    # load config
    cfg = load_yaml(args.config)
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['seed'])

    # load semoae model
    if cfg['use_semoae']:
        from semoae.semoae import SemoAE
        print("Loading SemoAE for IMU augmentation")
        semo = SemoAE(feat_dim=45, encode_dim=32).to(device)
        semo_ckpt = torch.load(cfg['semoae_ckpt'], map_location=device, weights_only=True)
        semo.load_state_dict(semo_ckpt["model_state_dict"])
        semo.eval()
        print("  → SemoAE loaded.\n")

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

    finetune = cfg.get('pretrained') is not None
    if finetune:
        # load pretrained model
        model.load_state_dict(torch.load(cfg['pretrained'], map_location=device, weights_only=True))
        print(f"Loaded pretrained: {cfg['pretrained']}")
        output_dir = Path(cfg['output_dir']) / Path(cfg['wandb_run_name']) / "finetune"
    else:
        output_dir = Path(cfg['output_dir']) / Path(cfg['wandb_run_name'])

    # create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

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
            imu = imu.to(device)         # [B, T, 60/45]
            joints = joints.to(device)   # [B, T, 24, 3]
            vel = vel.to(device)         # [B, T, 24, 3]
            contact = contact.to(device) # [B, T, 2]
            pose_6d = pose_6d.to(device) # [B, T, 24*6]
            
            B, T = pose_6d.shape[:2]
            pose_6d = pose_6d.view(B, T, 24, 6)[:, :, joint_set.reduced].view(B, T, -1)

            # add secondary motion to IMU
            if cfg['use_semoae'] and torch.rand(1) < cfg['semo_prob']:
                imu = semo.add_secondary_motion(imu, cfg['semo_eta']) # [B, T, 60/45]
            
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

            # compute losses
            joints_loss = nn.MSELoss()(pred_joints, joints.view(B, T, -1))
            pose_loss = nn.MSELoss()(pred_pose, pose_6d)
            pose_jerk_loss = compute_jerk_loss(pred_pose)
            joints_jerk_loss = compute_jerk_loss(pred_joints)
            loss = joints_loss + pose_loss + 1e-5 * (pose_jerk_loss + joints_jerk_loss)
            
            # compute contact and velocity losses
            if not finetune:
                pred_contact = model.foot_contact(torch.cat([noisy_joints_contact, imu], dim=-1))
                pred_vel = model.velocity(torch.cat([noisy_joints_vel, imu], dim=-1))
                contact_loss = nn.BCEWithLogitsLoss()(pred_contact, contact)
                vel_loss = sum(compute_vel_loss(pred_vel, vel.view(B, T, -1), i) for i in [1, 3, 9])
                loss += contact_loss
                loss += 0.5 * vel_loss

            # backward 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        val_loss = evaluate(model, val_loader, device, finetune)
        
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
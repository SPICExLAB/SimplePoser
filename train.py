import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser

from model import MobilePoser
from data import get_dataloaders
from utils import load_yaml, set_seed
from config import joint_set
from loss import compute_loss



@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_pose_loss = 0.0
    total_joints_loss = 0.0
    total_vel_loss = 0.0
    total_contact_loss = 0.0
    
    for batch in tqdm(val_loader, desc='Validating'):
        # unpack batch
        imu, pose_6d, joints, tran, vel, contact, lengths = batch
        
        # move to device
        imu = imu.to(device)
        pose_6d = pose_6d.to(device)
        joints = joints.to(device)
        vel = vel.to(device)
        contact = contact.to(device)
        lengths = lengths.to(device)

        B, T, _ = pose_6d.shape
        pose_6d = pose_6d.view(B, T, 24, 6)[:, :, joint_set.reduced].view(B, T, -1)
        
        # forward pass
        pred_pose, pred_joints, pred_vel, pred_contact = model(imu)
        
        # compute loss
        loss, loss_components = compute_loss(
            pred_pose, pred_joints, pred_vel, pred_contact,
            pose_6d, joints, vel, contact
        )
        
        # accumulate losses
        total_loss += loss.item()
        total_pose_loss += loss_components['pose']
        total_joints_loss += loss_components['joints']
        total_vel_loss += loss_components['velocity']
        total_contact_loss += loss_components['contact']
    
    # average over batches
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_components = {
        'pose': total_pose_loss / num_batches,
        'joints': total_joints_loss / num_batches,
        'velocity': total_vel_loss / num_batches,
        'contact': total_contact_loss / num_batches
    }
    
    return avg_loss, avg_components


def train():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # load config
    cfg = load_yaml(args.config)

    # setup
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['seed'])
    output_dir = Path('checkpoints')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Training for {cfg['num_epochs']} epochs")
    print(f"Batch size: {cfg['batch_size']} | LR: {cfg['learning_rate']}")    

    # model
    model = MobilePoser(cfg).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # data
    train_loader, val_loader = get_dataloaders(cfg, device)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    
    # training loop
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

            B, T, _ = pose_6d.shape
            pose_6d = pose_6d.view(B, T, 24, 6)[:, :, joint_set.reduced].view(B, T, -1)
            
            # forward
            pred_pose, pred_joints, pred_vel, pred_contact = model(imu)
            loss, _ = compute_loss(
                pred_pose, pred_joints, pred_vel, pred_contact,
                pose_6d, joints, vel, contact
            )

            # backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # validate
        val_loss, val_comp = evaluate(model, val_loader, device)
        
        # log
        print(f"\nEpoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")
        print(f"  Pose {val_comp['pose']:.4f} | Joints {val_comp['joints']:.4f} | "
              f"Vel {val_comp['velocity']:.4f} | Contact {val_comp['contact']:.4f}")
        
        # save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': cfg
        }
        torch.save(checkpoint, output_dir / 'latest.pt')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'best.pt')
            print(f"  → Saved best model!")
    
    print(f"\nDone! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    train()
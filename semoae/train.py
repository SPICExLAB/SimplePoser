import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import torch.nn as nn
from semoae.dataset import SynPairedIMUData
from semoae.semoae import SemoAE
from utils import load_yaml, set_seed

@torch.no_grad()
def evaluate(model, test_loader, device, criterion, d_loss_weight):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_dis = 0.0
    for syn_imu, real_imu in tqdm(test_loader, desc='Validating'):
        syn_imu = syn_imu.to(device)
        real_imu = real_imu.to(device)
        
        # encode data
        syn_encoded = model.encoder(syn_imu)
        real_encoded = model.encoder(real_imu)
        loss_distribution = model.dis_normalizer(syn_encoded, real_encoded)
        
        # decode data
        syn_decoded = model.decoder(syn_encoded)
        real_decoded = model.decoder(real_encoded)
        loss_syn = criterion(syn_decoded, syn_imu)
        loss_real = criterion(real_decoded, real_imu)
        
        # compute total loss
        loss_recon = loss_syn + loss_real
        loss = loss_recon + loss_distribution * d_loss_weight
        
        total_loss += loss.item()
        total_recon += loss_recon.item()
        total_dis += loss_distribution.item()
    return total_loss / len(test_loader), total_recon / len(test_loader), total_dis / len(test_loader)

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['seed'])
    
    data_train_path = '/data/projects/Pose/dataset_work/IMUPoser/train.pt'
    data_test_path = '/data/projects/Pose/dataset_work/IMUPoser/test.pt'
    
    train_dataset = SynPairedIMUData(data_train_path)
    test_dataset = SynPairedIMUData(data_test_path)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    model = SemoAE(feat_dim=45, encode_dim=32).to(device)  # 45 = 5*3 + 5*6
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.MSELoss()
    best_loss = float('inf')
    
    for epoch in range(cfg['num_epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg["num_epochs"]}')
        for syn_imu, real_imu in pbar:
            syn_imu = syn_imu.to(device)
            real_imu = real_imu.to(device)
            
            # encode data
            syn_encoded = model.encoder(syn_imu)
            real_encoded = model.encoder(real_imu)
            loss_distribution = model.dis_normalizer(syn_encoded, real_encoded)
            
            # decode data
            syn_decoded = model.decoder(syn_encoded)
            real_decoded = model.decoder(real_encoded)
            loss_syn = criterion(syn_decoded, syn_imu)
            loss_real = criterion(real_decoded, real_imu)
            
            # compute total loss
            loss_recon = loss_syn + loss_real
            loss = loss_recon + loss_distribution * cfg['d_loss_weight']
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{loss_recon.item():.4f}',
                'dis': f'{loss_distribution.item():.4f}'
            })
        
        # evaluate
        test_loss, test_recon, test_dis = evaluate(model, test_loader, device, criterion, cfg['d_loss_weight'])
        print(f'\nEpoch {epoch+1}: Test loss = {test_loss:.4f} (recon: {test_recon:.4f}, dis: {test_dis:.4f})')
        
        # save checkpoint
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
            }, 'semoae/checkpoints/semoae_best.pt')
            print(f'New best saved!')

if __name__ == '__main__':
    main()
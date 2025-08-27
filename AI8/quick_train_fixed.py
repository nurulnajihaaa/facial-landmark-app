"""
Quick training script for the fixed HRNet model
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_hrnet_facial import HRNetConfig, FacialKeypointsDataset, JointsMSELoss
from fixed_hrnet import FixedHRNet
from pathlib import Path

def quick_train():
    config = HRNetConfig()
    config.batch_size = 8
    config.epochs = 20
    config.learning_rate = 0.001
    
    # Create model
    model = FixedHRNet(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Training on {device}")
    
    # Dataset
    train_dataset = FacialKeypointsDataset('coco_annotations.json', 'dataset', config, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    
    # Loss and optimizer
    criterion = JointsMSELoss(use_target_weight=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    print(f"Training with {len(train_dataset)} samples")
    
    # Create checkpoint directory
    Path('checkpoints').mkdir(exist_ok=True)
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            heatmaps = batch['heatmaps'].to(device)
            target_weight = batch['target_weight'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Check output size
            if outputs.shape[2:] != heatmaps.shape[2:]:
                print(f"Size mismatch: {outputs.shape} vs {heatmaps.shape}")
                break
                
            loss = criterion(outputs, heatmaps, target_weight)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}/{config.epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}: Average Loss = {avg_loss:.6f}')
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == config.epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }
            torch.save(checkpoint, f'checkpoints/fixed_hrnet_epoch_{epoch}.pth')
            print(f'Saved checkpoint: fixed_hrnet_epoch_{epoch}.pth')

if __name__ == '__main__':
    quick_train()

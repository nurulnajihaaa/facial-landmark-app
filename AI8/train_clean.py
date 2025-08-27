"""
Quick training script for clean data
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append('.')

from train_hrnet_facial import FacialKeypointsDataset, HRNetConfig
from fixed_hrnet import FixedHRNet
import numpy as np
from pathlib import Path

def train_on_clean_data():
    print("=== TRAINING ON CLEAN DATA ===")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    config = HRNetConfig()
    config.batch_size = 8  # Smaller batch size due to less data
    config.epochs = 30  # More epochs since less data
    
    # Create datasets with clean data
    print("Loading clean training data...")
    train_dataset = FacialKeypointsDataset(
        coco_file='coco_annotations_clean.json',
        image_root='dataset',
        config=config,
        is_train=True
    )
    
    val_dataset = FacialKeypointsDataset(
        coco_file='coco_annotations_clean.json', 
        image_root='dataset',
        config=config,
        is_train=False
    )
    
    print(f"Clean dataset size: {len(train_dataset)}")
    
    # Split into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_subset)}")
    print(f"Val samples: {len(val_subset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = FixedHRNet(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            target_heatmaps = batch['heatmaps'].to(device)
            target_weights = batch['target_weight'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_heatmaps = model(images)
            
            # Calculate loss (weighted by target_weights)
            loss = 0
            for i in range(config.num_keypoints):
                weight = target_weights[:, i, 0]  # [batch_size]
                pred_map = pred_heatmaps[:, i]     # [batch_size, H, W]
                target_map = target_heatmaps[:, i] # [batch_size, H, W]
                
                # Weighted MSE loss
                mse = nn.functional.mse_loss(pred_map, target_map, reduction='none')
                weighted_mse = (mse.view(mse.shape[0], -1).mean(dim=1) * weight).mean()
                loss += weighted_mse
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                target_heatmaps = batch['heatmaps'].to(device)
                target_weights = batch['target_weight'].to(device)
                
                pred_heatmaps = model(images)
                
                # Calculate validation loss
                loss = 0
                for i in range(config.num_keypoints):
                    weight = target_weights[:, i, 0]
                    pred_map = pred_heatmaps[:, i]
                    target_map = target_heatmaps[:, i]
                    
                    mse = nn.functional.mse_loss(pred_map, target_map, reduction='none')
                    weighted_mse = (mse.view(mse.shape[0], -1).mean(dim=1) * weight).mean()
                    loss += weighted_mse
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:2d}/{config.epochs}: "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_path = checkpoint_dir / "clean_fixed_hrnet_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, checkpoint_path)
            
            print(f"✅ Saved best model: {checkpoint_path}")
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    train_on_clean_data()

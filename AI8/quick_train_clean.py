"""
Quick training script that saves weights only
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.append('.')
from train_hrnet_facial import HRNetConfig, FacialKeypointsDataset
from proper_hrnet import ProperHRNet, HRNetLoss

class QuickConfig(HRNetConfig):
    """Quick config for training"""
    def __init__(self):
        super().__init__()
        self.input_size = [256, 256]
        self.output_size = [64, 64]
        self.batch_size = 8
        self.epochs = 15  # Quick training
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.sigma = 2.0
        self.num_keypoints = 14

def quick_train():
    """Quick training to generate a clean checkpoint"""
    print("=== QUICK TRAINING FOR CLEAN CHECKPOINT ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    config = QuickConfig()
    
    # Load dataset
    dataset = FacialKeypointsDataset(
        coco_file='coco_annotations_clean.json',
        image_root='dataset',
        config=config,
        is_train=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_subset, val_subset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # Model
    model = ProperHRNet(config).to(device)
    criterion = HRNetLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
            pred_heatmaps = model(images)
            loss = criterion(pred_heatmaps, target_heatmaps, target_weights)
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
                loss = criterion(pred_heatmaps, target_heatmaps, target_weights)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:2d}/{config.epochs}: "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}")
        
        # Save best model (weights only)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Save only weights and basic info
            weights_path = checkpoint_dir / "hrnet_weights_only.pth"
            torch.save(model.state_dict(), weights_path)
            
            # Save metadata separately
            metadata = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'model_params': sum(p.numel() for p in model.parameters())
            }
            metadata_path = checkpoint_dir / "hrnet_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Saved best model: {weights_path}")
    
    print(f"\\n🎉 Quick training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    return best_val_loss

if __name__ == "__main__":
    quick_train()

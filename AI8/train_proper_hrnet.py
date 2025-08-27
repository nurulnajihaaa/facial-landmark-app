"""
Training script for proper HRNet facial keypoint detection
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from torchvision import transforms
import json

sys.path.append('.')
from train_hrnet_facial import HRNetConfig, FacialKeypointsDataset
from proper_hrnet import ProperHRNet, HRNetLoss

class EnhancedHRNetConfig(HRNetConfig):
    """Enhanced config for proper HRNet"""
    def __init__(self):
        super().__init__()
        # Adjust for proper HRNet
        self.input_size = [256, 256]
        self.output_size = [64, 64]  # HRNet native output size
        self.batch_size = 8
        self.epochs = 40
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.sigma = 2.0  # Gaussian sigma for heatmaps
        
        # Data augmentation settings
        self.color_factor = 0.15
        self.rotation_factor = 30
        self.scale_factor = 0.25

class RobustDataset(FacialKeypointsDataset):
    """Enhanced dataset with better preprocessing for HRNet"""
    
    def __init__(self, coco_file, image_root, config, is_train=True):
        super().__init__(coco_file, image_root, config, is_train)
        
        # Enhanced transforms for training
        if is_train:
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(
                    brightness=config.color_factor,
                    contrast=config.color_factor,
                    saturation=config.color_factor,
                    hue=0.1
                ),
                transforms.RandomRotation(5),  # Small rotation
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def generate_heatmap(self, keypoints, output_size):
        """Generate high-quality heatmaps for HRNet"""
        heatmaps = np.zeros((self.config.num_keypoints, output_size[1], output_size[0]))
        target_weight = np.ones((self.config.num_keypoints, 1))
        
        for i in range(self.config.num_keypoints):
            x = keypoints[i * 3]
            y = keypoints[i * 3 + 1]
            v = keypoints[i * 3 + 2]
            
            if v == 0:
                target_weight[i] = 0
                continue
            
            # Scale to output size with proper aspect ratio
            x_scaled = x * output_size[0] / self.config.input_size[0]
            y_scaled = y * output_size[1] / self.config.input_size[1]
            
            if x_scaled >= 0 and y_scaled >= 0 and x_scaled < output_size[0] and y_scaled < output_size[1]:
                # Generate Gaussian heatmap with proper sigma
                sigma = self.config.sigma
                tmp_size = sigma * 3
                
                # Generate gaussian
                size = 2 * tmp_size + 1
                x_range = np.arange(0, size, 1, np.float32)
                y_range = x_range[:, None]
                x0 = y0 = size // 2
                
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x_range - x0) ** 2 + (y_range - y0) ** 2) / (2 * sigma ** 2))
                
                # Usable gaussian range
                g_x = max(0, -int(x_scaled) + int(tmp_size))
                g_y = max(0, -int(y_scaled) + int(tmp_size))
                g_x_max = min(int(size), output_size[0] - int(x_scaled) + int(tmp_size))
                g_y_max = min(int(size), output_size[1] - int(y_scaled) + int(tmp_size))
                
                # Image range
                img_x = max(0, int(x_scaled) - int(tmp_size))
                img_y = max(0, int(y_scaled) - int(tmp_size))
                img_x_max = min(output_size[0], int(x_scaled) + int(tmp_size) + 1)
                img_y_max = min(output_size[1], int(y_scaled) + int(tmp_size) + 1)
                
                if img_x < img_x_max and img_y < img_y_max:
                    heatmaps[i, img_y:img_y_max, img_x:img_x_max] = \
                        g[g_y:g_y_max, g_x:g_x_max]
        
        return heatmaps, target_weight

def validate_training_data():
    """Validate that training data makes sense"""
    print("=== VALIDATING TRAINING DATA ===")
    
    config = EnhancedHRNetConfig()
    dataset = RobustDataset(
        coco_file='coco_annotations_clean.json',
        image_root='dataset',
        config=config,
        is_train=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check a few samples for sanity
    for i in range(min(3, len(dataset))):
        try:
            sample = dataset[i]
            keypoints = sample['keypoints']
            heatmaps = sample['heatmaps']
            
            print(f"\nSample {i+1}:")
            print(f"  Keypoints range: X=[{keypoints[::3].min():.1f}, {keypoints[::3].max():.1f}], "
                  f"Y=[{keypoints[1::3].min():.1f}, {keypoints[1::3].max():.1f}]")
            print(f"  Heatmap peaks: {[float(heatmaps[j].max()) for j in range(min(5, config.num_keypoints))]}")
            
            # Check if keypoints are reasonable
            x_coords = keypoints[::3]
            y_coords = keypoints[1::3]
            
            if (x_coords.min() >= 0 and x_coords.max() <= config.input_size[0] and
                y_coords.min() >= 0 and y_coords.max() <= config.input_size[1]):
                print(f"  ✅ Keypoints within image bounds")
            else:
                print(f"  ❌ Keypoints out of bounds!")
                
        except Exception as e:
            print(f"  ❌ Error in sample {i}: {e}")
    
    return dataset

def train_proper_hrnet():
    print("=== TRAINING PROPER HRNET ===")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Enhanced config
    config = EnhancedHRNetConfig()
    print(f"Config: {config.input_size} -> {config.output_size}, {config.num_keypoints} keypoints")
    
    # Validate and load data
    dataset = validate_training_data()
    
    # Create train/val split
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_subset)}")
    print(f"Val samples: {len(val_subset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # Initialize proper HRNet
    model = ProperHRNet(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Proper loss and optimizer
    criterion = HRNetLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Training tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            target_heatmaps = batch['heatmaps'].to(device)
            target_weights = batch['target_weight'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_heatmaps = model(images)
            
            # Calculate loss
            loss = criterion(pred_heatmaps, target_heatmaps, target_weights)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Progress logging
            if batch_idx % 2 == 0:
                print(f"Epoch {epoch+1:2d}/{config.epochs}, "
                      f"Batch {batch_idx:2d}/{len(train_loader)}, "
                      f"Loss: {loss.item():.6f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
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
        
        # Update learning rate
        scheduler.step()
        
        # Track losses
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1:2d}/{config.epochs}: "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_path = checkpoint_dir / "proper_hrnet_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, checkpoint_path)
            
            print(f"✅ Saved best model: {checkpoint_path}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"proper_hrnet_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, checkpoint_path)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training Progress (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    output_dir = Path("outputs/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "proper_hrnet_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Training curves saved to: {output_dir / 'proper_hrnet_training.png'}")

if __name__ == "__main__":
    train_proper_hrnet()

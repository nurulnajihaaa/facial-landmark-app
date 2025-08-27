"""
Improved training script with better data augmentation and training strategies
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append('.')
from train_hrnet_facial import HRNetConfig, FacialKeypointsDataset
from improved_hrnet import ImprovedHRNet, ImprovedFocalLoss

class ImprovedDataset(FacialKeypointsDataset):
    """Enhanced dataset with better augmentation"""
    
    def __init__(self, coco_file, image_root, config, is_train=True):
        super().__init__(coco_file, image_root, config, is_train)
        
        # Better augmentation pipeline
        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info['id']
        
        # Load image
        image_file = image_info['file_name']
        image_path = self.image_root / image_file
        
        # Alternative paths
        if not image_path.exists():
            image_name = Path(image_file).name
            for split in ['train', 'valid', 'test']:
                alt_path = self.image_root / split / 'images' / image_name
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        if not image_path.exists():
            raise FileNotFoundError(f"Could not find image: {image_path}")
            
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotation
        ann = self.annotations[image_id][0]
        keypoints = ann['keypoints']
        bbox = ann['bbox']
        
        # Enhanced crop and resize with padding
        x, y, w, h = bbox
        
        # Add more aggressive padding for better context
        padding = 0.4  # Increased from 0.3
        x = max(0, x - w * padding / 2)
        y = max(0, y - h * padding / 2)
        w = w * (1 + padding)
        h = h * (1 + padding)
        
        # Make crop square for better aspect ratio handling
        size = max(w, h)
        center_x = x + w / 2
        center_y = y + h / 2
        x = center_x - size / 2
        y = center_y - size / 2
        w = h = size
        
        # Ensure within image bounds
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        # Crop image
        cropped_image = image[int(y):int(y+h), int(x):int(x+w)]
        
        # Adjust keypoints relative to crop
        adjusted_keypoints = []
        keypoint_coords = []
        for i in range(0, len(keypoints), 3):
            kp_x = keypoints[i] - x
            kp_y = keypoints[i+1] - y
            kp_v = keypoints[i+2]
            adjusted_keypoints.extend([kp_x, kp_y, kp_v])
            if kp_v > 0:  # Only add visible keypoints for augmentation
                keypoint_coords.append((kp_x, kp_y))
        
        # Resize to input size
        resized_image = cv2.resize(cropped_image, tuple(self.config.input_size))
        
        # Scale keypoints
        scale_x = self.config.input_size[0] / w
        scale_y = self.config.input_size[1] / h
        
        scaled_keypoints = []
        scaled_coords = []
        for i in range(0, len(adjusted_keypoints), 3):
            kp_x = adjusted_keypoints[i] * scale_x
            kp_y = adjusted_keypoints[i+1] * scale_y
            kp_v = adjusted_keypoints[i+2]
            scaled_keypoints.extend([kp_x, kp_y, kp_v])
            if kp_v > 0:
                scaled_coords.append((kp_x, kp_y))
        
        # Apply albumentation transforms
        if len(scaled_coords) > 0:
            transformed = self.transform(image=resized_image, keypoints=scaled_coords)
            transformed_image = transformed['image']
            transformed_keypoints = transformed['keypoints']
            
            # Update scaled_keypoints with transformed coordinates
            coord_idx = 0
            final_keypoints = []
            for i in range(0, len(scaled_keypoints), 3):
                kp_v = scaled_keypoints[i+2]
                if kp_v > 0 and coord_idx < len(transformed_keypoints):
                    kp_x, kp_y = transformed_keypoints[coord_idx]
                    coord_idx += 1
                else:
                    kp_x, kp_y = scaled_keypoints[i], scaled_keypoints[i+1]
                final_keypoints.extend([kp_x, kp_y, kp_v])
        else:
            transformed_image = self.transform(image=resized_image)['image']
            final_keypoints = scaled_keypoints
        
        # Generate improved heatmaps with better Gaussian
        heatmaps, target_weight = self.generate_improved_heatmap(
            final_keypoints, self.config.output_size
        )
        
        return {
            'image': transformed_image,
            'heatmaps': torch.FloatTensor(heatmaps),
            'target_weight': torch.FloatTensor(target_weight),
            'keypoints': torch.FloatTensor(final_keypoints),
            'image_id': image_id
        }
    
    def generate_improved_heatmap(self, keypoints, output_size):
        """Generate better heatmaps with adaptive sigma"""
        heatmaps = np.zeros((self.config.num_keypoints, output_size[1], output_size[0]))
        target_weight = np.ones((self.config.num_keypoints, 1))
        
        for i in range(self.config.num_keypoints):
            x = keypoints[i * 3]
            y = keypoints[i * 3 + 1]
            v = keypoints[i * 3 + 2]
            
            if v == 0:
                target_weight[i] = 0
                continue
            
            # Scale to output size
            x_scaled = x * output_size[0] / self.config.input_size[0]
            y_scaled = y * output_size[1] / self.config.input_size[1]
            
            if x_scaled >= 0 and y_scaled >= 0 and x_scaled < output_size[0] and y_scaled < output_size[1]:
                # Adaptive sigma based on output size
                sigma = max(1, min(output_size) / 32)  # Better sigma calculation
                
                # Generate Gaussian with sub-pixel precision
                x_int, y_int = int(x_scaled), int(y_scaled)
                x_offset, y_offset = x_scaled - x_int, y_scaled - y_int
                
                # Create meshgrid around the keypoint
                size = int(6 * sigma + 3)
                if size % 2 == 0:
                    size += 1
                    
                x_range = np.arange(size, dtype=np.float32) - size // 2 + x_offset
                y_range = np.arange(size, dtype=np.float32) - size // 2 + y_offset
                xx, yy = np.meshgrid(x_range, y_range)
                
                # Generate Gaussian
                g = np.exp(-((xx) ** 2 + (yy) ** 2) / (2 * sigma ** 2))
                
                # Place on heatmap
                x_min = max(0, x_int - size // 2)
                x_max = min(output_size[0], x_int + size // 2 + 1)
                y_min = max(0, y_int - size // 2)
                y_max = min(output_size[1], y_int + size // 2 + 1)
                
                g_x_min = max(0, size // 2 - x_int)
                g_x_max = g_x_min + (x_max - x_min)
                g_y_min = max(0, size // 2 - y_int)
                g_y_max = g_y_min + (y_max - y_min)
                
                if x_max > x_min and y_max > y_min:
                    heatmaps[i, y_min:y_max, x_min:x_max] = np.maximum(
                        heatmaps[i, y_min:y_max, x_min:x_max],
                        g[g_y_min:g_y_max, g_x_min:g_x_max]
                    )
        
        return heatmaps, target_weight

def train_improved_model():
    print("=== TRAINING IMPROVED HRNET MODEL ===")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Enhanced config
    config = HRNetConfig()
    config.batch_size = 6  # Slightly smaller for more stable training
    config.epochs = 50  # More epochs for better convergence
    config.learning_rate = 0.0005  # Lower learning rate for stability
    config.weight_decay = 1e-5  # Stronger regularization
    
    # Load datasets
    print("Loading improved training data...")
    train_dataset = ImprovedDataset(
        coco_file='coco_annotations_clean.json',
        image_root='dataset',
        config=config,
        is_train=True
    )
    
    val_dataset = ImprovedDataset(
        coco_file='coco_annotations_clean.json',
        image_root='dataset', 
        config=config,
        is_train=False
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # Enhanced train/val split
    total_size = len(train_dataset)
    train_size = int(0.85 * total_size)  # Use more data for training
    val_size = total_size - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    print(f"Train samples: {len(train_subset)}")
    print(f"Val samples: {len(val_subset)}")
    
    # Data loaders with better settings
    train_loader = DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    # Initialize improved model
    model = ImprovedHRNet(config).to(device)
    
    # Better loss function
    criterion = ImprovedFocalLoss(alpha=2, beta=4)
    
    # Enhanced optimizer with scheduling
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
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
            
            # Calculate improved loss
            loss = criterion(pred_heatmaps, target_heatmaps, target_weights)
            
            # Add L2 regularization on predictions to prevent overfitting
            l2_reg = 0.001 * torch.norm(pred_heatmaps)
            total_loss = loss + l2_reg
            
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # Progress tracking
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1:2d}/{config.epochs}, "
                      f"Batch {batch_idx:2d}/{len(train_loader)}, "
                      f"Loss: {total_loss.item():.6f}")
        
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
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
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
            
            checkpoint_path = checkpoint_dir / "improved_hrnet_best.pth"
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
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"improved_hrnet_epoch_{epoch+1}.pth"
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
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    output_dir = Path("outputs/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "improved_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Training curves saved to: {output_dir / 'improved_training_curves.png'}")

if __name__ == "__main__":
    train_improved_model()

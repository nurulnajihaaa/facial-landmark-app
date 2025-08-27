"""
HRNet Training Configuration for Facial Keypoints
Based on HRNet-W18-C-Small-v2 for 14-point lateral face detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from collections import OrderedDict


class HRNetConfig:
    """Configuration for HRNet model"""
    def __init__(self):
        # Model configuration
        self.model_name = 'hrnet_w18_small_v2'
        self.num_keypoints = 14
        self.input_size = [256, 256]
        self.output_size = [64, 64]  # Heatmap size
        self.sigma = 2  # Gaussian sigma for heatmap generation
        
        # Training configuration
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.lr_factor = 0.1
        self.lr_step = [80, 90]
        
        # Data augmentation
        self.flip_prob = 0.5
        self.rotation_factor = 30
        self.scale_factor = 0.25
        self.color_factor = 0.2
        
        # Loss configuration
        self.use_target_weight = True
        self.loss_type = 'mse'  # or 'joint_mse'


class FacialKeypointsDataset(Dataset):
    """Dataset for facial keypoints in COCO format"""
    
    def __init__(self, coco_file, image_root, config, is_train=True):
        self.config = config
        self.is_train = is_train
        self.image_root = Path(image_root)
        
        # Load COCO annotations
        with open(coco_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image_id to annotation mapping
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(ann)
        
        # Filter images that have annotations
        self.images = [img for img in self.coco_data['images'] 
                      if img['id'] in self.annotations]
        
        # Data transformations
        if is_train:
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(
                    brightness=config.color_factor,
                    contrast=config.color_factor,
                    saturation=config.color_factor
                ),
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
    
    def __len__(self):
        return len(self.images)
    
    def generate_heatmap(self, keypoints, output_size):
        """Generate heatmap from keypoints"""
        heatmaps = np.zeros((self.config.num_keypoints, output_size[1], output_size[0]))
        target_weight = np.ones((self.config.num_keypoints, 1))
        
        for i in range(self.config.num_keypoints):
            x = keypoints[i * 3]
            y = keypoints[i * 3 + 1]
            v = keypoints[i * 3 + 2]
            
            if v == 0:  # Not labeled
                target_weight[i] = 0
                continue
            
            # Scale to output size
            x_scaled = x * output_size[0] / self.config.input_size[0]
            y_scaled = y * output_size[1] / self.config.input_size[1]
            
            if x_scaled >= 0 and y_scaled >= 0 and x_scaled < output_size[0] and y_scaled < output_size[1]:
                # Generate Gaussian heatmap
                sigma = self.config.sigma
                size = 6 * sigma + 3
                x_range = np.arange(0, size, 1, np.float32)
                y_range = x_range[:, None]
                x0 = y0 = size // 2
                
                # Gaussian kernel
                g = np.exp(-((x_range - x0) ** 2 + (y_range - y0) ** 2) / (2 * sigma ** 2))
                
                # Determine the range to place the Gaussian
                x_min, y_min = int(x_scaled - size // 2), int(y_scaled - size // 2)
                x_max, y_max = x_min + size, y_min + size
                
                # Clip to heatmap boundaries
                g_x_min = max(0, -x_min)
                g_y_min = max(0, -y_min)
                g_x_max = min(size, output_size[0] - x_min)
                g_y_max = min(size, output_size[1] - y_min)
                
                h_x_min = max(0, x_min)
                h_y_min = max(0, y_min)
                h_x_max = min(output_size[0], x_max)
                h_y_max = min(output_size[1], y_max)
                
                if h_x_max > h_x_min and h_y_max > h_y_min:
                    heatmaps[i, h_y_min:h_y_max, h_x_min:h_x_max] = \
                        g[g_y_min:g_y_max, g_x_min:g_x_max]
        
        return heatmaps, target_weight
    
    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info['id']
        
        # Load image - handle different path formats
        image_file = image_info['file_name']
        if not image_file.startswith('train/') and not image_file.startswith('valid/') and not image_file.startswith('test/'):
            # If no split prefix, add it based on the image being in our list
            image_path = self.image_root / image_file
        else:
            image_path = self.image_root / image_file
            
        # Alternative path if main path doesn't exist
        if not image_path.exists():
            # Try different splits
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
        
        # Get annotation (assuming one face per image for simplicity)
        ann = self.annotations[image_id][0]
        keypoints = ann['keypoints']
        bbox = ann['bbox']
        
        # Crop and resize image based on bbox
        x, y, w, h = bbox
        # Add padding around face
        padding = 0.3
        x = max(0, x - w * padding / 2)
        y = max(0, y - h * padding / 2)
        w = w * (1 + padding)
        h = h * (1 + padding)
        
        # Ensure bbox is within image boundaries
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        # Crop image
        cropped_image = image[int(y):int(y+h), int(x):int(x+w)]
        
        # Adjust keypoints relative to crop
        adjusted_keypoints = []
        for i in range(0, len(keypoints), 3):
            kp_x = keypoints[i] - x
            kp_y = keypoints[i+1] - y
            kp_v = keypoints[i+2]
            adjusted_keypoints.extend([kp_x, kp_y, kp_v])
        
        # Resize image to input size
        resized_image = cv2.resize(cropped_image, tuple(self.config.input_size))
        
        # Scale keypoints
        scale_x = self.config.input_size[0] / w
        scale_y = self.config.input_size[1] / h
        
        scaled_keypoints = []
        for i in range(0, len(adjusted_keypoints), 3):
            kp_x = adjusted_keypoints[i] * scale_x
            kp_y = adjusted_keypoints[i+1] * scale_y
            kp_v = adjusted_keypoints[i+2]
            scaled_keypoints.extend([kp_x, kp_y, kp_v])
        
        # Generate heatmaps
        heatmaps, target_weight = self.generate_heatmap(scaled_keypoints, self.config.output_size)
        
        # Apply transformations
        if self.transforms:
            resized_image = self.transforms(resized_image)
        
        return {
            'image': resized_image,
            'heatmaps': torch.FloatTensor(heatmaps),
            'target_weight': torch.FloatTensor(target_weight),
            'keypoints': torch.FloatTensor(scaled_keypoints),
            'image_id': image_id
        }


def load_pretrained_hrnet(config, pretrained_path=None):
    """Load HRNet model with pretrained weights"""
    try:
        from hrnet_model import get_pose_net, hrnet_w18_small_config
        model = get_pose_net(hrnet_w18_small_config, is_train=True)
    except ImportError:
        # Fallback to simple implementation
        model = SimpleHRNet(config)
    
    if pretrained_path and Path(pretrained_path).exists():
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        # Load weights (ignore size mismatches for final layer)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)} layers from pretrained model")
    
    return model


class SimpleHRNet(nn.Module):
    """Simplified HRNet implementation for facial keypoints"""
    
    def __init__(self, config):
        super(SimpleHRNet, self).__init__()
        self.num_keypoints = config.num_keypoints
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Multi-scale feature extraction
        self.stage1 = self._make_layer(64, 64, 2)
        self.stage2 = self._make_layer(64, 128, 2)
        self.stage3 = self._make_layer(128, 256, 2)
        
        # Final prediction layer
        self.final_layer = nn.Conv2d(256, self.num_keypoints, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        # allow controlling stride for the first conv (use stride=1 here so stages
        # don't further reduce spatial resolution)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # keep conv2 at stride=2 to produce a total downsample of 4 (256 -> 64)
        x = F.relu(self.bn2(self.conv2(x)))

        # stage layers were modified to use stride=1 for their first conv so they preserve
        # the current spatial resolution; this keeps output heatmap resolution high (64x64)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.final_layer(x)

        return x


class JointsMSELoss(nn.Module):
    """MSE loss for keypoint detection"""
    
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        # use sum reduction so we can control normalization explicitly
        self.criterion = nn.MSELoss(reduction='sum')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        """Compute per-joint MSE (sum over pixels) and normalize per-joint by
        number of visible examples to avoid vanishing gradients when heatmaps
        are large and sparse.

        Args:
            output: Tensor (B, J, H, W)
            target: Tensor (B, J, H, W)
            target_weight: Tensor (B, J, 1) or (B, J)
        Returns:
            scalar loss
        """
        # Vectorized, visibility-aware MSE that normalizes by the number of visible
        # joint samples and the heatmap spatial size to avoid excessively small gradients
        # when heatmaps are large and sparse.
        batch_size, num_joints, H, W = output.shape

        # normalize target_weight shape to (B, J, 1, 1)
        if target_weight is not None:
            if target_weight.dim() == 3 and target_weight.size(2) == 1:
                tw = target_weight.squeeze(2)
            else:
                tw = target_weight
            tw = tw.view(batch_size, num_joints, 1, 1).to(output.dtype)
        else:
            tw = None

        diff2 = (output - target) ** 2  # (B, J, H, W)

        if self.use_target_weight and tw is not None:
            weighted = diff2 * tw
            # number of visible joint samples (scalar)
            visible_count = int(tw.sum().item())
            if visible_count == 0:
                return torch.tensor(0.0, device=output.device, dtype=output.dtype)
            se = weighted.sum()
            loss = 0.5 * (se / (visible_count * H * W))
        else:
            se = diff2.sum()
            loss = 0.5 * (se / (batch_size * num_joints * H * W))

        return loss


def train_model(config, train_loader, val_loader, model, device, save_dir):
    """Training function"""
    model = model.to(device)
    
    # Loss and optimizer
    criterion = JointsMSELoss(use_target_weight=config.use_target_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, 
                                weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.lr_step, gamma=config.lr_factor
    )
    
    best_acc = 0.0
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            heatmaps = batch['heatmaps'].to(device)
            target_weight = batch['target_weight'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            # Ensure output heatmap spatial size matches target heatmap size
            if outputs is not None and isinstance(outputs, torch.Tensor):
                if outputs.dim() == 4 and outputs.shape[2:] != heatmaps.shape[2:]:
                    outputs = F.interpolate(outputs, size=heatmaps.shape[2:], mode='bilinear', align_corners=False)
            elif isinstance(outputs, (list, tuple)):
                # if model returns list/tuple (e.g. [heatmaps, ...]) try to resize the first element
                first = outputs[0]
                if isinstance(first, torch.Tensor) and first.dim() == 4 and first.shape[2:] != heatmaps.shape[2:]:
                    first = F.interpolate(first, size=heatmaps.shape[2:], mode='bilinear', align_corners=False)
                    outputs = (first, ) + tuple(outputs[1:]) if isinstance(outputs, tuple) else [first] + list(outputs[1:])
            
            loss = criterion(outputs, heatmaps, target_weight)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}/{config.epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}')
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    heatmaps = batch['heatmaps'].to(device)
                    target_weight = batch['target_weight'].to(device)

                    outputs = model(images)
                    # Match output heatmap size to target
                    if outputs is not None and isinstance(outputs, torch.Tensor):
                        if outputs.dim() == 4 and outputs.shape[2:] != heatmaps.shape[2:]:
                            outputs = F.interpolate(outputs, size=heatmaps.shape[2:], mode='bilinear', align_corners=False)
                    elif isinstance(outputs, (list, tuple)):
                        first = outputs[0]
                        if isinstance(first, torch.Tensor) and first.dim() == 4 and first.shape[2:] != heatmaps.shape[2:]:
                            first = F.interpolate(first, size=heatmaps.shape[2:], mode='bilinear', align_corners=False)
                            outputs = (first, ) + tuple(outputs[1:]) if isinstance(outputs, tuple) else [first] + list(outputs[1:])

                    loss = criterion(outputs, heatmaps, target_weight)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.6f}, '
                  f'Val Loss: {val_loss:.6f}')
        
        scheduler.step()
        
        # Save checkpoint
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__
            }
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Save best model
        if val_loader and val_loss < best_acc:
            best_acc = val_loss
            torch.save(model.state_dict(), save_dir / 'best_model.pth')


def main():
    parser = argparse.ArgumentParser(description='Train HRNet for facial keypoints')
    parser.add_argument('--coco_file', type=str, required=True,
                        help='Path to COCO annotation file')
    parser.add_argument('--image_root', type=str, required=True,
                        help='Root directory containing images')
    parser.add_argument('--pretrained', type=str, default='hrnet_w18_small_model_v1.pth',
                        help='Path to pretrained model')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    # Configuration
    config = HRNetConfig()
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Datasets
    train_dataset = FacialKeypointsDataset(args.coco_file, args.image_root, config, is_train=True)
    # For validation, you can split the dataset or use a separate validation set
    val_dataset = FacialKeypointsDataset(args.coco_file, args.image_root, config, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=4)
    
    # Model
    model = load_pretrained_hrnet(config, args.pretrained)
    
    # Train
    train_model(config, train_loader, val_loader, model, device, args.save_dir)


if __name__ == '__main__':
    main()

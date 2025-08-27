"""
Create a proper HRNet model that outputs the correct size and retrain it
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_hrnet_facial import HRNetConfig

class FixedHRNet(nn.Module):
    """Fixed HRNet that actually outputs 64x64 heatmaps"""
    
    def __init__(self, config):
        super(FixedHRNet, self).__init__()
        self.config = config
        self.num_keypoints = config.num_keypoints
        
        # Initial convolutions - reduce downsampling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)  # 256->128
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Keep 128
        self.bn2 = nn.BatchNorm2d(64)
        
        # Feature extraction stages with minimal downsampling
        self.stage1 = self._make_stage(64, 128, num_blocks=2, stride=1)   # Keep 128
        self.stage2 = self._make_stage(128, 256, num_blocks=2, stride=2)  # 128->64
        self.stage3 = self._make_stage(256, 512, num_blocks=2, stride=1)  # Keep 64
        
        # Final prediction layer
        self.final_layer = nn.Conv2d(512, self.num_keypoints, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        
        # First block with potential stride
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Remaining blocks with stride=1
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: 256x256
        x = F.relu(self.bn1(self.conv1(x)))  # -> 128x128
        x = F.relu(self.bn2(self.conv2(x)))  # -> 128x128
        
        x = self.stage1(x)  # -> 128x128
        x = self.stage2(x)  # -> 64x64
        x = self.stage3(x)  # -> 64x64
        
        x = self.final_layer(x)  # -> 64x64 with num_keypoints channels
        
        return x

def test_fixed_model():
    """Test the fixed model architecture"""
    config = HRNetConfig()
    model = FixedHRNet(config)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (1, {config.num_keypoints}, {config.output_size[1]}, {config.output_size[0]})")
    
    if output.shape[2:] == tuple(config.output_size[::-1]):
        print("✅ Model outputs correct size!")
        return True
    else:
        print("❌ Model output size is wrong")
        return False

def create_training_script():
    """Create a script to quickly retrain the fixed model"""
    script_content = '''"""
Quick training script for the fixed HRNet model
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_hrnet_facial import HRNetConfig, FacialKeypointsDataset, JointsMSELoss
from fixed_hrnet import FixedHRNet

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
                print(f"❌ Size mismatch: {outputs.shape} vs {heatmaps.shape}")
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
'''
    
    with open('quick_train_fixed.py', 'w') as f:
        f.write(script_content)
    
    print("Created quick_train_fixed.py")

if __name__ == '__main__':
    print("=== TESTING FIXED MODEL ARCHITECTURE ===")
    if test_fixed_model():
        create_training_script()
        print("\\n✅ Fixed model architecture works!")
        print("Run: python quick_train_fixed.py")
    else:
        print("\\n❌ Need to fix architecture further")

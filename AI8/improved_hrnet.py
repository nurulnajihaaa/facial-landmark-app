"""
Improved HRNet model with better architecture and training strategies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImprovedHRNet(nn.Module):
    """Improved HRNet with better feature extraction and attention mechanisms"""
    
    def __init__(self, config):
        super(ImprovedHRNet, self).__init__()
        self.config = config
        self.num_keypoints = config.num_keypoints
        
        # Initial feature extraction with more capacity
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Progressive downsampling with residual connections
        self.layer1 = self._make_layer(64, 128, 2, stride=2)   # 128x128
        self.layer2 = self._make_layer(128, 256, 2, stride=2)  # 64x64
        self.layer3 = self._make_layer(256, 512, 2, stride=1)  # 64x64
        
        # Multi-scale feature fusion
        self.fusion_conv = nn.Conv2d(512, 256, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(256)
        
        # Spatial attention module
        self.spatial_attention = SpatialAttentionModule(256)
        
        # Channel attention module  
        self.channel_attention = ChannelAttentionModule(256)
        
        # Feature refinement
        self.refine_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.refine_bn1 = nn.BatchNorm2d(256)
        self.refine_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.refine_bn2 = nn.BatchNorm2d(128)
        
        # Final heatmap prediction with better initialization
        self.final_conv = nn.Conv2d(128, self.num_keypoints, kernel_size=1)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Proper weight initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for final layer
        nn.init.normal_(self.final_conv.weight, std=0.001)
        nn.init.constant_(self.final_conv.bias, 0)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Progressive feature learning
        x = self.layer1(x)  # 128x128
        x = self.layer2(x)  # 64x64
        x = self.layer3(x)  # 64x64
        
        # Feature fusion
        x = self.relu(self.fusion_bn(self.fusion_conv(x)))
        
        # Apply attention mechanisms
        x = self.spatial_attention(x)
        x = self.channel_attention(x)
        
        # Feature refinement
        x = self.relu(self.refine_bn1(self.refine_conv1(x)))
        x = self.relu(self.refine_bn2(self.refine_conv2(x)))
        
        # Final heatmap prediction
        heatmaps = self.final_conv(x)
        
        # Apply sigmoid to ensure positive values
        heatmaps = torch.sigmoid(heatmaps)
        
        return heatmaps

class ResidualBlock(nn.Module):
    """Improved residual block with better feature learning"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = self.relu(out)
        
        return out

class SpatialAttentionModule(nn.Module):
    """Spatial attention to focus on important facial regions"""
    
    def __init__(self, channels):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Global pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and generate attention map
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(attention_input))
        
        return x * attention_map

class ChannelAttentionModule(nn.Module):
    """Channel attention to emphasize important features"""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Global pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Generate channel attention weights
        attention_weights = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention_weights.expand_as(x)

class ImprovedFocalLoss(nn.Module):
    """Improved focal loss for better keypoint localization"""
    
    def __init__(self, alpha=2, beta=4, reduction='mean'):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, pred, target, weights=None):
        """
        Args:
            pred: [B, K, H, W] predicted heatmaps
            target: [B, K, H, W] target heatmaps  
            weights: [B, K, 1] keypoint weights
        """
        # Ensure positive predictions
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        # Calculate focal loss
        pos_loss = -self.alpha * (1 - pred) ** self.beta * target * torch.log(pred)
        neg_loss = -self.alpha * pred ** self.beta * (1 - target) * torch.log(1 - pred)
        
        loss = pos_loss + neg_loss
        
        # Apply keypoint weights if provided
        if weights is not None:
            weights = weights.unsqueeze(-1)  # [B, K, 1, 1]
            loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

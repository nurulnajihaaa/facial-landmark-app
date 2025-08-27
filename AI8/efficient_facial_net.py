"""
Fast and effective facial keypoint model with efficient architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EfficientFacialNet(nn.Module):
    """Efficient and effective facial keypoint detection model"""
    
    def __init__(self, config):
        super(EfficientFacialNet, self).__init__()
        self.config = config
        self.num_keypoints = config.num_keypoints
        
        # Efficient feature extraction backbone
        self.backbone = nn.Sequential(
            # Initial conv block
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 1: 256 -> 128
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 2: 128 -> 64  
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 3: Feature refinement at 64x64
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Facial attention module - focuses on facial regions
        self.facial_attention = FacialAttention(256)
        
        # Multi-scale feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Keypoint-specific heads
        self.keypoint_heads = nn.ModuleList([
            KeypointHead(128) for _ in range(self.num_keypoints)
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # [B, 256, 64, 64]
        
        # Apply facial attention
        attended_features = self.facial_attention(features)
        
        # Feature fusion
        fused_features = self.feature_fusion(attended_features)  # [B, 128, 64, 64]
        
        # Generate keypoint-specific heatmaps
        heatmaps = []
        for head in self.keypoint_heads:
            heatmap = head(fused_features)  # [B, 1, 64, 64]
            heatmaps.append(heatmap)
        
        # Combine all heatmaps
        output = torch.cat(heatmaps, dim=1)  # [B, num_keypoints, 64, 64]
        
        # Apply sigmoid for proper probability distribution
        output = torch.sigmoid(output)
        
        return output

class FacialAttention(nn.Module):
    """Attention module specifically designed for facial features"""
    
    def __init__(self, channels):
        super(FacialAttention, self).__init__()
        
        # Spatial attention for facial regions
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention for feature importance
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Spatial attention
        spatial_att = self.spatial_conv(x)
        x_spatial = x * spatial_att
        
        # Channel attention
        channel_att = self.channel_conv(self.global_pool(x))
        x_attended = x_spatial * channel_att
        
        return x_attended

class KeypointHead(nn.Module):
    """Individual head for each keypoint type"""
    
    def __init__(self, in_channels):
        super(KeypointHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)  # Single channel output
        )
    
    def forward(self, x):
        return self.head(x)

class AdaptiveFocalLoss(nn.Module):
    """Adaptive focal loss that adjusts based on keypoint difficulty"""
    
    def __init__(self, alpha=2, beta=4, gamma=2):
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, pred, target, weights=None):
        """
        Args:
            pred: [B, K, H, W] predicted heatmaps
            target: [B, K, H, W] target heatmaps
            weights: [B, K, 1] keypoint weights
        """
        # Ensure predictions are in valid range
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        # Calculate adaptive weights based on target difficulty
        target_sum = target.sum(dim=[2, 3], keepdim=True)  # [B, K, 1, 1]
        difficulty = 1.0 / (target_sum + 1e-7)  # Harder keypoints get higher weight
        
        # Focal loss components
        pos_loss = -self.alpha * (1 - pred) ** self.beta * target * torch.log(pred)
        neg_loss = -self.alpha * pred ** self.beta * (1 - target) * torch.log(1 - pred)
        
        # Apply difficulty weighting
        focal_loss = (pos_loss + neg_loss) * difficulty
        
        # Apply keypoint weights if provided
        if weights is not None:
            weights = weights.unsqueeze(-1)  # [B, K, 1, 1]
            focal_loss = focal_loss * weights
        
        return focal_loss.mean()

def create_efficient_model(config):
    """Factory function to create the efficient model"""
    return EfficientFacialNet(config)

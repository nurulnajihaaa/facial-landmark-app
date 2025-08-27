"""
Proper HRNet implementation for facial keypoint detection
Based on the official HRNet paper with facial-specific modifications
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProperHRNet(nn.Module):
    """
    High-Resolution Network for facial keypoint detection
    Based on the official implementation with improvements for faces
    """
    
    def __init__(self, config):
        super(ProperHRNet, self).__init__()
        self.config = config
        self.num_keypoints = config.num_keypoints
        
        # Stem network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1 - Single resolution (64x64)
        self.layer1 = self._make_layer(64, 256, 4, stride=1)
        
        # Transition 1: 1 -> 2 branches
        self.transition1 = self._make_transition_layer([256], [32, 64])
        
        # Stage 2 - Two resolutions (64x64, 32x32)
        self.stage2 = self._make_stage(2, [32, 64], [4, 4])
        
        # Transition 2: 2 -> 3 branches  
        self.transition2 = self._make_transition_layer([32, 64], [32, 64, 128])
        
        # Stage 3 - Three resolutions (64x64, 32x32, 16x16)
        self.stage3 = self._make_stage(3, [32, 64, 128], [4, 4, 4])
        
        # Transition 3: 3 -> 4 branches
        self.transition3 = self._make_transition_layer([32, 64, 128], [32, 64, 128, 256])
        
        # Stage 4 - Four resolutions (64x64, 32x32, 16x16, 8x8)
        self.stage4 = self._make_stage(4, [32, 64, 128, 256], [4, 4, 4, 4])
        
        # Final layer - only use highest resolution
        self.final_layer = nn.Conv2d(32, self.num_keypoints, kernel_size=1, stride=1, padding=0)
        
        self._init_weights()
    
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        
        layers = []
        layers.append(BasicBlock(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        
        return nn.Sequential(*layers)
    
    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 
                                3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        
        return nn.ModuleList(transition_layers)
    
    def _make_stage(self, num_branches, num_channels, num_blocks):
        modules = []
        for i in range(num_blocks[0]):
            if i == 0:
                reset_multi_scale_output = not (i == num_blocks[0] - 1)
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches, num_channels, reset_multi_scale_output)
            )
        return nn.Sequential(*modules)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for final layer
        nn.init.normal_(self.final_layer.weight, std=0.001)
        nn.init.constant_(self.final_layer.bias, 0)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 1
        x = self.layer1(x)
        
        # Transition 1
        x_list = []
        for i in range(len(self.transition1)):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        
        # Stage 2
        y_list = self.stage2(x_list)
        
        # Transition 2
        x_list = []
        for i in range(len(self.transition2)):
            if self.transition2[i] is not None:
                if i < len(y_list):
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        
        # Stage 3
        y_list = self.stage3(x_list)
        
        # Transition 3
        x_list = []
        for i in range(len(self.transition3)):
            if self.transition3[i] is not None:
                if i < len(y_list):
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        
        # Stage 4
        y_list = self.stage4(x_list)
        
        # Use only the highest resolution output
        x = self.final_layer(y_list[0])
        
        return x

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, num_channels, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        
        self.branches = self._make_branches(num_branches, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)
    
    def _make_branches(self, num_branches, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(nn.Sequential(
                BasicBlock(num_channels[i], num_channels[i]),
                BasicBlock(num_channels[i], num_channels[i]),
                BasicBlock(num_channels[i], num_channels[i]),
                BasicBlock(num_channels[i], num_channels[i]),
            ))
        return nn.ModuleList(branches)
    
    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        
        num_branches = self.num_branches
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.branches[j][0].conv1.in_channels, 
                                self.branches[i][0].conv1.in_channels, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(self.branches[i][0].conv1.in_channels),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = self.branches[i][0].conv1.in_channels
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.branches[j][0].conv1.in_channels,
                                        num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)
                            ))
                        else:
                            num_outchannels_conv3x3 = self.branches[j][0].conv1.in_channels
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.branches[j][0].conv1.in_channels,
                                        num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        
        return nn.ModuleList(fuse_layers)
    
    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        
        return x_fuse

class HRNetLoss(nn.Module):
    """Proper loss function for HRNet facial keypoint detection"""
    
    def __init__(self):
        super(HRNetLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target, target_weight):
        """
        Args:
            pred: [B, K, H, W] predicted heatmaps
            target: [B, K, H, W] target heatmaps
            target_weight: [B, K, 1] keypoint weights
        """
        batch_size, num_keypoints, height, width = pred.shape
        
        # Reshape for loss calculation
        pred = pred.reshape(batch_size, num_keypoints, -1)
        target = target.reshape(batch_size, num_keypoints, -1)
        
        loss = 0
        for i in range(num_keypoints):
            # Get keypoint-specific predictions and targets
            pred_kp = pred[:, i, :]  # [B, H*W]
            target_kp = target[:, i, :]  # [B, H*W]
            weight_kp = target_weight[:, i, 0]  # [B]
            
            # Calculate MSE for this keypoint
            kp_loss = ((pred_kp - target_kp) ** 2).mean(dim=1)  # [B]
            
            # Apply weights
            weighted_loss = (kp_loss * weight_kp).mean()
            loss += weighted_loss
        
        return loss / num_keypoints

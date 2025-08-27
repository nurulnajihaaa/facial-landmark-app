"""
Model utilities for loading and handling different HRNet configurations
"""

import torch
import torch.nn as nn
from pathlib import Path
from collections import OrderedDict


def _adapt_tensor_to_shape(src: torch.Tensor, target_shape):
    """
    Adapt `src` tensor to `target_shape` by slicing or tiling as needed.
    This is lossy but allows loading as many pretrained values as possible.
    Rules:
    - If a dimension in src is larger than target, slice the first target_size.
    - If a dimension in src is smaller than target, tile (repeat) along that dim and then slice.
    Works for 1D (BN/linear) and 4D (conv) tensors and general n-D tensors.
    """
    src = src.detach()
    src_shape = tuple(src.shape)
    tgt_shape = tuple(target_shape)

    if src_shape == tgt_shape:
        return src

    # Create a tensor by repeating src enough times to cover target dims, then slice
    reps = []
    for s, t in zip(src_shape, tgt_shape):
        if s == 0:
            reps.append(1)
        elif s >= t:
            reps.append(1)
        else:
            reps.append((t + s - 1) // s)

    # If src has fewer dims than target, pad with ones (unlikely)
    if len(src_shape) < len(tgt_shape):
        # reshape src to have leading dims of 1
        new_shape = (1,) * (len(tgt_shape) - len(src_shape)) + src_shape
        src = src.view(new_shape)
        src_shape = tuple(src.shape)
        reps = [1] * (len(tgt_shape) - len(src_shape)) + reps

    tiled = src.repeat(*reps)

    # Now slice to target shape
    slices = tuple(slice(0, t) for t in tgt_shape)
    adapted = tiled[slices]

    # Cast to same dtype/device as src
    return adapted

def create_compatible_model(pretrained_path, target_keypoints=14):
    """
    Create a model compatible with the pretrained weights
    """
    if not Path(pretrained_path).exists():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
    
    # Load the checkpoint to inspect its structure
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    clean_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        clean_state_dict[name] = v
    
    # Analyze the architecture from the state dict
    has_hrnet_structure = any('stage' in key for key in clean_state_dict.keys())
    
    if has_hrnet_structure:
        # This is an HRNet model, try to infer the configuration
        model = create_hrnet_from_state_dict(clean_state_dict, target_keypoints)
    else:
        # This might be a different architecture, create a generic model
        model = create_generic_model(clean_state_dict, target_keypoints)
    
    return model, clean_state_dict

def create_hrnet_from_state_dict(state_dict, target_keypoints):
    """
    Create HRNet model that matches the state dict structure
    """
    try:
        from hrnet_model import get_pose_net
        # Try to infer precise NUM_CHANNELS per stage from the checkpoint
        def infer_hrnet_channels(sd):
            # Stage2 channels
            s2 = []
            for i in range(4):
                key = f'stage2.0.branches.{i}.0.conv1.weight'
                if key in sd:
                    s2.append(int(sd[key].shape[0]))
            if not s2:
                # fallback: transition1.*.0.weight
                i = 0
                while True:
                    key = f'transition1.{i}.0.weight'
                    if key in sd:
                        s2.append(int(sd[key].shape[0]))
                        i += 1
                    else:
                        break

            # Stage3 channels
            s3 = []
            for i in range(4):
                key = f'stage3.0.branches.{i}.0.conv1.weight'
                if key in sd:
                    s3.append(int(sd[key].shape[0]))
            if not s3 and s2:
                s3 = [s2[0], s2[0]*2, s2[0]*4][:3]

            # Stage4 channels
            s4 = []
            for i in range(4):
                key = f'stage4.0.branches.{i}.0.conv1.weight'
                if key in sd:
                    s4.append(int(sd[key].shape[0]))
            if not s4 and s3:
                s4 = [s3[0], s3[0]*2, s3[0]*4, s3[0]*8]

            return s2, s3, s4

        s2, s3, s4 = infer_hrnet_channels(state_dict)
        # use defaults if inference failed
        if not s2:
            s2 = [18, 36]
        if not s3:
            s3 = [18, 36, 72]
        if not s4:
            s4 = [18, 36, 72, 144]

        config = {
            'NUM_JOINTS': target_keypoints,
            'PRETRAINED_LAYERS': ['*'],
            'FINAL_CONV_KERNEL': 1,
            'STAGE2': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': len(s2),
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4] * len(s2),
                'NUM_CHANNELS': s2,
                'FUSE_METHOD': 'SUM'
            },
            'STAGE3': {
                'NUM_MODULES': 4,
                'NUM_BRANCHES': len(s3),
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4] * len(s3),
                'NUM_CHANNELS': s3,
                'FUSE_METHOD': 'SUM'
            },
            'STAGE4': {
                'NUM_MODULES': 3,
                'NUM_BRANCHES': len(s4),
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4] * len(s4),
                'NUM_CHANNELS': s4,
                'FUSE_METHOD': 'SUM'
            }
        }
        model = get_pose_net(config, is_train=False)
        # Try strict load if shapes match
        try:
            model.load_state_dict(state_dict, strict=True)
            print('✅ Strictly loaded checkpoint into inferred HRNet architecture')
            return model
        except Exception as e:
            print(f'⚠️ Strict load failed: {e}. Falling back to adaptive load.')

    except ImportError:
        # Fallback to simple model
        model = SimpleCompatibleModel(target_keypoints)
    
    return model

def create_generic_model(state_dict, target_keypoints):
    """
    Create a generic model that might work with various architectures
    """
    return SimpleCompatibleModel(target_keypoints)

class SimpleCompatibleModel(nn.Module):
    """
    A simple model that can adapt to different input sizes
    """
    def __init__(self, num_keypoints=14):
        super(SimpleCompatibleModel, self).__init__()
        self.num_keypoints = num_keypoints
        
        # Simple CNN architecture
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling layers to get back to reasonable resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Final layer
        self.final = nn.Conv2d(128, num_keypoints, kernel_size=1)
        
        self._initialize_weights()
    
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
        x = self.features(x)
        x = self.upsample(x)
        x = self.final(x)
        return x

def load_model_safely(model_path, target_keypoints=14, device='cpu'):
    """
    Safely load a model, handling architecture mismatches
    """
    try:
        # First, try to create a compatible model and get the checkpoint state dict
        model, state_dict = create_compatible_model(model_path, target_keypoints)

        model_dict = model.state_dict()

        # Build a new dict where we adapt mismatched tensors where reasonable
        adapted_dict = OrderedDict()
        loaded_count = 0
        skipped_count = 0

        for name, param in model_dict.items():
            if name in state_dict:
                ckpt_tensor = state_dict[name]
                try:
                    if tuple(ckpt_tensor.shape) == tuple(param.shape):
                        adapted = ckpt_tensor
                        loaded_count += 1
                    else:
                        # Try to adapt shapes (conv/linear/bn etc.)
                        adapted = _adapt_tensor_to_shape(ckpt_tensor, param.shape)
                        loaded_count += 1
                        print(f"🔧 Adapted {name}: checkpoint {tuple(ckpt_tensor.shape)} -> model {tuple(param.shape)}")
                except Exception as e:
                    print(f"⚠️ Could not adapt {name}: {e}")
                    adapted = param
                    skipped_count += 1
            else:
                # Missing in checkpoint, keep model init
                adapted = param
                skipped_count += 1

            adapted_dict[name] = adapted

        # Load the adapted parameters (strict=False to allow missing extras)
        model.load_state_dict(adapted_dict, strict=False)

        print(f"✅ Loaded/adapted {loaded_count} params, skipped {skipped_count} params from {model_path}")
        return model.to(device)
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Creating new simple model instead...")
        
        # Fallback: create a simple new model
        model = SimpleCompatibleModel(target_keypoints)
        return model.to(device)

def get_model_info(model_path):
    """
    Get information about a model checkpoint
    """
    if not Path(model_path).exists():
        return {"error": f"Model file not found: {model_path}"}
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            extra_info = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            extra_info = {k: v for k, v in checkpoint.items() if k != 'model'}
        else:
            state_dict = checkpoint
            extra_info = {}
        
        # Analyze the state dict
        total_params = sum(p.numel() for p in state_dict.values())
        layer_types = {}
        
        for key in state_dict.keys():
            layer_type = key.split('.')[0] if '.' in key else key
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        # Try to determine architecture
        arch_type = "Unknown"
        if any('stage' in key for key in state_dict.keys()):
            arch_type = "HRNet"
        elif any('layer' in key for key in state_dict.keys()):
            arch_type = "ResNet-like"
        elif any('features' in key for key in state_dict.keys()):
            arch_type = "Feature-based CNN"
        
        return {
            "total_parameters": total_params,
            "total_layers": len(state_dict),
            "architecture_type": arch_type,
            "layer_types": layer_types,
            "extra_info": extra_info,
            "file_size_mb": Path(model_path).stat().st_size / (1024 * 1024)
        }
        
    except Exception as e:
        return {"error": f"Error analyzing model: {e}"}

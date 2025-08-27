"""
Diagnostic script to understand what's going wrong with our models
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

sys.path.append('.')
from improved_hrnet import ImprovedHRNet

def simple_diagnostic():
    """Diagnose what's happening with our models"""
    print("=== DIAGNOSTIC ANALYSIS ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load a test image
    test_images = list(Path('dataset/test/images').glob('*.jpg'))
    if not test_images:
        print("❌ No test images found")
        return
    
    image_path = test_images[0]
    print(f"🖼️  Test image: {image_path.name}")
    
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (256, 256))
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_resized).float()
    image_tensor = image_tensor.permute(2, 0, 1) / 255.0
    
    # Apply normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    print(f"📐 Image tensor shape: {image_tensor.shape}")
    
    # Test different models
    models_to_test = [
        ("Improved HRNet", "checkpoints/improved_hrnet_best.pth", ImprovedHRNet)
    ]
    
    for model_name, model_path, model_class in models_to_test:
        if not Path(model_path).exists():
            print(f"❌ {model_name} not found: {model_path}")
            continue
        
        print(f"\\n🔍 Testing {model_name}")
        
        try:
            # Create config
            class Config:
                def __init__(self):
                    self.num_keypoints = 14
                    self.num_classes = 14
            
            config = Config()
            
            # Load model
            model = model_class(config).to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"  📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Forward pass
            with torch.no_grad():
                output = model(image_tensor)
            
            print(f"  📤 Output shape: {output.shape}")
            print(f"  📈 Output range: [{output.min():.6f}, {output.max():.6f}]")
            print(f"  🎯 Output mean: {output.mean():.6f}")
            print(f"  📊 Output std: {output.std():.6f}")
            
            # Analyze heatmaps
            heatmaps = output[0].cpu().numpy()  # Remove batch dimension
            
            # Check each keypoint channel
            confident_keypoints = 0
            for i in range(heatmaps.shape[0]):
                heatmap = heatmaps[i]
                max_val = heatmap.max()
                
                if max_val > 0.1:
                    confident_keypoints += 1
                    # Find peak location
                    peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    print(f"    KP {i:2d}: max={max_val:.3f} at {peak_idx}")
            
            print(f"  ✅ Confident keypoints: {confident_keypoints}/{heatmaps.shape[0]}")
            
            # Visualize first few heatmaps
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            for i in range(min(8, heatmaps.shape[0])):
                ax = axes[i]
                im = ax.imshow(heatmaps[i], cmap='hot')
                ax.set_title(f"KP {i} (max: {heatmaps[i].max():.3f})")
                ax.axis('off')
                plt.colorbar(im, ax=ax)
            
            # Hide unused subplots
            for i in range(min(8, heatmaps.shape[0]), 8):
                axes[i].axis('off')
            
            plt.suptitle(f"{model_name} - Heatmaps")
            plt.tight_layout()
            
            output_dir = Path("outputs/diagnostic")
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"{model_name.lower().replace(' ', '_')}_heatmaps.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"  💾 Saved heatmaps: {save_path}")
            
        except Exception as e:
            print(f"  ❌ Error testing {model_name}: {e}")
    
    # Also check the training data annotations
    print(f"\\n🔍 Checking training data...")
    
    # Load COCO annotations
    coco_file = 'coco_annotations_clean.json'
    if Path(coco_file).exists():
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        print(f"📋 COCO annotations loaded:")
        print(f"  Images: {len(coco_data['images'])}")
        print(f"  Annotations: {len(coco_data['annotations'])}")
        
        # Check a few annotations
        for i, ann in enumerate(coco_data['annotations'][:3]):
            keypoints = ann['keypoints']
            kp_array = np.array(keypoints).reshape(-1, 3)
            visible_kps = np.sum(kp_array[:, 2] > 0)
            
            print(f"  Ann {i}: {visible_kps} visible keypoints")
            print(f"    X range: {kp_array[kp_array[:, 2] > 0, 0].min():.1f} - {kp_array[kp_array[:, 2] > 0, 0].max():.1f}")
            print(f"    Y range: {kp_array[kp_array[:, 2] > 0, 1].min():.1f} - {kp_array[kp_array[:, 2] > 0, 1].max():.1f}")
    
    print(f"\\n✅ Diagnostic completed!")

if __name__ == "__main__":
    simple_diagnostic()

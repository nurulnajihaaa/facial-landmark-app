"""
Simple test script for proper HRNet without config dependencies
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append('.')
from proper_hrnet import ProperHRNet

class SimpleConfig:
    """Simple config class for testing"""
    def __init__(self):
        self.input_size = [256, 256]
        self.output_size = [64, 64]
        self.num_keypoints = 14
        self.sigma = 2.0
        self.batch_size = 8

def heatmaps_to_keypoints(heatmaps, input_size):
    """Convert heatmaps to keypoint coordinates"""
    batch_size, num_joints, h, w = heatmaps.shape
    keypoints = []
    
    for b in range(batch_size):
        batch_keypoints = []
        for j in range(num_joints):
            heatmap = heatmaps[b, j]
            
            # Find maximum location
            max_val = torch.max(heatmap)
            if max_val < 0.1:  # Low confidence
                batch_keypoints.extend([0, 0, 0])
                continue
            
            # Get coordinates of maximum
            max_idx = torch.argmax(heatmap.flatten())
            y = max_idx // w
            x = max_idx % w
            
            # Sub-pixel refinement using weighted average around peak
            if 1 <= x < w-1 and 1 <= y < h-1:
                # 3x3 window around peak
                window = heatmap[y-1:y+2, x-1:x+2]
                weights = F.softmax(window.flatten(), dim=0).view(3, 3)
                
                # Weighted coordinates
                y_coords = torch.arange(y-1, y+2, dtype=torch.float32)
                x_coords = torch.arange(x-1, x+2, dtype=torch.float32)
                
                refined_y = torch.sum(weights * y_coords.view(-1, 1)).item()
                refined_x = torch.sum(weights * x_coords.view(1, -1)).item()
            else:
                refined_x, refined_y = float(x), float(y)
            
            # Scale to input image size
            x_scaled = refined_x * input_size[0] / w
            y_scaled = refined_y * input_size[1] / h
            
            batch_keypoints.extend([x_scaled, y_scaled, float(max_val)])
        
        keypoints.append(batch_keypoints)
    
    return np.array(keypoints)

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess image for HRNet inference"""
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path
    
    original_shape = image.shape
    
    # Resize to model input size
    image_resized = cv2.resize(image, target_size)
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_resized).float()
    image_tensor = image_tensor.permute(2, 0, 1) / 255.0
    
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor.unsqueeze(0), original_shape, image_resized

def simple_anatomical_check(coords):
    """Simple anatomical correctness check"""
    if len(coords) < 4:
        return False, "Insufficient keypoints"
    
    # Check spread
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    
    if x_range < 20 or y_range < 20:
        return False, f"Poor spread: x_range={x_range:.1f}, y_range={y_range:.1f}"
    
    # Check if points are not all clustered
    center_x = np.mean(coords[:, 0])
    center_y = np.mean(coords[:, 1])
    
    distances = np.sqrt((coords[:, 0] - center_x)**2 + (coords[:, 1] - center_y)**2)
    avg_distance = np.mean(distances)
    
    if avg_distance < 15:
        return False, f"Too clustered: avg_distance={avg_distance:.1f}"
    
    return True, "Anatomically plausible"

def test_single_image(model, image_path, config, device):
    """Test model on a single image"""
    print(f"\\n🔍 Testing: {Path(image_path).name}")
    
    try:
        # Preprocess
        image_tensor, original_shape, resized_image = preprocess_image(image_path, tuple(config.input_size))
        
        # Inference
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            heatmaps = model(image_tensor)
            
            if isinstance(heatmaps, list):
                heatmaps = heatmaps[-1]  # Use final stage output
        
        # Convert to keypoints
        keypoints = heatmaps_to_keypoints(heatmaps.cpu(), config.input_size)
        
        # Extract coordinates for validation
        coords = []
        for i in range(0, len(keypoints[0]), 3):
            x, y, conf = keypoints[0][i:i+3]
            if conf > 0.1:  # Only include confident predictions
                coords.append([x, y])
        
        coords = np.array(coords) if coords else np.zeros((0, 2))
        
        # Simple anatomical check
        is_good, message = simple_anatomical_check(coords)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image with keypoints
        axes[0].imshow(resized_image)
        if len(coords) > 0:
            axes[0].scatter(coords[:, 0], coords[:, 1], c='red', s=50, alpha=0.8)
            # Add keypoint numbers
            for i, (x, y) in enumerate(coords):
                axes[0].text(x+2, y+2, str(i), color='yellow', fontsize=8)
        axes[0].set_title(f"Predictions (n={len(coords)})")
        axes[0].axis('off')
        
        # Heatmap sum
        heatmap_sum = np.sum(heatmaps[0].cpu().numpy(), axis=0)
        axes[1].imshow(resized_image, alpha=0.7)
        im = axes[1].imshow(heatmap_sum, alpha=0.6, cmap='hot')
        axes[1].set_title("Heatmap Overlay")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Validation results
        axes[2].axis('off')
        result_text = f"ANATOMICAL CHECK\\n\\n"
        
        if is_good:
            result_text += "✅ ANATOMICALLY PLAUSIBLE\\n"
        else:
            result_text += "❌ ANATOMICALLY INCORRECT\\n"
        
        result_text += f"\\n{message}\\n\\n"
        result_text += f"Keypoints detected: {len(coords)}\\n"
        result_text += f"Confidence range: {keypoints[0][2::3].min():.3f} - {keypoints[0][2::3].max():.3f}\\n"
        
        if len(coords) > 0:
            result_text += f"X range: {coords[:, 0].min():.1f} - {coords[:, 0].max():.1f}\\n"
            result_text += f"Y range: {coords[:, 1].min():.1f} - {coords[:, 1].max():.1f}\\n"
        
        axes[2].text(0.05, 0.95, result_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[2].set_title("Validation Results")
        
        plt.tight_layout()
        
        # Save result
        output_dir = Path("outputs/proper_hrnet_simple_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"test_{Path(image_path).stem}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"  Result: {'✅ Good' if is_good else '❌ Poor'} - {message}")
        print(f"  Saved: {save_path}")
        
        return is_good, coords, message
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False, np.array([]), str(e)

def main():
    print("=== SIMPLE PROPER HRNET TEST ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create config
    config = SimpleConfig()
    
    # Initialize model
    model = ProperHRNet(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load only the model weights (skip config)
    model_path = "checkpoints/proper_hrnet_best.pth"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        # Load checkpoint but only extract model weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Loaded model weights from {model_path}")
        if 'val_loss' in checkpoint:
            print(f"📈 Training validation loss: {checkpoint['val_loss']:.6f}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Find test images
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(Path('dataset/test/images').glob(ext))
    
    if not test_images:
        print("❌ No test images found in dataset/test/images/")
        return
    
    print(f"📸 Found {len(test_images)} test images")
    
    # Test on a few images
    good_results = 0
    total_tested = 0
    
    for i, image_path in enumerate(test_images[:5]):  # Test first 5 images
        is_good, coords, message = test_single_image(model, str(image_path), config, device)
        
        total_tested += 1
        if is_good:
            good_results += 1
    
    # Summary
    if total_tested > 0:
        accuracy_rate = good_results / total_tested
        print(f"\\n📊 SUMMARY")
        print(f"Good predictions: {good_results}/{total_tested} ({accuracy_rate:.1%})")
        
        if accuracy_rate >= 0.8:
            print("🎉 EXCELLENT results!")
        elif accuracy_rate >= 0.6:
            print("✅ GOOD results")
        elif accuracy_rate >= 0.4:
            print("⚠️  MODERATE results")
        else:
            print("❌ POOR results")

if __name__ == "__main__":
    main()

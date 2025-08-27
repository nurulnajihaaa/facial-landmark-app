"""
Test the improved HRNet model that was working
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append('.')

# Import the improved HRNet classes
from improved_hrnet import ImprovedHRNet

class SimpleConfig:
    """Simple config for testing"""
    def __init__(self):
        self.input_size = [256, 256]
        self.output_size = [64, 64]
        self.num_keypoints = 14
        self.num_classes = 14  # Same as num_keypoints

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess image for inference"""
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path
    
    original_shape = image.shape
    image_resized = cv2.resize(image, target_size)
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_resized).float()
    image_tensor = image_tensor.permute(2, 0, 1) / 255.0
    
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor.unsqueeze(0), original_shape, image_resized

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
            
            # Scale to input image size
            x_scaled = float(x) * input_size[0] / w
            y_scaled = float(y) * input_size[1] / h
            
            batch_keypoints.extend([x_scaled, y_scaled, float(max_val)])
        
        keypoints.append(batch_keypoints)
    
    return np.array(keypoints)

def analyze_predictions(coords):
    """Analyze anatomical correctness of predictions"""
    if len(coords) < 4:
        return False, "Insufficient keypoints", 0.0
    
    # Calculate spread
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    
    spread_score = min(1.0, (x_range + y_range) / 200.0)
    
    # Check clustering
    center_x, center_y = np.mean(coords, axis=0)
    distances = np.sqrt((coords[:, 0] - center_x)**2 + (coords[:, 1] - center_y)**2)
    avg_distance = np.mean(distances)
    
    cluster_score = min(1.0, avg_distance / 30.0)
    
    # Overall score
    overall_score = (spread_score + cluster_score) / 2.0
    
    is_good = overall_score > 0.5 and x_range > 20 and y_range > 20
    
    message = f"Spread: {x_range:.1f}x{y_range:.1f}, Avg dist: {avg_distance:.1f}, Score: {overall_score:.3f}"
    
    return is_good, message, overall_score

def test_improved_hrnet():
    """Test the improved HRNet model"""
    print("=== TESTING IMPROVED HRNET ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    config = SimpleConfig()
    
    # Initialize model
    model = ImprovedHRNet(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load the working improved model
    model_path = "checkpoints/improved_hrnet_best.pth"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Loaded improved HRNet from {model_path}")
        if 'val_loss' in checkpoint:
            print(f"📈 Validation loss: {checkpoint['val_loss']:.6f}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Find test images
    test_images = list(Path('dataset/test/images').glob('*.jpg'))
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"📸 Found {len(test_images)} test images")
    
    # Test images
    good_results = 0
    total_tested = 0
    
    output_dir = Path("outputs/improved_hrnet_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, image_path in enumerate(test_images[:5]):
        print(f"\\n🔍 Testing: {image_path.name}")
        
        try:
            # Preprocess
            image_tensor, original_shape, resized_image = preprocess_image(str(image_path))
            
            # Inference
            with torch.no_grad():
                image_tensor = image_tensor.to(device)
                heatmaps = model(image_tensor)
            
            # Convert to keypoints
            keypoints = heatmaps_to_keypoints(heatmaps.cpu(), config.input_size)
            
            # Extract coordinates
            coords = []
            for j in range(0, len(keypoints[0]), 3):
                x, y, conf = keypoints[0][j:j+3]
                if conf > 0.1:
                    coords.append([x, y])
            
            coords = np.array(coords) if coords else np.zeros((0, 2))
            
            # Analyze
            is_good, message, score = analyze_predictions(coords)
            
            # Visualize
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original with keypoints
            axes[0].imshow(resized_image)
            if len(coords) > 0:
                axes[0].scatter(coords[:, 0], coords[:, 1], c='red', s=50, alpha=0.8)
                for idx, (x, y) in enumerate(coords):
                    axes[0].text(x+2, y+2, str(idx), color='yellow', fontsize=8)
            axes[0].set_title(f"Predictions (n={len(coords)})")
            axes[0].axis('off')
            
            # Heatmap overlay
            heatmap_sum = np.sum(heatmaps[0].cpu().numpy(), axis=0)
            axes[1].imshow(resized_image, alpha=0.7)
            im = axes[1].imshow(heatmap_sum, alpha=0.6, cmap='hot')
            axes[1].set_title("Heatmap Overlay")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            # Results
            axes[2].axis('off')
            result_text = "ANALYSIS\\n\\n"
            result_text += f"{'✅ GOOD' if is_good else '❌ POOR'}\\n\\n"
            result_text += f"{message}\\n\\n"
            result_text += f"Keypoints: {len(coords)}\\n"
            if len(coords) > 0:
                result_text += f"X: {coords[:, 0].min():.1f}-{coords[:, 0].max():.1f}\\n"
                result_text += f"Y: {coords[:, 1].min():.1f}-{coords[:, 1].max():.1f}\\n"
            
            axes[2].text(0.05, 0.95, result_text, transform=axes[2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[2].set_title("Analysis")
            
            plt.tight_layout()
            
            save_path = output_dir / f"test_{i+1}_{image_path.stem}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            total_tested += 1
            if is_good:
                good_results += 1
                print(f"  ✅ GOOD - {message}")
            else:
                print(f"  ❌ POOR - {message}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    if total_tested > 0:
        accuracy = good_results / total_tested
        print(f"\\n📊 RESULTS: {good_results}/{total_tested} ({accuracy:.1%})")
        
        if accuracy >= 0.8:
            print("🎉 EXCELLENT anatomical accuracy!")
        elif accuracy >= 0.6:
            print("✅ GOOD anatomical accuracy")
        elif accuracy >= 0.4:
            print("⚠️  MODERATE anatomical accuracy")
        else:
            print("❌ POOR anatomical accuracy")
        
        print(f"💾 Results saved to: {output_dir}")

if __name__ == "__main__":
    test_improved_hrnet()

"""
Test the clean trained model
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('.')

from facial_keypoints_app import FacialKeypointsInference

def test_clean_model():
    print("=== TESTING CLEAN TRAINED MODEL ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "checkpoints/clean_fixed_hrnet_best.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Clean model not found: {model_path}")
        return
    
    print(f"Loading clean model: {model_path}")
    inference = FacialKeypointsInference(model_path, device)
    print("✅ Clean model loaded!")
    
    # Test on a few images
    test_images = [
        "dataset/train/images/17-900-Kid-Face-Side-View-Stock-Photos-Pictures-Royalty-Free-Images-iStock-Boy-head-side-view_jpg.rf.2c9487846c045251229cbbdfc8e0fb51.jpg",
        "dataset/train/images/1_jpg.rf.76115b6b11e4f88bad230d67d34a4ac7.jpg", 
        "dataset/train/images/Facial-expressions-beautiful-woman-face-different-emotions-set_jpg.rf.1b1e8b66f2eca23b3d2e77d4cdbb38b7.jpg"
    ]
    
    results = []
    
    for i, image_path in enumerate(test_images):
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            continue
            
        print(f"\n--- Testing image {i+1}: {Path(image_path).name} ---")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load: {image_path}")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        keypoints, confidence, processed_image = inference.predict(image_rgb)
        
        print(f"✅ Prediction successful!")
        print(f"Keypoints shape: {keypoints.shape}")
        print(f"Mean confidence: {confidence.mean():.3f}")
        print(f"Confidence range: {confidence.min():.3f} - {confidence.max():.3f}")
        
        # Quality check
        keypoint_spread_x = keypoints[:, 0].max() - keypoints[:, 0].min()
        keypoint_spread_y = keypoints[:, 1].max() - keypoints[:, 1].min()
        print(f"Keypoint spread: X={keypoint_spread_x:.1f}px, Y={keypoint_spread_y:.1f}px")
        
        # Basic sanity check
        mean_x, mean_y = keypoints.mean(axis=0)
        print(f"Mean keypoint position: ({mean_x:.1f}, {mean_y:.1f})")
        
        # Check if keypoints are within reasonable bounds
        if keypoint_spread_x > 50 and keypoint_spread_y > 50:
            print("✅ GOOD: Reasonable keypoint spread")
        else:
            print("⚠️  WARNING: Keypoints might be clustered")
        
        if 50 < mean_x < 200 and 50 < mean_y < 200:
            print("✅ GOOD: Keypoints in reasonable position")
        else:
            print("⚠️  WARNING: Keypoints in unusual position")
        
        results.append({
            'image_path': image_path,
            'keypoints': keypoints,
            'confidence': confidence,
            'processed_image': processed_image,
            'spread_x': keypoint_spread_x,
            'spread_y': keypoint_spread_y
        })
    
    # Create comparison visualization
    if results:
        create_comparison_visualization(results)
    
    return results

def create_comparison_visualization(results):
    """Create visualization comparing clean model results"""
    
    n_images = len(results)
    fig, axes = plt.subplots(2, n_images, figsize=(5*n_images, 10))
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    for i, result in enumerate(results):
        keypoints = result['keypoints']
        processed_image = result['processed_image']
        image_name = Path(result['image_path']).name
        
        # Original processed image
        axes[0, i].imshow(processed_image)
        axes[0, i].set_title(f"Original: {image_name[:30]}...")
        axes[0, i].axis('off')
        
        # Image with keypoints
        axes[1, i].imshow(processed_image)
        axes[1, i].scatter(keypoints[:, 0], keypoints[:, 1], 
                          c='red', s=50, alpha=0.8)
        
        # Annotate keypoint numbers
        for j, (x, y) in enumerate(keypoints):
            axes[1, i].annotate(str(j), (x, y), xytext=(3, 3), 
                              textcoords='offset points', fontsize=8, 
                              color='yellow', weight='bold')
        
        axes[1, i].set_title(f"Clean Model Predictions\n"
                           f"Spread: {result['spread_x']:.0f}x{result['spread_y']:.0f}px")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path("outputs/clean_model_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "clean_model_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved clean model test results: {output_path}")

if __name__ == "__main__":
    test_clean_model()

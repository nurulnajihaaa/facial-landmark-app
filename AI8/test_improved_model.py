"""
Test the improved HRNet model
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('.')

from facial_keypoints_app import FacialKeypointsInference

def test_improved_model():
    print("=== TESTING IMPROVED HRNET MODEL ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "checkpoints/improved_hrnet_best.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Improved model not found: {model_path}")
        print("Available checkpoints:")
        for ckpt in Path("checkpoints").glob("*.pth"):
            print(f"  - {ckpt}")
        return
    
    print(f"Loading improved model: {model_path}")
    inference = FacialKeypointsInference(model_path, device)
    print("✅ Improved model loaded!")
    
    # Test on multiple images
    test_images = [
        "dataset/train/images/17-900-Kid-Face-Side-View-Stock-Photos-Pictures-Royalty-Free-Images-iStock-Boy-head-side-view_jpg.rf.2c9487846c045251229cbbdfc8e0fb51.jpg",
        "dataset/train/images/1_jpg.rf.76115b6b11e4f88bad230d67d34a4ac7.jpg",
    ]
    
    results = []
    
    for i, image_path in enumerate(test_images):
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            continue
            
        print(f"\n--- Testing improved model on image {i+1}: {Path(image_path).name} ---")
        
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
        
        # Detailed keypoint analysis
        print("\n🔍 DETAILED KEYPOINT ANALYSIS:")
        keypoint_names = [
            'forehead', 'temple', 'eye_outer', 'eye_inner', 'nose_bridge',
            'nose_tip', 'nose_base', 'lip_top', 'lip_corner', 'lip_bottom',
            'chin', 'jaw_point', 'jaw_angle', 'ear'
        ]
        
        for j, (name, kp, conf) in enumerate(zip(keypoint_names, keypoints, confidence)):
            x, y = kp
            print(f"  {j:2d}. {name:12s}: ({x:5.1f}, {y:5.1f}) conf={conf:.3f}")
        
        # Anatomical checks
        print("\n🧬 ANATOMICAL VALIDATION:")
        
        # Check Y-coordinate progression (top to bottom)
        y_coords = keypoints[:, 1]
        
        # Expected rough ordering (allowing some flexibility)
        expected_order = [
            (0, "forehead"),      # Should be near top
            (5, "nose_tip"),      # Should be in middle
            (10, "chin"),         # Should be near bottom
        ]
        
        valid_anatomy = True
        for idx, name in expected_order:
            y_pos = y_coords[idx]
            print(f"  {name}: Y={y_pos:.1f}")
            
            # Basic anatomical checks
            if name == "forehead" and y_pos > 100:
                print(f"    ⚠️  Warning: {name} seems too low (Y={y_pos:.1f})")
                valid_anatomy = False
            elif name == "chin" and y_pos < 150:
                print(f"    ⚠️  Warning: {name} seems too high (Y={y_pos:.1f})")
                valid_anatomy = False
            else:
                print(f"    ✅ {name} position looks reasonable")
        
        # Check spread
        keypoint_spread_x = keypoints[:, 0].max() - keypoints[:, 0].min()
        keypoint_spread_y = keypoints[:, 1].max() - keypoints[:, 1].min()
        print(f"\n📏 KEYPOINT SPREAD:")
        print(f"  X-spread: {keypoint_spread_x:.1f}px")
        print(f"  Y-spread: {keypoint_spread_y:.1f}px")
        
        if keypoint_spread_x > 80 and keypoint_spread_y > 120:
            print("  ✅ GOOD: Reasonable facial coverage")
        else:
            print("  ⚠️  WARNING: Limited facial coverage")
        
        # Overall assessment
        mean_x, mean_y = keypoints.mean(axis=0)
        print(f"\n📊 OVERALL ASSESSMENT:")
        print(f"  Mean position: ({mean_x:.1f}, {mean_y:.1f})")
        print(f"  Confidence: {confidence.mean():.3f}")
        
        if valid_anatomy and keypoint_spread_x > 80 and keypoint_spread_y > 120:
            print("  🎉 EXCELLENT: Model predictions look anatomically correct!")
        elif keypoint_spread_x > 80 and keypoint_spread_y > 120:
            print("  ✅ GOOD: Reasonable keypoint distribution")
        else:
            print("  ❌ POOR: Model needs improvement")
        
        results.append({
            'image_path': image_path,
            'keypoints': keypoints,
            'confidence': confidence,
            'processed_image': processed_image,
            'spread_x': keypoint_spread_x,
            'spread_y': keypoint_spread_y,
            'valid_anatomy': valid_anatomy
        })
    
    # Create comprehensive visualization
    if results:
        create_improved_visualization(results)
    
    return results

def create_improved_visualization(results):
    """Create detailed visualization of improved model results"""
    
    n_images = len(results)
    fig, axes = plt.subplots(2, n_images, figsize=(6*n_images, 12))
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    keypoint_names = [
        'forehead', 'temple', 'eye_outer', 'eye_inner', 'nose_bridge',
        'nose_tip', 'nose_base', 'lip_top', 'lip_corner', 'lip_bottom',
        'chin', 'jaw_point', 'jaw_angle', 'ear'
    ]
    
    for i, result in enumerate(results):
        keypoints = result['keypoints']
        processed_image = result['processed_image']
        image_name = Path(result['image_path']).name
        
        # Original processed image
        axes[0, i].imshow(processed_image)
        axes[0, i].set_title(f"Image {i+1}: {image_name[:25]}...")
        axes[0, i].axis('off')
        
        # Image with improved keypoints
        axes[1, i].imshow(processed_image)
        
        # Color-coded keypoints by region
        colors = ['red', 'orange', 'yellow', 'green', 'cyan', 
                 'blue', 'purple', 'pink', 'brown', 'gray',
                 'olive', 'navy', 'maroon', 'lime']
        
        for j, (x, y) in enumerate(keypoints):
            color = colors[j % len(colors)]
            axes[1, i].scatter(x, y, c=color, s=80, alpha=0.8, edgecolors='white', linewidth=2)
            axes[1, i].annotate(f'{j}', (x, y), xytext=(3, 3), 
                              textcoords='offset points', fontsize=10, 
                              color='white', weight='bold',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
        
        # Add anatomical connections
        connections = [
            (2, 3),   # eye_outer to eye_inner
            (4, 5),   # nose_bridge to nose_tip
            (5, 6),   # nose_tip to nose_base
            (7, 8),   # lip_top to lip_corner
            (8, 9),   # lip_corner to lip_bottom
            (10, 11), # chin to jaw_point
            (11, 12), # jaw_point to jaw_angle
        ]
        
        for start, end in connections:
            x_start, y_start = keypoints[start]
            x_end, y_end = keypoints[end]
            axes[1, i].plot([x_start, x_end], [y_start, y_end], 
                           'white', linewidth=2, alpha=0.6)
        
        # Assessment text
        status = "🎉 EXCELLENT" if result['valid_anatomy'] and result['spread_x'] > 80 else "⚠️ NEEDS WORK"
        axes[1, i].set_title(f"Improved Model Results\n"
                           f"Spread: {result['spread_x']:.0f}×{result['spread_y']:.0f}px\n"
                           f"{status}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path("outputs/improved_model_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "improved_model_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved improved model test results: {output_path}")

if __name__ == "__main__":
    test_improved_model()

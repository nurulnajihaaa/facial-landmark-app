"""
Manually inspect the ground truth annotations to see if they make sense
"""
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def inspect_ground_truth():
    print("=== INSPECTING GROUND TRUTH ANNOTATIONS ===")
    
    # Load annotations
    with open('coco_annotations_clean.json', 'r') as f:
        coco_data = json.load(f)
    
    # Get first few samples
    images = coco_data['images'][:3]
    annotations = coco_data['annotations'][:3]
    
    keypoint_names = [
        'forehead', 'eyebrow_outer', 'eyebrow_inner', 'eye_outer',
        'eye_inner', 'nose_bridge', 'nose_tip', 'nose_bottom', 
        'lip_top', 'lip_corner', 'lip_bottom', 'chin_tip',
        'jaw_mid', 'jaw_angle'
    ]
    
    for i, (img_info, ann) in enumerate(zip(images, annotations)):
        print(f"\n--- Sample {i+1}: {Path(img_info['file_name']).name} ---")
        
        # Load image
        image_path = Path("dataset") / img_info['file_name']
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            continue
            
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Image size: {image_rgb.shape}")
        
        # Get keypoints and bbox
        keypoints = ann['keypoints']
        bbox = ann['bbox']
        
        print(f"Bbox: {bbox}")
        print(f"Raw keypoints (first 6): {keypoints[:6]}")
        
        # Parse keypoints
        kp_coords = []
        for j in range(0, len(keypoints), 3):
            x = keypoints[j]
            y = keypoints[j+1] 
            v = keypoints[j+2]
            kp_coords.append([x, y, v])
        
        kp_coords = np.array(kp_coords)
        print(f"Keypoint coordinates shape: {kp_coords.shape}")
        
        # Check if keypoints are within image bounds
        img_h, img_w = image_rgb.shape[:2]
        valid_keypoints = []
        
        for j, (x, y, v) in enumerate(kp_coords):
            in_bounds = 0 <= x < img_w and 0 <= y < img_h
            print(f"  {j:2d}. {keypoint_names[j]:12s}: ({x:6.1f}, {y:6.1f}) v={v} {'✓' if in_bounds else '✗'}")
            if v > 0 and in_bounds:
                valid_keypoints.append([x, y])
        
        valid_keypoints = np.array(valid_keypoints)
        
        # Visualize
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title("Original Image")
        plt.axis('off')
        
        # Image with bbox
        plt.subplot(1, 3, 2)
        plt.imshow(image_rgb)
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        plt.title("Image with Bbox")
        plt.axis('off')
        
        # Image with keypoints
        plt.subplot(1, 3, 3)
        plt.imshow(image_rgb)
        
        # Plot all keypoints
        for j, (x, y, v) in enumerate(kp_coords):
            if v > 0:  # Only plot visible keypoints
                color = 'red' if 0 <= x < img_w and 0 <= y < img_h else 'orange'
                plt.scatter(x, y, c=color, s=50, alpha=0.8)
                plt.annotate(str(j), (x, y), xytext=(3, 3), textcoords='offset points',
                           fontsize=8, color='yellow', weight='bold')
        
        plt.title("Ground Truth Keypoints")
        plt.axis('off')
        
        # Save visualization
        output_dir = Path("outputs/ground_truth_inspection")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_dir / f"ground_truth_sample_{i+1}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved inspection: {output_dir / f'ground_truth_sample_{i+1}.png'}")
        
        # Check if keypoints make anatomical sense
        print("--- Anatomical sense check ---")
        if len(valid_keypoints) > 0:
            min_x, max_x = valid_keypoints[:, 0].min(), valid_keypoints[:, 0].max()
            min_y, max_y = valid_keypoints[:, 1].min(), valid_keypoints[:, 1].max()
            spread_x = max_x - min_x
            spread_y = max_y - min_y
            
            print(f"Keypoint spread: {spread_x:.1f} x {spread_y:.1f} pixels")
            print(f"Keypoint bounds: X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]")
            
            # Check if keypoints are clustered vs spread out
            if spread_x > img_w * 0.2 and spread_y > img_h * 0.2:
                print("✅ Keypoints seem to span a reasonable face area")
            else:
                print("⚠️  Keypoints might be too clustered")
                
        else:
            print("❌ No valid keypoints found")

if __name__ == "__main__":
    inspect_ground_truth()

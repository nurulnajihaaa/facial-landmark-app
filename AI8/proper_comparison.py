"""
Proper coordinate transformation comparison
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
sys.path.append('.')

from facial_keypoints_app import FacialKeypointsInference

def transform_gt_to_processed_coords(gt_keypoints, bbox, input_size):
    """Transform ground truth keypoints to processed image coordinates"""
    x, y, w, h = bbox
    
    # Add padding (same as in dataset preprocessing)
    padding = 0.3
    x_pad = max(0, x - w * padding / 2)
    y_pad = max(0, y - h * padding / 2) 
    w_pad = w * (1 + padding)
    h_pad = h * (1 + padding)
    
    transformed_kps = []
    for i in range(0, len(gt_keypoints), 3):
        kp_x = gt_keypoints[i]
        kp_y = gt_keypoints[i+1]
        kp_v = gt_keypoints[i+2]
        
        if kp_v > 0:
            # Adjust relative to crop
            adj_x = kp_x - x_pad
            adj_y = kp_y - y_pad
            
            # Scale to input size
            scale_x = input_size[0] / w_pad
            scale_y = input_size[1] / h_pad
            
            final_x = adj_x * scale_x
            final_y = adj_y * scale_y
            
            transformed_kps.append([final_x, final_y])
        else:
            transformed_kps.append([np.nan, np.nan])
    
    return np.array(transformed_kps)

def proper_comparison():
    print("=== PROPER COORDINATE COMPARISON ===")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "checkpoints/clean_fixed_hrnet_best.pth"
    inference = FacialKeypointsInference(model_path, device)
    
    # Load ground truth
    with open('coco_annotations_clean.json', 'r') as f:
        coco_data = json.load(f)
    
    keypoint_names = [
        'forehead', 'eyebrow_outer', 'eyebrow_inner', 'eye_outer',
        'eye_inner', 'nose_bridge', 'nose_tip', 'nose_bottom', 
        'lip_top', 'lip_corner', 'lip_bottom', 'chin_tip',
        'jaw_mid', 'jaw_angle'
    ]
    
    # Test on first image
    img_info = coco_data['images'][0]
    ann = coco_data['annotations'][0]
    
    # Load image
    image_path = Path("dataset") / img_info['file_name']
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get ground truth
    gt_keypoints = ann['keypoints']
    gt_bbox = ann['bbox']
    
    # Transform ground truth to processed coordinates
    input_size = [256, 256]  # From config
    gt_transformed = transform_gt_to_processed_coords(gt_keypoints, gt_bbox, input_size)
    
    # Get model prediction (already in processed coordinates)
    pred_keypoints, confidence, processed_image = inference.predict(image_rgb, gt_bbox)
    
    print(f"Image: {Path(img_info['file_name']).name}")
    print(f"Original bbox: {gt_bbox}")
    print(f"Processed image shape: {processed_image.shape}")
    
    # Create proper comparison
    plt.figure(figsize=(20, 10))
    
    # Processed image with transformed ground truth
    plt.subplot(2, 4, 1)
    plt.imshow(processed_image)
    
    for i, (x, y) in enumerate(gt_transformed):
        if not np.isnan(x):
            plt.scatter(x, y, c='green', s=80, alpha=0.8, edgecolors='white', linewidth=1)
            plt.annotate(f"{i}", (x, y), xytext=(3, 3),
                        textcoords='offset points', fontsize=8, color='white', weight='bold')
    
    plt.title("Processed Image + Transformed GT (Green)")
    plt.axis('off')
    
    # Processed image with predictions
    plt.subplot(2, 4, 2)
    plt.imshow(processed_image)
    
    for i, (x, y) in enumerate(pred_keypoints):
        plt.scatter(x, y, c='red', s=80, alpha=0.8, edgecolors='white', linewidth=1)
        plt.annotate(f"{i}", (x, y), xytext=(3, 3),
                    textcoords='offset points', fontsize=8, color='yellow', weight='bold')
    
    plt.title("Processed Image + Predictions (Red)")
    plt.axis('off')
    
    # Overlay both
    plt.subplot(2, 4, 3)
    plt.imshow(processed_image)
    
    for i, ((gt_x, gt_y), (pred_x, pred_y)) in enumerate(zip(gt_transformed, pred_keypoints)):
        if not np.isnan(gt_x):
            plt.scatter(gt_x, gt_y, c='green', s=100, alpha=0.7, edgecolors='white', linewidth=2)
            plt.scatter(pred_x, pred_y, c='red', s=60, alpha=0.8, edgecolors='yellow', linewidth=1)
            
            # Draw error line
            plt.plot([gt_x, pred_x], [gt_y, pred_y], 'yellow', alpha=0.6, linewidth=1)
            
            plt.annotate(f"{i}", (gt_x, gt_y), xytext=(3, 3),
                        textcoords='offset points', fontsize=8, color='white', weight='bold')
    
    plt.title("Overlay: GT (Green) vs Pred (Red)")
    plt.axis('off')
    
    # Error magnitude visualization
    plt.subplot(2, 4, 4)
    errors = []
    valid_points = []
    
    for i, ((gt_x, gt_y), (pred_x, pred_y)) in enumerate(zip(gt_transformed, pred_keypoints)):
        if not np.isnan(gt_x):
            error = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
            errors.append(error)
            valid_points.append(i)
    
    if errors:
        plt.bar(range(len(errors)), errors, color='orange', alpha=0.7)
        plt.xticks(range(len(errors)), [keypoint_names[i][:4] for i in valid_points], rotation=45)
        plt.ylabel('Error (pixels)')
        plt.title('Per-Keypoint Errors')
        plt.grid(alpha=0.3)
    
    # Coordinate plots
    plt.subplot(2, 4, 5)
    valid_mask = ~np.isnan(gt_transformed).any(axis=1)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        plt.scatter(valid_indices, gt_transformed[valid_mask, 0], c='green', label='GT X', alpha=0.7, s=60)
        plt.scatter(range(len(pred_keypoints)), pred_keypoints[:, 0], c='red', label='Pred X', alpha=0.7, s=60)
        plt.xticks(range(len(keypoint_names)), [name[:4] for name in keypoint_names], rotation=45)
        plt.ylabel('X Coordinate')
        plt.title('X Coordinates (Same Space)')
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.subplot(2, 4, 6)
    if len(valid_indices) > 0:
        plt.scatter(valid_indices, gt_transformed[valid_mask, 1], c='green', label='GT Y', alpha=0.7, s=60)
        plt.scatter(range(len(pred_keypoints)), pred_keypoints[:, 1], c='red', label='Pred Y', alpha=0.7, s=60)
        plt.xticks(range(len(keypoint_names)), [name[:4] for name in keypoint_names], rotation=45)
        plt.ylabel('Y Coordinate')
        plt.title('Y Coordinates (Same Space)')
        plt.legend()
        plt.grid(alpha=0.3)
    
    # Statistics
    plt.subplot(2, 4, 7)
    if errors:
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        stats_text = f"""Error Statistics:
Mean: {mean_error:.1f} px
Max: {max_error:.1f} px  
Min: {min_error:.1f} px
Valid Points: {len(errors)}/{len(keypoint_names)}"""
        
        plt.text(0.1, 0.7, stats_text, fontsize=10, transform=plt.gca().transAxes, 
                verticalalignment='top', fontfamily='monospace')
        
        # Quality assessment
        if mean_error < 20:
            quality = "EXCELLENT"
            color = 'green'
        elif mean_error < 50:
            quality = "GOOD"
            color = 'orange'
        else:
            quality = "POOR"
            color = 'red'
            
        plt.text(0.1, 0.3, f"Quality: {quality}", fontsize=12, weight='bold',
                color=color, transform=plt.gca().transAxes)
        
        print(f"\n--- Error Analysis ---")
        print(f"Mean error: {mean_error:.1f} pixels")
        print(f"Error range: {min_error:.1f} - {max_error:.1f} pixels")
        print(f"Quality assessment: {quality}")
        
    plt.axis('off')
    
    # Detailed coordinate comparison
    plt.subplot(2, 4, 8)
    print(f"\n--- Detailed Comparison (Processed Space) ---")
    comparison_text = []
    
    for i, name in enumerate(keypoint_names[:8]):  # First 8 points
        if i < len(gt_transformed) and not np.isnan(gt_transformed[i]).any():
            gt_x, gt_y = gt_transformed[i]
            pred_x, pred_y = pred_keypoints[i]
            error = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
            
            print(f"{i:2d}. {name:12s}: GT=({gt_x:5.1f},{gt_y:5.1f}) Pred=({pred_x:5.1f},{pred_y:5.1f}) Err={error:4.1f}")
            comparison_text.append(f"{i}. {name[:8]}: {error:.1f}px")
    
    if comparison_text:
        plt.text(0.05, 0.95, '\n'.join(comparison_text), fontsize=8, 
                transform=plt.gca().transAxes, verticalalignment='top', fontfamily='monospace')
    
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("outputs/proper_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "proper_coordinate_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved proper comparison: {output_dir / 'proper_coordinate_comparison.png'}")

if __name__ == "__main__":
    proper_comparison()

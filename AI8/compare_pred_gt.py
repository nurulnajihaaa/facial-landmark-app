"""
Compare model predictions with ground truth to understand the mismatch
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

def compare_prediction_vs_ground_truth():
    print("=== COMPARING PREDICTIONS VS GROUND TRUTH ===")
    
    # Load clean model
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
    
    # Get ground truth keypoints
    gt_keypoints = ann['keypoints']
    gt_bbox = ann['bbox']
    
    gt_coords = []
    for i in range(0, len(gt_keypoints), 3):
        x, y, v = gt_keypoints[i], gt_keypoints[i+1], gt_keypoints[i+2]
        if v > 0:
            gt_coords.append([x, y])
        else:
            gt_coords.append([np.nan, np.nan])
    gt_coords = np.array(gt_coords)
    
    # Get model prediction
    pred_keypoints, confidence, processed_image = inference.predict(image_rgb, gt_bbox)
    
    print(f"Image: {Path(img_info['file_name']).name}")
    print(f"Ground truth shape: {gt_coords.shape}")
    print(f"Predicted shape: {pred_keypoints.shape}")
    
    # Create detailed comparison
    plt.figure(figsize=(20, 12))
    
    # Original image with ground truth
    plt.subplot(2, 3, 1)
    plt.imshow(image_rgb)
    
    # Draw bbox
    x, y, w, h = gt_bbox
    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
    plt.gca().add_patch(rect)
    
    # Plot ground truth keypoints
    for i, (x, y) in enumerate(gt_coords):
        if not np.isnan(x):
            plt.scatter(x, y, c='green', s=100, alpha=0.8)
            plt.annotate(f"{i}:{keypoint_names[i][:4]}", (x, y), xytext=(3, 3),
                        textcoords='offset points', fontsize=8, color='white', weight='bold')
    
    plt.title("Original Image + Ground Truth (Green)")
    plt.axis('off')
    
    # Processed image with predictions  
    plt.subplot(2, 3, 2)
    plt.imshow(processed_image)
    
    # Plot predicted keypoints
    for i, (x, y) in enumerate(pred_keypoints):
        plt.scatter(x, y, c='red', s=100, alpha=0.8)
        plt.annotate(f"{i}:{keypoint_names[i][:4]}", (x, y), xytext=(3, 3),
                    textcoords='offset points', fontsize=8, color='yellow', weight='bold')
    
    plt.title("Processed Image + Predictions (Red)")
    plt.axis('off')
    
    # Overlay comparison on processed image
    plt.subplot(2, 3, 3)
    plt.imshow(processed_image)
    
    # Need to transform ground truth to processed image coordinates
    # This is complex, so let's skip for now and focus on the coordinate analysis
    
    plt.title("Overlay (Complex - Skip for now)")
    plt.axis('off')
    
    # Coordinate analysis plots
    plt.subplot(2, 3, 4)
    valid_gt = ~np.isnan(gt_coords).any(axis=1)
    if valid_gt.any():
        plt.scatter(range(len(keypoint_names)), gt_coords[valid_gt, 0], 
                   c='green', label='GT X', alpha=0.7)
        plt.scatter(range(len(keypoint_names)), pred_keypoints[:, 0], 
                   c='red', label='Pred X', alpha=0.7)
    plt.xticks(range(len(keypoint_names)), [name[:4] for name in keypoint_names], rotation=45)
    plt.ylabel('X Coordinate')
    plt.title('X Coordinates Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 3, 5)
    if valid_gt.any():
        plt.scatter(range(len(keypoint_names)), gt_coords[valid_gt, 1], 
                   c='green', label='GT Y', alpha=0.7)
        plt.scatter(range(len(keypoint_names)), pred_keypoints[:, 1], 
                   c='red', label='Pred Y', alpha=0.7)
    plt.xticks(range(len(keypoint_names)), [name[:4] for name in keypoint_names], rotation=45)
    plt.ylabel('Y Coordinate')
    plt.title('Y Coordinates Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Error analysis
    plt.subplot(2, 3, 6)
    
    # Calculate coordinate-wise differences (rough approximation)
    print("\n--- Coordinate Analysis ---")
    print("Ground Truth coordinates (first 5):")
    for i in range(min(5, len(keypoint_names))):
        if not np.isnan(gt_coords[i]).any():
            print(f"  {i:2d}. {keypoint_names[i]:12s}: ({gt_coords[i, 0]:6.1f}, {gt_coords[i, 1]:6.1f})")
    
    print("\nPredicted coordinates (first 5):")
    for i in range(min(5, len(keypoint_names))):
        print(f"  {i:2d}. {keypoint_names[i]:12s}: ({pred_keypoints[i, 0]:6.1f}, {pred_keypoints[i, 1]:6.1f})")
    
    # Simple text summary
    plt.text(0.1, 0.8, "Coordinate Analysis:", fontsize=12, weight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"GT mean: ({np.nanmean(gt_coords[:, 0]):.1f}, {np.nanmean(gt_coords[:, 1]):.1f})", 
             fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Pred mean: ({np.mean(pred_keypoints[:, 0]):.1f}, {np.mean(pred_keypoints[:, 1]):.1f})", 
             fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, "Check console for detailed coords", fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("outputs/prediction_vs_ground_truth")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "prediction_vs_gt_detailed.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved detailed comparison: {output_dir / 'prediction_vs_gt_detailed.png'}")

if __name__ == "__main__":
    compare_prediction_vs_ground_truth()

"""
Test the newly trained fixed HRNet model
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from train_hrnet_facial import HRNetConfig, FacialKeypointsDataset
from fixed_hrnet import FixedHRNet

def test_new_model():
    print("=== TESTING NEWLY TRAINED FIXED MODEL ===")
    
    config = HRNetConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the new model
    model = FixedHRNet(config)
    
    # Load latest checkpoint
    checkpoint_path = "checkpoints/fixed_hrnet_epoch_19.pth"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Training loss was: {checkpoint['loss']:.6f}")
    else:
        print("No checkpoint found, using random weights")
    
    model = model.to(device)
    model.eval()
    
    # Load dataset 
    dataset = FacialKeypointsDataset('coco_annotations.json', 'dataset', config, is_train=False)
    sample = dataset[0]
    
    # Run inference
    with torch.no_grad():
        img_batch = sample['image'].unsqueeze(0).to(device)
        pred_heatmaps = model(img_batch)
        pred_heatmaps = pred_heatmaps.cpu()
        
    print(f"Model prediction shape: {pred_heatmaps.shape}")
    print(f"GT heatmap shape: {sample['heatmaps'].shape}")
    
    # Extract keypoints
    batch_size, num_joints, heatmap_h, heatmap_w = pred_heatmaps.shape
    heatmaps_reshaped = pred_heatmaps.reshape(batch_size, num_joints, -1)
    max_vals, max_indices = torch.max(heatmaps_reshaped, dim=2)
    
    # Convert to coordinates
    pred_coords = []
    gt_kpts = sample['keypoints'].numpy().reshape(-1, 3)
    
    print(f"\\n=== KEYPOINT COMPARISON ===")
    total_error = 0
    visible_count = 0
    
    for j in range(num_joints):
        # Predicted coordinates
        idx = max_indices[0, j].item()
        y = idx // heatmap_w
        x = idx % heatmap_w
        
        # Scale to input image size
        x_scaled = x * (config.input_size[0] / heatmap_w)
        y_scaled = y * (config.input_size[1] / heatmap_h)
        
        pred_coords.append([x_scaled, y_scaled])
        
        # Compare with GT
        gt_x, gt_y, gt_v = gt_kpts[j]
        if gt_v > 0:
            error = np.sqrt((gt_x - x_scaled)**2 + (gt_y - y_scaled)**2)
            total_error += error
            visible_count += 1
            
            confidence = max_vals[0, j].item()
            print(f"Joint {j}: GT=({gt_x:.1f},{gt_y:.1f}) Pred=({x_scaled:.1f},{y_scaled:.1f}) Error={error:.1f}px Conf={confidence:.3f}")
    
    if visible_count > 0:
        mean_error = total_error / visible_count
        print(f"\\nMean pixel error: {mean_error:.1f}px")
        
        if mean_error < 20:
            print("✅ EXCELLENT! Error < 20px")
        elif mean_error < 50:
            print("✅ GOOD! Error < 50px")
        elif mean_error < 100:
            print("⚠️  MODERATE. Error < 100px")
        else:
            print("❌ HIGH ERROR. Model needs more training")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Input image with GT and predicted keypoints
    img_vis = sample['image'].permute(1, 2, 0).numpy()
    img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_vis = np.clip(img_vis, 0, 1)
    
    axes[0,0].imshow(img_vis)
    for i, (x, y, v) in enumerate(gt_kpts):
        if v > 0:
            axes[0,0].plot(x, y, 'go', markersize=8, alpha=0.8)
            axes[0,0].text(x+3, y-3, str(i), color='green', fontsize=10, weight='bold')
    for i, (x, y) in enumerate(pred_coords):
        axes[0,0].plot(x, y, 'rx', markersize=8, alpha=0.8)
        axes[0,0].text(x+3, y+15, str(i), color='red', fontsize=10, weight='bold')
    axes[0,0].set_title(f"GT (green) vs Pred (red) - Error: {mean_error:.1f}px")
    axes[0,0].axis('off')
    
    # GT heatmaps sum
    gt_hm_sum = sample['heatmaps'].numpy().sum(axis=0)
    im1 = axes[0,1].imshow(gt_hm_sum, cmap='hot')
    axes[0,1].set_title("GT Heatmaps (Sum)")
    plt.colorbar(im1, ax=axes[0,1])
    
    # Predicted heatmaps sum
    pred_hm_sum = pred_heatmaps[0].numpy().sum(axis=0)
    im2 = axes[1,0].imshow(pred_hm_sum, cmap='hot')
    axes[1,0].set_title("Pred Heatmaps (Sum)")
    plt.colorbar(im2, ax=axes[1,0])
    
    # Individual joint comparison (first few joints)
    axes[1,1].axis('off')
    for j in range(min(4, num_joints)):
        gt_hm = sample['heatmaps'][j].numpy()
        pred_hm = pred_heatmaps[0, j].numpy()
        
        gt_max = gt_hm.max()
        pred_max = pred_hm.max()
        
        axes[1,1].text(0.1, 0.9 - j*0.2, f"Joint {j}: GT_max={gt_max:.3f}, Pred_max={pred_max:.3f}", 
                      transform=axes[1,1].transAxes, fontsize=10)
    
    axes[1,1].set_title("Heatmap Statistics")
    
    plt.tight_layout()
    
    output_dir = Path('outputs/diagnosis')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'new_model_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\\nSaved test results to {output_dir / 'new_model_test.png'}")
    
    return mean_error

if __name__ == '__main__':
    test_new_model()

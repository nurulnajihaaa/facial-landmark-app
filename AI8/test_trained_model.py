"""
Test with the actual trained model and fix coordinate system
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from train_hrnet_facial import HRNetConfig, FacialKeypointsDataset
from model_utils import load_model_safely

def test_trained_model():
    print("=== TESTING WITH TRAINED MODEL ===")
    
    config = HRNetConfig()
    
    # Load the actual trained model
    model_path = "checkpoints/finetune_head/checkpoint_epoch_11.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = load_model_safely(model_path, target_keypoints=14, device=device)
        print(f"✅ Loaded trained model from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load trained model: {e}")
        print("Trying alternative loading...")
        
        # Try direct checkpoint loading
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Try to load HRNet from hrnet_model
        try:
            from hrnet_model import get_pose_net, hrnet_w18_small_config
            model = get_pose_net(hrnet_w18_small_config, is_train=False)
            model.load_state_dict(state_dict, strict=False)
            print("✅ Loaded HRNet model")
        except:
            print("❌ Could not load HRNet, using SimpleHRNet")
            from train_hrnet_facial import SimpleHRNet
            model = SimpleHRNet(config)
            
    model = model.to(device)
    model.eval()
    
    # Load dataset 
    dataset = FacialKeypointsDataset('coco_annotations.json', 'dataset', config, is_train=False)
    sample = dataset[0]
    
    print(f"\n=== COORDINATE SYSTEM CHECK ===")
    gt_kpts = sample['keypoints'].numpy().reshape(-1, 3)
    print(f"GT keypoints range: x=[{gt_kpts[:,0].min():.1f}, {gt_kpts[:,0].max():.1f}], y=[{gt_kpts[:,1].min():.1f}, {gt_kpts[:,1].max():.1f}]")
    print(f"Input image size: {config.input_size}")
    print(f"Output heatmap size: {config.output_size}")
    
    # Run inference
    with torch.no_grad():
        img_batch = sample['image'].unsqueeze(0).to(device)
        pred_heatmaps = model(img_batch)
        
        if isinstance(pred_heatmaps, (list, tuple)):
            pred_heatmaps = pred_heatmaps[0]
            
        pred_heatmaps = pred_heatmaps.cpu()
        
    print(f"Model prediction shape: {pred_heatmaps.shape}")
    
    # Extract keypoints properly 
    batch_size, num_joints, heatmap_h, heatmap_w = pred_heatmaps.shape
    heatmaps_reshaped = pred_heatmaps.reshape(batch_size, num_joints, -1)
    max_vals, max_indices = torch.max(heatmaps_reshaped, dim=2)
    
    # Convert indices to coordinates
    pred_coords = []
    for j in range(num_joints):
        idx = max_indices[0, j].item()
        y = idx // heatmap_w
        x = idx % heatmap_w
        
        # Scale to input image size
        x_scaled = x * (config.input_size[0] / heatmap_w)
        y_scaled = y * (config.input_size[1] / heatmap_h)
        
        pred_coords.append([x_scaled, y_scaled])
        
        if j < 5:
            confidence = max_vals[0, j].item()
            print(f"Joint {j}: heatmap=({x},{y}) -> input=({x_scaled:.1f},{y_scaled:.1f}) conf={confidence:.3f}")
    
    # Compare with GT
    print(f"\n=== ACCURACY CHECK ===")
    total_error = 0
    for i in range(num_joints):
        gt_x, gt_y, gt_v = gt_kpts[i]
        pred_x, pred_y = pred_coords[i]
        if gt_v > 0:  # Only check visible joints
            error = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
            total_error += error
            if i < 5:
                print(f"Joint {i}: GT=({gt_x:.1f},{gt_y:.1f}) Pred=({pred_x:.1f},{pred_y:.1f}) Error={error:.1f}px")
    
    mean_error = total_error / num_joints
    print(f"Mean pixel error: {mean_error:.1f}px")
    
    if mean_error > 50:
        print("❌ VERY HIGH ERROR - Model predictions are wrong!")
    elif mean_error > 20:
        print("⚠️  HIGH ERROR - Model needs improvement")
    else:
        print("✅ Reasonable error range")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Input image with GT keypoints
    img_vis = sample['image'].permute(1, 2, 0).numpy()
    img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_vis = np.clip(img_vis, 0, 1)
    
    axes[0,0].imshow(img_vis)
    for i, (x, y, v) in enumerate(gt_kpts):
        if v > 0:
            axes[0,0].plot(x, y, 'go', markersize=6)
            axes[0,0].text(x+3, y-3, str(i), color='green', fontsize=8)
    axes[0,0].set_title("Input + GT Keypoints")
    axes[0,0].axis('off')
    
    # Input image with predicted keypoints
    axes[0,1].imshow(img_vis)
    for i, (x, y) in enumerate(pred_coords):
        axes[0,1].plot(x, y, 'ro', markersize=6)
        axes[0,1].text(x+3, y-3, str(i), color='red', fontsize=8)
    axes[0,1].set_title("Input + Predicted Keypoints")
    axes[0,1].axis('off')
    
    # Both overlaid
    axes[0,2].imshow(img_vis)
    for i, (x, y, v) in enumerate(gt_kpts):
        if v > 0:
            axes[0,2].plot(x, y, 'go', markersize=8, alpha=0.7, label='GT' if i==0 else "")
    for i, (x, y) in enumerate(pred_coords):
        axes[0,2].plot(x, y, 'rx', markersize=8, alpha=0.7, label='Pred' if i==0 else "")
    axes[0,2].set_title("GT (green) vs Pred (red)")
    axes[0,2].legend()
    axes[0,2].axis('off')
    
    # GT heatmaps sum
    gt_hm_sum = sample['heatmaps'].numpy().sum(axis=0)
    axes[1,0].imshow(gt_hm_sum, cmap='hot')
    axes[1,0].set_title("GT Heatmaps (Sum)")
    
    # Predicted heatmaps sum
    pred_hm_sum = pred_heatmaps[0].numpy().sum(axis=0)
    axes[1,1].imshow(pred_hm_sum, cmap='hot')
    axes[1,1].set_title("Pred Heatmaps (Sum)")
    
    # Difference
    diff = np.abs(gt_hm_sum - pred_hm_sum)
    im = axes[1,2].imshow(diff, cmap='hot')
    axes[1,2].set_title("Absolute Difference")
    plt.colorbar(im, ax=axes[1,2])
    
    plt.tight_layout()
    
    output_dir = Path('outputs/diagnosis')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'trained_model_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved comprehensive test to {output_dir / 'trained_model_test.png'}")
    
    return mean_error

if __name__ == '__main__':
    test_trained_model()

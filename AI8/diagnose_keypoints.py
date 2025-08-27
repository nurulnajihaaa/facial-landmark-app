"""
Diagnose fundamental issues with keypoint predictions
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from train_hrnet_facial import HRNetConfig, FacialKeypointsDataset, SimpleHRNet, JointsMSELoss

def diagnose_predictions():
    print("=== DIAGNOSIS: Keypoint Prediction Issues ===")
    
    # Load dataset and model
    config = HRNetConfig()
    dataset = FacialKeypointsDataset('coco_annotations.json', 'dataset', config, is_train=False)
    
    # Test with simple model first
    model = SimpleHRNet(config)
    model.eval()
    
    # Get one sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"GT heatmaps shape: {sample['heatmaps'].shape}")
    print(f"GT keypoints shape: {sample['keypoints'].shape}")
    
    # Check GT keypoints coordinates
    gt_kpts = sample['keypoints'].numpy().reshape(-1, 3)
    print(f"\nGround Truth Keypoints (first 5):")
    for i in range(min(5, len(gt_kpts))):
        x, y, v = gt_kpts[i]
        print(f"  Joint {i}: ({x:.1f}, {y:.1f}) visible={v}")
    
    # Run model forward
    with torch.no_grad():
        img_batch = sample['image'].unsqueeze(0)
        pred_heatmaps = model(img_batch)
        
    print(f"\nModel output shape: {pred_heatmaps.shape}")
    print(f"Expected heatmap shape: {sample['heatmaps'].shape}")
    
    # Check if output matches expected size
    if pred_heatmaps.shape[2:] != sample['heatmaps'].shape[1:]:
        print(f"❌ SIZE MISMATCH: pred={pred_heatmaps.shape[2:]} vs GT={sample['heatmaps'].shape[1:]}")
    else:
        print(f"✅ Output size matches GT heatmap size")
    
    # Extract predicted keypoints from heatmaps
    pred_kpts = []
    hm_np = pred_heatmaps[0].numpy()  # Remove batch dim
    
    print(f"\nPredicted heatmap analysis:")
    for j in range(config.num_keypoints):
        heatmap = hm_np[j]
        max_val = heatmap.max()
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y_pred, x_pred = max_idx
        
        # Scale back to input image coordinates (256x256)
        x_scaled = x_pred * (config.input_size[0] / config.output_size[0])
        y_scaled = y_pred * (config.input_size[1] / config.output_size[1])
        
        pred_kpts.append([x_scaled, y_scaled])
        
        if j < 5:  # Print first 5
            print(f"  Joint {j}: heatmap_max={max_val:.4f} at ({x_pred}, {y_pred}) -> scaled ({x_scaled:.1f}, {y_scaled:.1f})")
    
    # Compare GT vs Predicted
    print(f"\n=== GT vs PREDICTED COMPARISON ===")
    for i in range(min(5, len(gt_kpts))):
        gt_x, gt_y, gt_v = gt_kpts[i]
        pred_x, pred_y = pred_kpts[i]
        dist = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
        print(f"Joint {i}: GT=({gt_x:.1f},{gt_y:.1f}) Pred=({pred_x:.1f},{pred_y:.1f}) Dist={dist:.1f}px")
    
    # Visualize one sample
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    img_vis = sample['image'].permute(1, 2, 0).numpy()
    img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_vis = np.clip(img_vis, 0, 1)
    plt.imshow(img_vis)
    plt.title("Input Image")
    plt.axis('off')
    
    # GT heatmaps (sum)
    plt.subplot(1, 3, 2)
    gt_hm_sum = sample['heatmaps'].numpy().sum(axis=0)
    plt.imshow(gt_hm_sum, cmap='hot')
    plt.title("GT Heatmaps (Sum)")
    plt.colorbar()
    
    # Predicted heatmaps (sum)
    plt.subplot(1, 3, 3)
    pred_hm_sum = hm_np.sum(axis=0)
    plt.imshow(pred_hm_sum, cmap='hot')
    plt.title("Predicted Heatmaps (Sum)")
    plt.colorbar()
    
    plt.tight_layout()
    
    output_dir = Path('outputs/diagnosis')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'heatmap_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved diagnosis plot to {output_dir / 'heatmap_comparison.png'}")
    
    # Check if model was properly trained
    print(f"\n=== MODEL ANALYSIS ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check final layer weights
    final_weights = model.final_layer.weight
    print(f"Final layer weight shape: {final_weights.shape}")
    print(f"Final layer weight stats: mean={final_weights.mean():.4f}, std={final_weights.std():.4f}")
    
    if final_weights.std() < 0.01:
        print("❌ Final layer weights have very low variance - model may not be trained")
    else:
        print("✅ Final layer weights show reasonable variance")

if __name__ == '__main__':
    diagnose_predictions()

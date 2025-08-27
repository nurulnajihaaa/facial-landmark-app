"""
Debug the training data processing to understand why keypoints are wrong
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

sys.path.append('.')
from train_hrnet_facial import FacialKeypointsDataset, HRNetConfig
from facial_keypoints_app import FacialKeypointsInference

def debug_training_data():
    """Debug what the training data actually looks like"""
    print("=== DEBUGGING TRAINING DATA ===")
    
    # Load config and dataset
    config = HRNetConfig()
    dataset = FacialKeypointsDataset(
        coco_file='coco_annotations.json',
        image_root='dataset',
        config=config,
        is_train=False  # No augmentation for debugging
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Input size: {config.input_size}")
    print(f"Output size: {config.output_size}")
    
    # Check first few samples
    for i in range(3):
        print(f"\n--- Sample {i+1} ---")
        
        try:
            # Get sample from dataset
            sample = dataset[i]
            image_tensor = sample['image']
            heatmaps = sample['heatmaps']
            target_weight = sample['target_weight']
            keypoints = sample['keypoints']
            
            print(f"Image tensor shape: {image_tensor.shape}")
            print(f"Heatmaps shape: {heatmaps.shape}")
            print(f"Image ID: {sample['image_id']}")
            print(f"Raw keypoints: {sample['keypoints'][:6]}")  # First 3 keypoints
            
            # Convert tensor back to image for visualization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            image_np = image_tensor.permute(1, 2, 0).numpy()
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
            
            # Find keypoint locations from heatmaps
            predicted_keypoints = []
            for j in range(config.num_keypoints):
                heatmap = heatmaps[j]
                y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                
                # Scale back to input image size
                x_scaled = x_idx * config.input_size[0] / config.output_size[0]
                y_scaled = y_idx * config.input_size[1] / config.output_size[1]
                
                predicted_keypoints.append([x_scaled, y_scaled])
                print(f"  Keypoint {j}: heatmap=({x_idx}, {y_idx}) -> scaled=({x_scaled:.1f}, {y_scaled:.1f})")
            
            predicted_keypoints = np.array(predicted_keypoints)
            
            # Visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original processed image
            axes[0, 0].imshow(image_np)
            axes[0, 0].scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], 
                              c='red', s=50, alpha=0.8)
            for j, (x, y) in enumerate(predicted_keypoints):
                axes[0, 0].annotate(str(j), (x, y), xytext=(3, 3), 
                                  textcoords='offset points', fontsize=8, color='yellow')
            axes[0, 0].set_title(f"Sample {i+1}: Processed Image + Target Keypoints")
            axes[0, 0].axis('off')
            
            # Show first few heatmaps
            for j in range(min(5, config.num_keypoints)):
                row = (j // 3) + 0 if j < 3 else 1
                col = (j % 3) + 1 if j < 3 else j - 3
                
                if row < 2 and col < 3:
                    axes[row, col].imshow(heatmaps[j], cmap='hot')
                    axes[row, col].set_title(f"Keypoint {j} Heatmap")
                    axes[row, col].axis('off')
            
            # Save
            output_dir = Path("outputs/debug_training")
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_dir / f"training_sample_{i+1}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved debug image: {output_dir / f'training_sample_{i+1}.png'}")
            
        except Exception as e:
            print(f"❌ Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()

def compare_model_vs_training():
    """Compare what the model predicts vs what training data expects"""
    print("\n=== COMPARING MODEL vs TRAINING DATA ===")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "checkpoints/fixed_hrnet_epoch_19.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    inference = FacialKeypointsInference(model_path, device)
    print("✅ Model loaded")
    
    # Load dataset
    config = HRNetConfig()
    dataset = FacialKeypointsDataset(
        coco_file='coco_annotations.json',
        image_root='dataset',
        config=config,
        is_train=False
    )
    
    # Test on first image
    sample = dataset[0]
    image_tensor = sample['image']
    target_heatmaps = sample['heatmaps']
    target_weight = sample['target_weight']
    
    # Convert to RGB for model inference
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image_np = image_tensor.permute(1, 2, 0).numpy()
    image_np = image_np * std + mean
    image_np = np.clip(image_np, 0, 1)
    image_rgb = (image_np * 255).astype(np.uint8)
    
    # Get model prediction
    keypoints_pred, confidence_pred, processed_img = inference.predict(image_rgb)
    
    # Get target keypoints from heatmaps
    target_keypoints = []
    for j in range(config.num_keypoints):
        heatmap = target_heatmaps[j]
        y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Scale back to input image size
        x_scaled = x_idx * config.input_size[0] / config.output_size[0]
        y_scaled = y_idx * config.input_size[1] / config.output_size[1]
        target_keypoints.append([x_scaled, y_scaled])
    
    target_keypoints = np.array(target_keypoints)
    
    # Compare
    print(f"Target keypoints shape: {target_keypoints.shape}")
    print(f"Predicted keypoints shape: {keypoints_pred.shape}")
    print(f"Mean target keypoint: {np.mean(target_keypoints, axis=0)}")
    print(f"Mean predicted keypoint: {np.mean(keypoints_pred, axis=0)}")
    
    # Calculate errors
    errors = np.linalg.norm(target_keypoints - keypoints_pred, axis=1)
    print(f"Per-keypoint errors: {errors}")
    print(f"Mean error: {np.mean(errors):.2f} pixels")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.scatter(target_keypoints[:, 0], target_keypoints[:, 1], 
               c='green', s=100, alpha=0.7, label='Target')
    plt.title("Target Keypoints (Green)")
    plt.axis('off')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.imshow(image_rgb)
    plt.scatter(keypoints_pred[:, 0], keypoints_pred[:, 1], 
               c='red', s=100, alpha=0.7, label='Predicted')
    plt.title("Predicted Keypoints (Red)")
    plt.axis('off')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.imshow(image_rgb)
    plt.scatter(target_keypoints[:, 0], target_keypoints[:, 1], 
               c='green', s=100, alpha=0.7, label='Target')
    plt.scatter(keypoints_pred[:, 0], keypoints_pred[:, 1], 
               c='red', s=50, alpha=0.7, label='Predicted')
    # Draw error lines
    for i in range(len(target_keypoints)):
        plt.plot([target_keypoints[i, 0], keypoints_pred[i, 0]], 
                [target_keypoints[i, 1], keypoints_pred[i, 1]], 
                'yellow', alpha=0.5, linewidth=1)
    plt.title("Target vs Predicted")
    plt.axis('off')
    plt.legend()
    
    output_dir = Path("outputs/debug_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "model_vs_target_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparison: {output_dir / 'model_vs_target_comparison.png'}")

if __name__ == "__main__":
    debug_training_data()
    compare_model_vs_training()

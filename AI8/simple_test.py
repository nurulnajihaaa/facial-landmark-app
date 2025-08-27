"""
Simple test to verify the working facial keypoint detection
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def simple_test():
    print("=== SIMPLE FACIAL KEYPOINT TEST ===")
    
    try:
        from facial_keypoints_app import FacialKeypointsInference
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Try to load the working model
        model_path = "checkpoints/finetune_head/checkpoint_epoch_11.pth"
        if not Path(model_path).exists():
            model_path = "checkpoints/compat_hrnet_from_ckpt.pth"
        
        print(f"Loading model: {model_path}")
        inference = FacialKeypointsInference(model_path, device)
        print("✅ Model loaded successfully!")
        
        # Load a test image
        with open('coco_annotations.json', 'r') as f:
            coco_data = json.load(f)
        
        # Get first image
        image_info = coco_data['images'][0]
        image_file = image_info['file_name']
        
        # Find the image
        image_root = Path('dataset')
        image_path = None
        for split in ['train', 'valid', 'test']:
            candidate = image_root / split / 'images' / Path(image_file).name
            if candidate.exists():
                image_path = candidate
                break
        
        if not image_path:
            print(f"❌ Could not find test image: {image_file}")
            return
        
        print(f"Testing with: {image_path}")
        
        # Load and process image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get bbox if available
        bbox = None
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_info['id']:
                bbox = ann['bbox']
                break
        
        # Run inference
        if bbox:
            print(f"Using bbox: {bbox}")
            keypoints, confidence, processed_image = inference.predict(image_rgb, bbox)
        else:
            print("Using full image")
            keypoints, confidence, processed_image = inference.predict(image_rgb)
        
        print(f"✅ Prediction successful!")
        print(f"Keypoints shape: {keypoints.shape}")
        print(f"Mean confidence: {confidence.mean():.3f}")
        print(f"Confidence range: {confidence.min():.3f} - {confidence.max():.3f}")
        
        # Create simple visualization
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(image_rgb)
        
        # Plot keypoints with different colors based on confidence
        for i, ((x, y), conf) in enumerate(zip(keypoints, confidence)):
            color = 'green' if conf > 0.5 else 'yellow' if conf > 0.3 else 'red'
            plt.plot(x, y, 'o', color=color, markersize=8, alpha=0.8)
            plt.text(x+3, y-3, f"{i}", color=color, fontsize=10, weight='bold')
        
        plt.title(f"Detected Keypoints (Mean Conf: {confidence.mean():.3f})")
        plt.axis('off')
        
        # Save result
        output_dir = Path('outputs/simple_test')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'working_model_result.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved result to: {output_dir / 'working_model_result.png'}")
        
        # Print keypoint details
        print("\n=== KEYPOINT DETAILS ===")
        for i, ((x, y), conf) in enumerate(zip(keypoints, confidence)):
            print(f"Keypoint {i:2d}: ({x:6.1f}, {y:6.1f}) confidence: {conf:.3f}")
        
        # Check if results look reasonable
        if confidence.mean() > 0.3:
            print("✅ GOOD: Mean confidence > 0.3")
        else:
            print("⚠️  LOW: Mean confidence < 0.3")
        
        # Check keypoint spread
        x_range = keypoints[:, 0].max() - keypoints[:, 0].min()
        y_range = keypoints[:, 1].max() - keypoints[:, 1].min()
        
        print(f"Keypoint spread: X={x_range:.1f}px, Y={y_range:.1f}px")
        
        if x_range > 50 and y_range > 50:
            print("✅ GOOD: Keypoints spread across face")
        else:
            print("⚠️  POOR: Keypoints clustered together")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    simple_test()

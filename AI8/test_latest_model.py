"""
Test the latest trained FixedHRNet model
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def test_latest_model():
    print("=== TESTING LATEST FIXEDHRNET MODEL ===")
    
    try:
        from facial_keypoints_app import FacialKeypointsInference
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Use the latest trained model
        model_path = "checkpoints/fixed_hrnet_epoch_19.pth"
        if not Path(model_path).exists():
            print(f"❌ Latest model not found: {model_path}")
            print("Available checkpoints:")
            for ckpt in Path("checkpoints").glob("*.pth"):
                print(f"  - {ckpt}")
            return
        
        print(f"Loading latest model: {model_path}")
        inference = FacialKeypointsInference(model_path, device)
        print("✅ Latest model loaded successfully!")
        
        # Load a test image
        with open('coco_annotations.json', 'r') as f:
            coco_data = json.load(f)
        
        # Get first test image
        test_images = [img for img in coco_data['images'] if 'train' in img['file_name']][:3]
        
        for i, image_info in enumerate(test_images):
            print(f"\n--- Testing image {i+1}/3: {image_info['file_name']} ---")
            
            image_path = Path("dataset") / image_info['file_name']
            if not image_path.exists():
                print(f"❌ Image not found: {image_path}")
                continue
                
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
            print(f"Min/Max confidence: {confidence.min():.3f} / {confidence.max():.3f}")
            
            # Quick quality check
            keypoint_spread_x = keypoints[:, 0].max() - keypoints[:, 0].min()
            keypoint_spread_y = keypoints[:, 1].max() - keypoints[:, 1].min()
            print(f"Keypoint spread: X={keypoint_spread_x:.1f}px, Y={keypoint_spread_y:.1f}px")
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image_rgb)
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(processed_image)
            plt.title("Processed Image")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(processed_image)
            plt.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=50, alpha=0.8)
            for j, (x, y) in enumerate(keypoints):
                plt.annotate(str(j), (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='yellow', weight='bold')
            plt.title("Keypoint Predictions")
            plt.axis('off')
            
            # Save result
            output_dir = Path("outputs/latest_model_test")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"latest_model_test_{i+1}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved result to: {output_path}")
        
        print("\n=== SUMMARY ===")
        print("✅ Latest FixedHRNet model is working!")
        print("Check the output images to verify keypoint accuracy")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_latest_model()

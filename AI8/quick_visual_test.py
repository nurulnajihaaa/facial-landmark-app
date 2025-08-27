"""
Quick visual test of the latest model on a single image
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from facial_keypoints_app import FacialKeypointsInference, HRNetConfig

def test_single_image():
    # Load the latest model
    config = HRNetConfig()
    model_path = "checkpoints/finetune_head/checkpoint_epoch_11.pth"
    inference = FacialKeypointsInference(model_path, config)
    
    # Load COCO data to get an image
    with open('coco_annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    # Get first image
    image_info = coco_data['images'][0]
    image_file = image_info['file_name']
    
    # Try different paths
    image_root = Path('dataset')
    image_path = None
    for split in ['train', 'valid', 'test']:
        candidate = image_root / split / 'images' / Path(image_file).name
        if candidate.exists():
            image_path = candidate
            break
    
    if not image_path:
        print(f"Could not find image: {image_file}")
        return
    
    print(f"Testing with image: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get annotation for bbox
    ann = None
    for a in coco_data['annotations']:
        if a['image_id'] == image_info['id']:
            ann = a
            break
    
    if ann:
        bbox = ann['bbox']
        print(f"Using bbox: {bbox}")
        keypoints, confidence = inference.predict(image_rgb, bbox)
    else:
        print("No bbox found, using full image")
        keypoints, confidence = inference.predict(image_rgb)
    
    print(f"Predicted keypoints shape: {keypoints.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Mean confidence: {confidence.mean():.3f}")
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    
    # Plot predicted keypoints
    for i, (x, y) in enumerate(keypoints):
        plt.plot(x, y, 'ro', markersize=8)
        plt.text(x+5, y-5, str(i), color='red', fontsize=10)
    
    plt.title(f"Latest Model Predictions (Mean Conf: {confidence.mean():.3f})")
    plt.axis('off')
    
    # Save result
    output_dir = Path('outputs/quick_test')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'latest_model_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_dir / 'latest_model_test.png'}")

if __name__ == '__main__':
    test_single_image()

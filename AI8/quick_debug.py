"""
Simple debug to check if training data is actually different
"""
import sys
sys.path.append('.')
from train_hrnet_facial import FacialKeypointsDataset, HRNetConfig
import numpy as np

def quick_debug():
    print("=== QUICK TRAINING DATA DEBUG ===")
    
    config = HRNetConfig()
    dataset = FacialKeypointsDataset(
        coco_file='coco_annotations.json',
        image_root='dataset',
        config=config,
        is_train=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check first 5 samples
    for i in range(5):
        try:
            sample = dataset[i]
            image_id = sample['image_id']
            keypoints = sample['keypoints']
            heatmaps = sample['heatmaps']
            
            # Get peak locations from heatmaps
            peaks = []
            for j in range(min(3, config.num_keypoints)):  # First 3 keypoints only
                heatmap = heatmaps[j].numpy()
                y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                peaks.append((x_idx, y_idx))
            
            print(f"Sample {i+1}: ID={image_id}")
            print(f"  Raw keypoints: {keypoints[:6].numpy()}")
            print(f"  Heatmap peaks: {peaks}")
            print(f"  Heatmap max values: {[float(heatmaps[j].max()) for j in range(3)]}")
            print()
            
        except Exception as e:
            print(f"Error in sample {i}: {e}")

if __name__ == "__main__":
    quick_debug()

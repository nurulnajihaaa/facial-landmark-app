import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_annotations():
    """Load the annotations from the JSON file."""
    with open('coco_annotations_clean.json', 'r') as f:
        return json.load(f)

def analyze_keypoint_patterns():
    """Analyze the actual patterns of keypoints to understand what each kp represents."""
    data = load_annotations()
    
    # Group annotations by image for analysis
    images = {img['id']: img for img in data['images']}
    
    print("Analyzing keypoint patterns...")
    print("=" * 60)
    
    # Take first few annotations and visualize them
    sample_annotations = data['annotations'][:5]
    
    for i, ann in enumerate(sample_annotations):
        print(f"\n🔍 Sample {i+1}:")
        image_id = ann['image_id']
        image_info = images[image_id]
        print(f"Image: {image_info['file_name']}")
        
        # Parse keypoints
        keypoints = ann['keypoints']
        kp_data = []
        for j in range(0, len(keypoints), 3):
            x, y, vis = keypoints[j], keypoints[j+1], keypoints[j+2]
            if vis > 0:  # Visible keypoint
                kp_data.append((j//3 + 1, x, y))  # kp number (1-14), x, y
        
        # Sort by Y coordinate to see top-to-bottom pattern
        kp_data_sorted = sorted(kp_data, key=lambda x: x[2])  # Sort by Y
        
        print("Keypoints sorted by Y-coordinate (top to bottom):")
        for kp_num, x, y in kp_data_sorted:
            print(f"  kp{kp_num}: ({x:.1f}, {y:.1f})")
        
        # Also sort by X coordinate to see left-to-right pattern
        kp_data_sorted_x = sorted(kp_data, key=lambda x: x[1])  # Sort by X
        
        print("Keypoints sorted by X-coordinate (left to right):")
        for kp_num, x, y in kp_data_sorted_x:
            print(f"  kp{kp_num}: ({x:.1f}, {y:.1f})")
        
        # Load and visualize the image with keypoints
        image_path = Path("dataset") / image_info['file_name']
        if image_path.exists():
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            
            # Plot keypoints with labels
            colors = plt.cm.tab20(np.linspace(0, 1, 14))
            for kp_num, x, y in kp_data:
                plt.scatter(x, y, color=colors[kp_num-1], s=100, alpha=0.8)
                plt.annotate(f'kp{kp_num}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, 
                           color='white', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[kp_num-1], alpha=0.7))
            
            plt.title(f'Sample {i+1}: Keypoint Mapping Discovery\n{image_info["file_name"]}')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'keypoint_analysis_sample_{i+1}.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            break  # Just analyze first valid image for now

def create_keypoint_reference():
    """Create a reference mapping based on analysis."""
    print("\n" + "="*60)
    print("🎯 KEYPOINT MAPPING DISCOVERY")
    print("="*60)
    print("Based on the visual analysis above, we need to determine:")
    print("1. Which kp1-kp14 corresponds to which facial feature")
    print("2. The correct anatomical order for our model")
    print()
    print("Our current ASSUMED mapping was:")
    print("0: forehead, 1: eyebrow_outer, 2: eyebrow_inner, 3: eye_outer")
    print("4: eye_inner, 5: nose_tip, 6: nose_bridge, 7: mouth_corner")
    print("8: mouth_center, 9: lip_top, 10: lip_bottom, 11: chin_tip")
    print("12: chin_bottom, 13: jaw")
    print()
    print("But the actual kp1-kp14 mapping might be completely different!")
    print("We need to visually inspect the plotted keypoints to create the correct mapping.")

if __name__ == "__main__":
    analyze_keypoint_patterns()
    create_keypoint_reference()

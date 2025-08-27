"""
Filter out duplicate keypoint annotations and create clean training data
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def filter_duplicate_annotations():
    print("=== FILTERING DUPLICATE ANNOTATIONS ===")
    
    with open('coco_annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    # Group annotations by keypoint signature
    keypoint_groups = defaultdict(list)
    
    for ann in coco_data['annotations']:
        keypoints = ann['keypoints']
        # Create a signature from first few keypoints
        signature = tuple(keypoints[:12])  # First 4 keypoints (3 values each)
        keypoint_groups[signature].append(ann)
    
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Unique keypoint patterns: {len(keypoint_groups)}")
    
    # Show group sizes
    group_sizes = [len(group) for group in keypoint_groups.values()]
    print(f"Group sizes: {sorted(group_sizes, reverse=True)}")
    
    # Filter to keep only one annotation per unique keypoint pattern
    unique_annotations = []
    unique_image_ids = set()
    
    for signature, group in keypoint_groups.items():
        # Take the first annotation from each group
        ann = group[0]
        unique_annotations.append(ann)
        unique_image_ids.add(ann['image_id'])
        
        if len(group) > 1:
            print(f"Keypoint pattern {signature[:6]}... appears in {len(group)} images")
            print(f"  Keeping image_id: {ann['image_id']}")
            print(f"  Discarding: {[a['image_id'] for a in group[1:]]}")
    
    # Filter images to match unique annotations
    unique_images = [img for img in coco_data['images'] if img['id'] in unique_image_ids]
    
    print(f"\nFiltered results:")
    print(f"  Original: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    print(f"  Filtered: {len(unique_images)} images, {len(unique_annotations)} annotations")
    
    # Create new COCO data
    filtered_coco = {
        "info": coco_data["info"],
        "licenses": coco_data["licenses"],
        "images": unique_images,
        "annotations": unique_annotations,
        "categories": coco_data["categories"]
    }
    
    # Save filtered data
    with open('coco_annotations_clean.json', 'w') as f:
        json.dump(filtered_coco, f, indent=2)
    
    print(f"✅ Saved clean annotations to: coco_annotations_clean.json")
    
    return filtered_coco

def verify_clean_data():
    """Verify the clean data has unique keypoints"""
    print("\n=== VERIFYING CLEAN DATA ===")
    
    with open('coco_annotations_clean.json', 'r') as f:
        clean_data = json.load(f)
    
    # Check uniqueness
    keypoint_sets = []
    for ann in clean_data['annotations']:
        keypoints = tuple(ann['keypoints'])
        keypoint_sets.append(keypoints)
    
    unique_keypoints = set(keypoint_sets)
    
    print(f"Total clean annotations: {len(clean_data['annotations'])}")
    print(f"Unique keypoint patterns: {len(unique_keypoints)}")
    
    if len(clean_data['annotations']) == len(unique_keypoints):
        print("✅ All annotations have unique keypoints!")
    else:
        print(f"❌ Still have {len(clean_data['annotations']) - len(unique_keypoints)} duplicates")
    
    # Show sample diversity
    print("\n--- Sample keypoint diversity ---")
    for i, ann in enumerate(clean_data['annotations'][:5]):
        keypoints = ann['keypoints'][:6]  # First 3 keypoints
        print(f"Sample {i+1}: {keypoints}")

if __name__ == "__main__":
    filter_duplicate_annotations()
    verify_clean_data()

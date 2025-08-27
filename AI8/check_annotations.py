"""
Check if all training samples have the same keypoints (bug detection)
"""
import json
from pathlib import Path

def check_annotations():
    print("=== CHECKING ANNOTATION UNIQUENESS ===")
    
    with open('coco_annotations.json', 'r') as f:
        coco_data = json.load(f)
    
    annotations = coco_data['annotations']
    
    print(f"Total annotations: {len(annotations)}")
    
    # Check if all keypoints are the same
    first_keypoints = annotations[0]['keypoints']
    all_same = True
    
    for i, ann in enumerate(annotations[:10]):  # Check first 10
        current_keypoints = ann['keypoints']
        if current_keypoints != first_keypoints:
            all_same = False
            print(f"Annotation {i} has different keypoints")
            break
        else:
            print(f"Annotation {i}: image_id={ann['image_id']}, keypoints={current_keypoints[:6]}...")
    
    if all_same:
        print("❌ BUG FOUND: All annotations have identical keypoints!")
        print(f"All keypoints: {first_keypoints}")
    else:
        print("✅ Annotations have different keypoints (normal)")
    
    # Check image IDs
    image_ids = [ann['image_id'] for ann in annotations]
    unique_image_ids = set(image_ids)
    
    print(f"Total image IDs: {len(image_ids)}")
    print(f"Unique image IDs: {len(unique_image_ids)}")
    
    if len(image_ids) != len(unique_image_ids):
        print("⚠️  Some images have multiple annotations")
    
    # Show mapping between images and annotations
    print("\n--- Image to Annotation mapping (first 5) ---")
    for i, img in enumerate(coco_data['images'][:5]):
        img_id = img['id']
        img_name = img['file_name']
        
        # Find annotations for this image
        img_annotations = [ann for ann in annotations if ann['image_id'] == img_id]
        
        print(f"Image {img_id}: {Path(img_name).name}")
        for j, ann in enumerate(img_annotations):
            bbox = ann['bbox']
            keypoints = ann['keypoints'][:6]  # First 3 keypoints only
            print(f"  Annotation {j}: bbox={bbox}, keypoints={keypoints}...")

if __name__ == "__main__":
    check_annotations()

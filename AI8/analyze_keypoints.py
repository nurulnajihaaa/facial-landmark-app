"""
Check if the keypoint definitions in our data make anatomical sense
"""
import json
import numpy as np

def analyze_keypoint_definitions():
    print("=== ANALYZING KEYPOINT DEFINITIONS ===")
    
    with open('coco_annotations_clean.json', 'r') as f:
        coco_data = json.load(f)
    
    our_keypoint_names = [
        'forehead', 'eyebrow_outer', 'eyebrow_inner', 'eye_outer',
        'eye_inner', 'nose_bridge', 'nose_tip', 'nose_bottom', 
        'lip_top', 'lip_corner', 'lip_bottom', 'chin_tip',
        'jaw_mid', 'jaw_angle'
    ]
    
    print("Our keypoint definitions:")
    for i, name in enumerate(our_keypoint_names):
        print(f"  {i:2d}. {name}")
    
    print("\nExpected anatomical order (top to bottom, left to right):")
    expected_order = [
        "forehead (top)",
        "eyebrow_inner (closer to nose)", 
        "eyebrow_outer (closer to ear)",
        "eye_inner (closer to nose)",
        "eye_outer (closer to ear)", 
        "nose_bridge",
        "nose_tip",
        "nose_bottom", 
        "lip_top",
        "lip_corner",
        "lip_bottom",
        "chin_tip",
        "jaw_mid",
        "jaw_angle (bottom)"
    ]
    
    for i, desc in enumerate(expected_order):
        print(f"  {i:2d}. {desc}")
    
    # Analyze first few samples to see Y-coordinate progression
    print("\n=== Y-COORDINATE ANALYSIS ===")
    
    for sample_idx in range(min(3, len(coco_data['annotations']))):
        ann = coco_data['annotations'][sample_idx]
        keypoints = ann['keypoints']
        
        print(f"\nSample {sample_idx + 1}:")
        y_coords = []
        
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
            if v > 0:
                y_coords.append((i//3, our_keypoint_names[i//3], y))
                print(f"  {i//3:2d}. {our_keypoint_names[i//3]:12s}: Y={y:6.1f}")
        
        # Check if Y coordinates make anatomical sense
        print("  Anatomical sense check:")
        
        # Extract key landmarks
        landmark_ys = {}
        for idx, name, y in y_coords:
            landmark_ys[name] = y
        
        # Check expected relationships
        checks = [
            ("forehead should be above eyebrows", 
             landmark_ys.get('forehead', 999) < min(landmark_ys.get('eyebrow_outer', 0), landmark_ys.get('eyebrow_inner', 0))),
            ("eyebrows should be above eyes",
             max(landmark_ys.get('eyebrow_outer', 0), landmark_ys.get('eyebrow_inner', 0)) < min(landmark_ys.get('eye_outer', 999), landmark_ys.get('eye_inner', 999))),
            ("eyes should be above nose",
             max(landmark_ys.get('eye_outer', 0), landmark_ys.get('eye_inner', 0)) < landmark_ys.get('nose_tip', 999)),
            ("nose should be above lips", 
             landmark_ys.get('nose_tip', 0) < landmark_ys.get('lip_top', 999)),
            ("lips should be above chin",
             landmark_ys.get('lip_bottom', 0) < landmark_ys.get('chin_tip', 999))
        ]
        
        for check_desc, result in checks:
            status = "✓" if result else "✗"
            print(f"    {status} {check_desc}")
    
    print("\n=== CONCLUSION ===")
    print("If many anatomical checks fail, the keypoint definitions or")
    print("annotations might be incorrect or in a different order than expected.")

if __name__ == "__main__":
    analyze_keypoint_definitions()

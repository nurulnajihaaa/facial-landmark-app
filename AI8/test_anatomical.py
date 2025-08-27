"""
Final test with proper keypoint reordering based on ground truth analysis
"""
import sys
sys.path.append('.')
from facial_keypoints_app import FacialKeypointsInference
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def test_with_anatomical_reordering():
    """
    Test model with proper anatomical reordering based on our ground truth discovery.
    
    From our ground truth analysis, we found:
    kp1 = forehead, kp2 = temple, kp3 = eye_outer, kp4 = eye_inner, etc.
    
    The model was trained to predict in this order, so predictions[0] = forehead prediction, etc.
    """
    print("=== TESTING WITH ANATOMICAL REORDERING ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "checkpoints/clean_fixed_hrnet_best.pth"
    
    # Load model
    inference = FacialKeypointsInference(model_path, device)
    print("✅ Model loaded")
    
    # Test image
    test_image = "dataset/train/images/17-900-Kid-Face-Side-View-Stock-Photos-Pictures-Royalty-Free-Images-iStock-Boy-head-side-view_jpg.rf.2c9487846c045251229cbbdfc8e0fb51.jpg"
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found")
        return
    
    # Load and predict
    image = cv2.imread(test_image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints, confidence, processed_image = inference.predict(image_rgb)
    
    # The keypoints are already in the correct order based on our ground truth analysis!
    # kp0 = forehead, kp1 = temple, kp2 = eye_outer, etc.
    
    # Correct anatomical names (matching our ground truth discovery)
    anatomical_names = [
        'forehead',      # kp0 - should be at top of head
        'temple',        # kp1 - side of forehead  
        'eye_outer',     # kp2 - outer eye corner
        'eye_inner',     # kp3 - inner eye corner
        'nose_bridge',   # kp4 - nose bridge
        'nose_tip',      # kp5 - nose tip
        'nose_base',     # kp6 - nose base/nostril
        'lip_top',       # kp7 - upper lip
        'lip_corner',    # kp8 - mouth corner
        'lip_bottom',    # kp9 - lower lip
        'chin',          # kp10 - chin
        'jaw_point',     # kp11 - jaw point
        'jaw_angle',     # kp12 - jaw angle
        'ear'            # kp13 - ear
    ]
    
    print(f"\n=== ANATOMICAL ANALYSIS ===")
    print(f"Image shape: {processed_image.shape}")
    
    # Show predictions with anatomical context
    for i, (name, kp, conf) in enumerate(zip(anatomical_names, keypoints, confidence)):
        x, y = kp
        print(f"  {i:2d}. {name:12s}: ({x:6.1f}, {y:6.1f}) conf={conf:.3f}")
    
    # Anatomical validation
    print(f"\n=== ANATOMICAL CHECKS ===")
    
    # Check if the predictions make anatomical sense
    forehead_y = keypoints[0][1]  # kp0 = forehead
    nose_tip_y = keypoints[5][1]  # kp5 = nose_tip  
    chin_y = keypoints[10][1]     # kp10 = chin
    
    if forehead_y < nose_tip_y < chin_y:
        print("✅ GOOD: Basic face structure (forehead → nose → chin)")
    else:
        print("❌ Issue with face structure:")
        print(f"  forehead: {forehead_y:.1f}, nose_tip: {nose_tip_y:.1f}, chin: {chin_y:.1f}")
    
    # Check eye positioning for profile
    eye_outer_x = keypoints[2][0]  # kp2
    eye_inner_x = keypoints[3][0]  # kp3
    
    if eye_outer_x < eye_inner_x:
        print("✅ GOOD: Eye positioning for profile view")
    else:
        print(f"⚠️  Eye positioning: outer={eye_outer_x:.1f}, inner={eye_inner_x:.1f}")
    
    # Check nose structure
    nose_bridge_y = keypoints[4][1]  # kp4
    nose_base_y = keypoints[6][1]    # kp6
    
    if nose_bridge_y < nose_tip_y < nose_base_y:
        print("✅ GOOD: Nose structure (bridge → tip → base)")
    else:
        print(f"⚠️  Nose structure: bridge={nose_bridge_y:.1f}, tip={nose_tip_y:.1f}, base={nose_base_y:.1f}")
    
    # Check lip positioning
    lip_top_y = keypoints[7][1]    # kp7
    lip_bottom_y = keypoints[9][1] # kp9
    
    if lip_top_y < lip_bottom_y:
        print("✅ GOOD: Lip structure (top → bottom)")
    else:
        print(f"⚠️  Lip structure: top={lip_top_y:.1f}, bottom={lip_bottom_y:.1f}")
    
    # Visualization with anatomical labels
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(processed_image)
    plt.title("Original Processed Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed_image)
    
    # Color code by facial region
    region_colors = {
        'forehead': 'red', 'temple': 'red',
        'eye_outer': 'blue', 'eye_inner': 'blue', 
        'nose_bridge': 'green', 'nose_tip': 'green', 'nose_base': 'green',
        'lip_top': 'purple', 'lip_corner': 'purple', 'lip_bottom': 'purple',
        'chin': 'orange', 'jaw_point': 'orange', 'jaw_angle': 'orange',
        'ear': 'brown'
    }
    
    for i, (kp, name) in enumerate(zip(keypoints, anatomical_names)):
        x, y = kp
        color = region_colors.get(name, 'gray')
        plt.scatter(x, y, c=color, s=100, alpha=0.8, edgecolors='white', linewidth=2)
        plt.annotate(f"{i}:{name}", (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, 
                    color='white', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    
    plt.title("Anatomically Labeled Keypoints")
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("outputs/anatomical_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "anatomical_keypoints.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved anatomical analysis: {output_path}")
    
    # Return keypoints with correct anatomical labels for further analysis
    return keypoints, anatomical_names, processed_image

if __name__ == "__main__":
    test_with_anatomical_reordering()

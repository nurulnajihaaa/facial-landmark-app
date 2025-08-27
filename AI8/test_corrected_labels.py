"""
Test the model with correct keypoint interpretations
"""
import sys
sys.path.append('.')
from facial_keypoints_app import FacialKeypointsInference
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_with_correct_labels():
    print("=== TESTING WITH CORRECT KEYPOINT LABELS ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "checkpoints/clean_fixed_hrnet_best.pth"
    
    # Load model with corrected labels
    inference = FacialKeypointsInference(model_path, device)
    print("✅ Model loaded with correct keypoint labels")
    
    # Print the keypoint names for verification
    print("\nKeypoint Labels:")
    for i, name in enumerate(inference.keypoint_names):
        print(f"  {i:2d}: {name}")
    
    # Test on sample image
    test_image = "dataset/train/images/17-900-Kid-Face-Side-View-Stock-Photos-Pictures-Royalty-Free-Images-iStock-Boy-head-side-view_jpg.rf.2c9487846c045251229cbbdfc8e0fb51.jpg"
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    # Load and predict
    image = cv2.imread(test_image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    keypoints, confidence, processed_image = inference.predict(image_rgb)
    
    print(f"\n=== PREDICTION RESULTS ===")
    print(f"Image shape: {processed_image.shape}")
    print(f"Keypoints shape: {keypoints.shape}")
    
    # Show keypoint positions with correct labels
    print(f"\nKeypoint Positions:")
    for i, (name, kp, conf) in enumerate(zip(inference.keypoint_names, keypoints, confidence)):
        x, y = kp
        print(f"  {i:2d}. {name:12s}: ({x:6.1f}, {y:6.1f}) confidence={conf:.3f}")
    
    # Anatomical checks with correct labels
    print(f"\n=== ANATOMICAL VALIDATION ===")
    
    # Get keypoint indices
    def get_kp_idx(name):
        return inference.keypoint_names.index(name)
    
    forehead_y = keypoints[get_kp_idx('forehead')][1]
    chin_y = keypoints[get_kp_idx('chin')][1]
    nose_tip_y = keypoints[get_kp_idx('nose_tip')][1]
    lip_top_y = keypoints[get_kp_idx('lip_top')][1]
    
    # Check vertical ordering
    if forehead_y < nose_tip_y < lip_top_y < chin_y:
        print("✅ GOOD: Vertical face structure is correct (forehead → nose → lips → chin)")
    else:
        print("❌ BAD: Vertical face structure is wrong")
        print(f"  forehead: {forehead_y:.1f}, nose_tip: {nose_tip_y:.1f}, lip_top: {lip_top_y:.1f}, chin: {chin_y:.1f}")
    
    # Check eye positions
    eye_outer_x = keypoints[get_kp_idx('eye_outer')][0]
    eye_inner_x = keypoints[get_kp_idx('eye_inner')][0]
    
    if eye_outer_x < eye_inner_x:  # For profile view, outer should be more left
        print("✅ GOOD: Eye positioning looks correct for profile view")
    else:
        print("❌ BAD: Eye positioning might be wrong")
        print(f"  eye_outer: {eye_outer_x:.1f}, eye_inner: {eye_inner_x:.1f}")
    
    # Check nose structure
    nose_bridge_y = keypoints[get_kp_idx('nose_bridge')][1]
    nose_base_y = keypoints[get_kp_idx('nose_base')][1]
    
    if nose_bridge_y < nose_tip_y < nose_base_y:
        print("✅ GOOD: Nose structure is correct (bridge → tip → base)")
    else:
        print("❌ BAD: Nose structure might be wrong")
        print(f"  nose_bridge: {nose_bridge_y:.1f}, nose_tip: {nose_tip_y:.1f}, nose_base: {nose_base_y:.1f}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(processed_image)
    plt.title("Processed Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed_image)
    
    # Plot keypoints with correct labels
    colors = plt.cm.Set3(np.linspace(0, 1, len(keypoints)))
    for i, (kp, color, name) in enumerate(zip(keypoints, colors, inference.keypoint_names)):
        x, y = kp
        plt.scatter(x, y, c=[color], s=100, alpha=0.8, edgecolors='black', linewidth=1)
        plt.annotate(f"{i}: {name}", (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, 
                    color='white', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    plt.title("Keypoints with Correct Labels")
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("outputs/corrected_labels_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "corrected_labels_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved test result: {output_path}")

if __name__ == "__main__":
    import torch
    test_with_correct_labels()

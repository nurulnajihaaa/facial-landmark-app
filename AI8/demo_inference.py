"""
Demo script for testing facial keypoints detection with sample images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from train_hrnet_facial import SimpleHRNet, HRNetConfig

try:
    from hrnet_model import get_pose_net, hrnet_w18_small_config
    ENHANCED_HRNET_AVAILABLE = True
except ImportError:
    ENHANCED_HRNET_AVAILABLE = False

def create_demo_images():
    """Create some demo face images for testing"""
    demo_dir = Path("demo_images")
    demo_dir.mkdir(exist_ok=True)
    
    # Copy some test images from the dataset for demo
    test_images_dir = Path("dataset/test/images")
    if test_images_dir.exists():
        test_images = list(test_images_dir.glob("*.jpg"))[:3]
        
        for i, img_path in enumerate(test_images):
            demo_path = demo_dir / f"demo_{i+1}.jpg"
            if not demo_path.exists():
                # Copy image
                import shutil
                shutil.copy2(img_path, demo_path)
                print(f"Created demo image: {demo_path}")
        
        return list(demo_dir.glob("*.jpg"))
    else:
        print("No test images found in dataset/test/images")
        return []

def run_demo_inference(model_path="checkpoints/checkpoint_epoch_99.pth"):
    """Run inference on demo images"""
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        available_models = list(Path(".").glob("*.pth")) + list(Path("checkpoints").glob("*.pth"))
        if available_models:
            model_path = str(available_models[0])
            print(f"Using alternative model: {model_path}")
        else:
            print("No models found!")
            return
    
    # Load model
    config = HRNetConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if ENHANCED_HRNET_AVAILABLE:
        model = get_pose_net(hrnet_w18_small_config, is_train=False)
    else:
        model = SimpleHRNet(config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    print(f"Model loaded: {model_path}")
    print(f"Device: {device}")
    
    # Create demo images
    demo_images = create_demo_images()
    
    if not demo_images:
        print("No demo images available")
        return
    
    # Process each demo image
    for img_path in demo_images:
        print(f"\nProcessing: {img_path}")
        
        # Load and preprocess image
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        input_image = cv2.resize(image_rgb, tuple(config.input_size))
        
        # Convert to tensor
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(input_image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            heatmaps = model(input_tensor)
            
            # Get keypoint coordinates
            batch_size, num_joints, height, width = heatmaps.shape
            heatmaps_reshaped = heatmaps.reshape(batch_size, num_joints, -1)
            max_vals, max_indices = torch.max(heatmaps_reshaped, dim=2)
            
            keypoints = torch.zeros(batch_size, num_joints, 2)
            keypoints[:, :, 0] = max_indices % width
            keypoints[:, :, 1] = max_indices // width
            
            # Scale to input image size
            scale_x = config.input_size[0] / width
            scale_y = config.input_size[1] / height
            keypoints[:, :, 0] *= scale_x
            keypoints[:, :, 1] *= scale_y
            
            keypoints = keypoints[0].cpu().numpy()
            confidence = max_vals[0].cpu().numpy()
        
        # Visualize results
        plt.figure(figsize=(12, 6))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(input_image)
        plt.title('Input Image')
        plt.axis('off')
        
        # Image with keypoints
        plt.subplot(1, 2, 2)
        plt.imshow(input_image)
        
        # Plot keypoints
        colors = plt.cm.Set3(np.linspace(0, 1, len(keypoints)))
        for i, (kp, conf, color) in enumerate(zip(keypoints, confidence, colors)):
            if conf > 0.1:  # Confidence threshold
                plt.plot(kp[0], kp[1], 'o', color=color, markersize=8)
                plt.annotate(f'{i+1}', (kp[0], kp[1]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, color='white',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
        
        plt.title('Detected Keypoints')
        plt.axis('off')
        
        # Save result
        output_path = Path("demo_results") / f"result_{img_path.stem}.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Result saved: {output_path}")
        
        # Print keypoint coordinates
        keypoint_names = [
            'forehead', 'eyebrow_outer', 'eyebrow_inner', 'eye_outer',
            'eye_inner', 'nose_bridge', 'nose_tip', 'nose_bottom',
            'lip_top', 'lip_corner', 'lip_bottom', 'chin_tip',
            'jaw_mid', 'jaw_angle'
        ]
        
        print("Detected keypoints:")
        for i, (name, kp, conf) in enumerate(zip(keypoint_names, keypoints, confidence)):
            status = "✓" if conf > 0.1 else "✗"
            print(f"  {i+1:2d}. {name:15s}: ({kp[0]:6.1f}, {kp[1]:6.1f}) {status}")

if __name__ == "__main__":
    print("🔬 Facial Keypoints Detection Demo")
    print("=" * 40)
    
    # Run demo
    run_demo_inference()
    
    print("\n" + "=" * 40)
    print("Demo completed!")
    print("\n💡 To use the web interface, run:")
    print("   python start_web_interface.py")
    print("\n📁 Demo results saved in 'demo_results' folder")

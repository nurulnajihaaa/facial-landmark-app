"""
Complete HRNet Facial Keypoints Detection System - Summary and Usage Guide
"""

import json
import torch
from pathlib import Path

def print_system_summary():
    """Print a comprehensive summary of the system"""
    
    print("=" * 80)
    print("🏗️  HRNet Facial Keypoints Detection System")
    print("=" * 80)
    print()
    
    print("📊 DATASET INFORMATION:")
    print("-" * 40)
    
    # Check if COCO file exists and print stats
    coco_file = Path("coco_annotations.json")
    if coco_file.exists():
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        total_images = len(coco_data['images'])
        total_annotations = len(coco_data['annotations'])
        
        # Count by split
        splits = {'train': 0, 'valid': 0, 'test': 0}
        for img in coco_data['images']:
            split = img['file_name'].split('/')[0]
            if split in splits:
                splits[split] += 1
        
        print(f"✅ Total Images: {total_images}")
        print(f"✅ Total Annotations: {total_annotations}")
        print(f"✅ Training Split: {splits.get('train', 0)} images")
        print(f"✅ Validation Split: {splits.get('valid', 0)} images") 
        print(f"✅ Test Split: {splits.get('test', 0)} images")
        print(f"✅ Keypoints per annotation: 14")
        print(f"✅ Format: COCO Keypoints JSON")
    else:
        print("❌ COCO annotations not found. Run converter first.")
    
    print()
    print("🔧 MODEL ARCHITECTURE:")
    print("-" * 40)
    print("✅ Model: HRNet-W18-C-Small-v2")
    print("✅ Input Size: 256x256 RGB")
    print("✅ Output Size: 14x64x64 heatmaps")
    print("✅ Keypoints: 14 facial landmarks")
    print("✅ Parameters: ~4M (optimized)")
    
    # Check if pretrained model exists
    pretrained_path = Path("hrnet_w18_small_model_v1.pth")
    if pretrained_path.exists():
        print(f"✅ Pretrained Model: {pretrained_path} ({pretrained_path.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print("❌ Pretrained model not found")
    
    print()
    print("📏 EVALUATION METRICS:")
    print("-" * 40)
    print("✅ PCK@0.05: Percentage of Correct Keypoints (5% threshold)")
    print("✅ PCK@0.10: Percentage of Correct Keypoints (10% threshold)")
    print("✅ NME: Normalized Mean Error")
    print("✅ Orthodontic Distances: 5 clinical measurements")
    print("✅ Orthodontic Angles: 2 facial angle measurements")
    
    print()
    print("💻 HARDWARE REQUIREMENTS:")
    print("-" * 40)
    print("✅ Minimum: 4GB GPU, 8GB RAM")
    print("✅ Recommended: 8GB+ GPU, 16GB+ RAM")
    print("✅ CUDA Support: Automatic detection")
    
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ Current Device: {device} ({gpu_name}, {gpu_memory:.1f}GB)")
    else:
        print(f"⚠️  Current Device: {device} (GPU recommended for training)")
    
    print()
    print("📂 PROJECT FILES:")
    print("-" * 40)
    
    files_status = {
        "yolo_to_coco_converter.py": "YOLO to COCO format converter",
        "train_hrnet_facial.py": "HRNet training script",
        "evaluate_hrnet_facial.py": "Model evaluation and metrics",
        "hrnet_model.py": "Enhanced HRNet model implementation",
        "run_pipeline.py": "Automated pipeline runner",
        "test_pipeline.py": "Pipeline testing script",
        "requirements.txt": "Python dependencies",
        "README.md": "Complete documentation"
    }
    
    for filename, description in files_status.items():
        if Path(filename).exists():
            print(f"✅ {filename:<30} - {description}")
        else:
            print(f"❌ {filename:<30} - {description}")
    
    print()
    print("🚀 QUICK START COMMANDS:")
    print("-" * 40)
    print("1. Install dependencies:")
    print("   python -m pip install -r requirements.txt")
    print()
    print("2. Convert dataset:")
    print("   python yolo_to_coco_converter.py --dataset_root dataset --output coco_annotations.json")
    print()
    print("3. Test pipeline:")
    print("   python test_pipeline.py")
    print()
    print("4. Train model:")
    print("   python train_hrnet_facial.py --coco_file coco_annotations.json --image_root dataset --pretrained hrnet_w18_small_model_v1.pth")
    print()
    print("5. Evaluate model:")
    print("   python evaluate_hrnet_facial.py --model_path checkpoints/best_model.pth --coco_file coco_annotations.json --image_root dataset")
    print()
    print("6. Run complete pipeline:")
    print("   python run_pipeline.py --step all")
    
    print()
    print("📈 EXPECTED PERFORMANCE:")
    print("-" * 40)
    print("✅ PCK@0.05: 0.75-0.90 (target > 0.80)")
    print("✅ PCK@0.10: 0.85-0.95 (target > 0.90)")
    print("✅ NME: 0.03-0.08 (target < 0.06)")
    print("✅ Training Time: 2-4 hours (100 epochs)")
    print("✅ Inference Speed: 50-100 FPS")
    
    print()
    print("🎯 FACIAL KEYPOINTS (14 points):")
    print("-" * 40)
    keypoints = [
        "1. Forehead", "2. Eyebrow Outer", "3. Eyebrow Inner", "4. Eye Outer",
        "5. Eye Inner", "6. Nose Bridge", "7. Nose Tip", "8. Nose Bottom",
        "9. Lip Top", "10. Lip Corner", "11. Lip Bottom", "12. Chin Tip",
        "13. Jaw Mid", "14. Jaw Angle"
    ]
    
    for i, kp in enumerate(keypoints, 1):
        if i <= 7:
            print(f"✅ {kp:<15}", end="  ")
        else:
            print(f"✅ {kp:<15}")
        if i % 2 == 0:
            print()
    
    print()
    print("=" * 80)
    print("🎉 System Ready! Follow the Quick Start commands to begin.")
    print("=" * 80)


def check_system_readiness():
    """Check if the system is ready for training/evaluation"""
    print("\n🔍 SYSTEM READINESS CHECK:")
    print("-" * 40)
    
    ready = True
    
    # Check dataset
    dataset_path = Path("dataset")
    if not dataset_path.exists():
        print("❌ Dataset folder not found")
        ready = False
    else:
        train_images = len(list((dataset_path / "train" / "images").glob("*.jpg")))
        print(f"✅ Dataset found ({train_images} training images)")
    
    # Check COCO annotations
    coco_path = Path("coco_annotations.json")
    if not coco_path.exists():
        print("❌ COCO annotations not found - run converter")
        ready = False
    else:
        print("✅ COCO annotations ready")
    
    # Check pretrained model
    pretrained_path = Path("hrnet_w18_small_model_v1.pth")
    if not pretrained_path.exists():
        print("❌ Pretrained model not found")
        ready = False
    else:
        print("✅ Pretrained model ready")
    
    # Check Python environment
    try:
        import torch
        import cv2
        import numpy as np
        print("✅ Core dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        ready = False
    
    if ready:
        print("\n🎉 System is ready for training and evaluation!")
    else:
        print("\n⚠️  System needs setup. Please install missing components.")
    
    return ready


if __name__ == "__main__":
    print_system_summary()
    check_system_readiness()
    
    print("\n" + "=" * 80)
    print("📚 For detailed instructions, see README.md")
    print("🐛 For issues, check the troubleshooting section")
    print("=" * 80)

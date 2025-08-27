"""
HRNet Facial Keypoints Setup and Training Pipeline
Provides easy setup and execution commands
"""

import subprocess
import sys
from pathlib import Path
import argparse


def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Requirements installed successfully!")


def convert_dataset():
    """Convert YOLO format to COCO format"""
    print("Converting dataset from YOLO to COCO format...")
    subprocess.check_call([
        sys.executable, "yolo_to_coco_converter.py",
        "--dataset_root", "dataset",
        "--output", "coco_annotations.json"
    ])
    print("Dataset conversion completed!")


def train_model():
    """Train HRNet model"""
    print("Starting HRNet training...")
    subprocess.check_call([
        sys.executable, "train_hrnet_facial.py",
        "--coco_file", "coco_annotations.json",
        "--image_root", "dataset",
        "--pretrained", "hrnet_w18_small_model_v1.pth",
        "--save_dir", "checkpoints",
        "--batch_size", "16",
        "--epochs", "100"
    ])
    print("Training completed!")


def evaluate_model():
    """Evaluate trained model"""
    print("Evaluating model...")
    best_model_path = Path("checkpoints/best_model.pth")
    if not best_model_path.exists():
        best_model_path = Path("checkpoints/checkpoint_epoch_99.pth")
    
    subprocess.check_call([
        sys.executable, "evaluate_hrnet_facial.py",
        "--model_path", str(best_model_path),
        "--coco_file", "coco_annotations.json",
        "--image_root", "dataset",
        "--save_dir", "evaluation_results",
        "--batch_size", "16"
    ])
    print("Evaluation completed!")


def main():
    parser = argparse.ArgumentParser(description='HRNet Facial Keypoints Pipeline')
    parser.add_argument('--step', type=str, choices=['install', 'convert', 'train', 'evaluate', 'all'],
                        default='all', help='Pipeline step to execute')
    
    args = parser.parse_args()
    
    if args.step in ['install', 'all']:
        install_requirements()
    
    if args.step in ['convert', 'all']:
        convert_dataset()
    
    if args.step in ['train', 'all']:
        train_model()
    
    if args.step in ['evaluate', 'all']:
        evaluate_model()


if __name__ == '__main__':
    main()

"""
Quick Launch Script for Facial Keypoints Interface
"""

import subprocess
import sys
import os
from pathlib import Path

def launch_interface():
    """Launch the Streamlit interface"""
    
    print("🚀 Launching HRNet Facial Keypoints Interface...")
    print("=" * 50)
    
    # Check if we have a trained model
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        print("⚠️  No checkpoints directory found.")
        print("   Please train a model first using:")
        print("   python train_hrnet_facial.py --coco_file coco_annotations.json --image_root dataset --pretrained hrnet_w18_small_model_v1.pth")
        return
    
    # Check for model files
    model_files = list(checkpoints_dir.glob("*.pth"))
    if not model_files:
        print("⚠️  No trained models found in checkpoints directory.")
        print("   Please train a model first.")
        return
    
    print(f"✅ Found {len(model_files)} model file(s):")
    for model_file in model_files:
        print(f"   - {model_file.name}")
    
    print("\n🌐 Starting Streamlit server...")
    print("   The interface will open in your default web browser")
    print("   URL: http://localhost:8501")
    print("\n📋 Instructions:")
    print("   1. Load a model using the sidebar")
    print("   2. Upload a lateral face image")
    print("   3. Click 'Detect Keypoints' to analyze")
    print("   4. View results and download metrics")
    print("\n🛑 To stop the server, press Ctrl+C")
    print("=" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "facial_keypoints_interface.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Interface stopped by user")
    except Exception as e:
        print(f"❌ Error launching interface: {e}")

if __name__ == "__main__":
    launch_interface()

#!/usr/bin/env python3
"""
Launch script for the Facial Keypoints Detection Interface
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def main():
    print("🚀 Starting Facial Keypoints Detection Interface...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("facial_keypoints_app.py").exists():
        print("❌ Error: facial_keypoints_app.py not found!")
        print("Please run this script from the project directory.")
        return
    
    # Check for trained models
    model_files = list(Path(".").glob("*.pth")) + list(Path("checkpoints").glob("*.pth"))
    if not model_files:
        print("⚠️  Warning: No trained models found!")
        print("Please ensure you have a .pth model file available.")
        print("You can:")
        print("1. Train a model using: python train_hrnet_facial.py")
        print("2. Place a pre-trained model in the current directory")
        return
    
    print(f"✅ Found {len(model_files)} model(s):")
    for model in model_files:
        print(f"   - {model}")
    
    print("\n🌐 Launching web interface...")
    print("📍 URL: http://localhost:8501")
    print("\n💡 Tips:")
    print("   - Upload lateral face images for best results")
    print("   - Select your trained model from the sidebar")
    print("   - Use GPU if available for faster inference")
    print("\n" + "=" * 50)
    
    try:
        # Start Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "facial_keypoints_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false"
        ]
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run Streamlit
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Interface stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting interface: {e}")

if __name__ == "__main__":
    main()

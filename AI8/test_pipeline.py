"""
Test Training Script - Quick verification of the training pipeline
"""

import torch
import json
from pathlib import Path
from train_hrnet_facial import HRNetConfig, FacialKeypointsDataset, load_pretrained_hrnet

def test_training_pipeline():
    """Test the training pipeline with a small subset"""
    
    # Configuration
    config = HRNetConfig()
    config.batch_size = 2  # Small batch for testing
    config.epochs = 2      # Few epochs for testing
    
    # Check if COCO file exists
    coco_file = "coco_annotations.json"
    if not Path(coco_file).exists():
        print(f"Error: {coco_file} not found. Please run the converter first.")
        return False
    
    # Test dataset loading
    try:
        print("Testing dataset loading...")
        dataset = FacialKeypointsDataset(coco_file, "dataset", config, is_train=True)
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        
        # Test data loader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, 
                                                shuffle=True, num_workers=0)
        
        # Test loading a batch
        for batch in dataloader:
            print(f"Batch shapes:")
            print(f"  Images: {batch['image'].shape}")
            print(f"  Heatmaps: {batch['heatmaps'].shape}")
            print(f"  Target weights: {batch['target_weight'].shape}")
            print(f"  Keypoints: {batch['keypoints'].shape}")
            break
            
    except Exception as e:
        print(f"Error in dataset loading: {e}")
        return False
    
    # Test model loading
    try:
        print("\nTesting model loading...")
        model = load_pretrained_hrnet(config, "hrnet_w18_small_model_v1.pth")
        print(f"Model loaded successfully")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Model output shape: {output.shape}")
        
    except Exception as e:
        print(f"Error in model loading: {e}")
        return False
    
    print("\n✅ Training pipeline test completed successfully!")
    print("\nNext steps:")
    print("1. Run full training: python train_hrnet_facial.py --coco_file coco_annotations.json --image_root dataset --pretrained hrnet_w18_small_model_v1.pth")
    print("2. Or run automated pipeline: python run_pipeline.py --step train")
    
    return True

if __name__ == "__main__":
    test_training_pipeline()

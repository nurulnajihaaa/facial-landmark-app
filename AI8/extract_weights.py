"""
Extract just the model weights from the checkpoint
"""
import torch
from pathlib import Path

def extract_model_weights():
    """Extract just model weights from checkpoint"""
    
    input_path = "checkpoints/proper_hrnet_best.pth"
    output_path = "checkpoints/proper_hrnet_weights_only.pth"
    
    if not Path(input_path).exists():
        print(f"❌ Checkpoint not found: {input_path}")
        return
    
    try:
        # Try to load with weights_only first
        try:
            checkpoint = torch.load(input_path, map_location='cpu', weights_only=True)
            print("✅ Loaded with weights_only=True")
        except:
            # If that fails, load normally but only extract what we need
            checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
            print("⚠️  Loaded with weights_only=False")
        
        # Extract only the model state dict
        if 'model_state_dict' in checkpoint:
            model_weights = checkpoint['model_state_dict']
            print(f"📦 Extracted model state dict with {len(model_weights)} keys")
            
            # Save just the weights
            torch.save(model_weights, output_path)
            print(f"💾 Saved weights to: {output_path}")
            
            # Also save some metadata if available
            metadata = {}
            if 'val_loss' in checkpoint:
                metadata['val_loss'] = checkpoint['val_loss']
            if 'epoch' in checkpoint:
                metadata['epoch'] = checkpoint['epoch']
            
            if metadata:
                metadata_path = "checkpoints/proper_hrnet_metadata.pth"
                torch.save(metadata, metadata_path)
                print(f"📋 Saved metadata to: {metadata_path}")
                
        else:
            print("❌ No model_state_dict found in checkpoint")
            
    except Exception as e:
        print(f"❌ Error processing checkpoint: {e}")

if __name__ == "__main__":
    extract_model_weights()

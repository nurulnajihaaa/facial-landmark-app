"""
Test proper HRNet facial keypoint detection with anatomical accuracy focus
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from PIL import Image
import streamlit as st

sys.path.append('.')
from proper_hrnet import ProperHRNet
from train_hrnet_facial import HRNetConfig

class AnatomicalValidator:
    """Validate anatomical correctness of facial landmarks"""
    
    # Facial landmark anatomical constraints (approximate ratios)
    ANATOMICAL_CHECKS = {
        'face_symmetry': {
            'description': 'Left and right facial features should be symmetric',
            'threshold': 0.3  # Allow 30% asymmetry
        },
        'eye_line': {
            'description': 'Eyes should be approximately on the same horizontal line',
            'threshold': 0.1  # Allow 10% vertical difference
        },
        'nose_center': {
            'description': 'Nose should be near the center of the face',
            'threshold': 0.2  # Allow 20% offset from center
        },
        'mouth_below_nose': {
            'description': 'Mouth should be below the nose',
            'min_distance': 0.05  # Minimum distance ratio
        },
        'facial_proportions': {
            'description': 'Standard facial proportions should be maintained',
            'eye_to_nose_ratio': (0.2, 0.6),  # Eyes to nose distance
            'nose_to_mouth_ratio': (0.1, 0.4)  # Nose to mouth distance
        }
    }
    
    def __init__(self, keypoint_mapping=None):
        """
        Initialize with keypoint mapping if available
        keypoint_mapping: dict mapping anatomical names to keypoint indices
        """
        # Default generic mapping - adjust based on your dataset
        self.keypoint_mapping = keypoint_mapping or {
            'left_eye': [0, 1],      # Assume first keypoints are left eye region
            'right_eye': [2, 3],     # Right eye region  
            'nose': [4, 5, 6],       # Nose region
            'mouth': [7, 8, 9, 10],  # Mouth region
            'face_outline': list(range(11, 14))  # Face boundary
        }
    
    def validate_predictions(self, keypoints, image_shape=None):
        """
        Validate anatomical correctness of predictions
        keypoints: (N, 2) numpy array of (x, y) coordinates
        Returns: dict with validation results
        """
        results = {
            'is_anatomically_correct': True,
            'errors': [],
            'warnings': [],
            'scores': {}
        }
        
        if len(keypoints) < 4:
            results['is_anatomically_correct'] = False
            results['errors'].append("Insufficient keypoints for validation")
            return results
        
        try:
            # Basic coordinate sanity check
            if image_shape:
                h, w = image_shape[:2]
                if (keypoints[:, 0] < 0).any() or (keypoints[:, 0] >= w).any() or \
                   (keypoints[:, 1] < 0).any() or (keypoints[:, 1] >= h).any():
                    results['warnings'].append("Some keypoints are outside image bounds")
            
            # Check if keypoints are clustered (poor spread)
            spread_score = self._check_keypoint_spread(keypoints)
            results['scores']['spread'] = spread_score
            
            if spread_score < 0.3:
                results['errors'].append(f"Poor keypoint spread: {spread_score:.3f}")
                results['is_anatomically_correct'] = False
            
            # Check facial symmetry if we have eye keypoints
            symmetry_score = self._check_facial_symmetry(keypoints)
            results['scores']['symmetry'] = symmetry_score
            
            if symmetry_score < 0.5:
                results['warnings'].append(f"Poor facial symmetry: {symmetry_score:.3f}")
            
            # Check vertical ordering (eyes above nose above mouth)
            ordering_score = self._check_vertical_ordering(keypoints)
            results['scores']['vertical_ordering'] = ordering_score
            
            if ordering_score < 0.5:
                results['errors'].append(f"Poor vertical ordering: {ordering_score:.3f}")
                results['is_anatomically_correct'] = False
                
            # Overall anatomical score
            overall_score = np.mean([
                spread_score,
                symmetry_score,
                ordering_score
            ])
            results['scores']['overall'] = overall_score
            
            if overall_score < 0.4:
                results['is_anatomically_correct'] = False
                
        except Exception as e:
            results['errors'].append(f"Validation error: {str(e)}")
            results['is_anatomically_correct'] = False
        
        return results
    
    def _check_keypoint_spread(self, keypoints):
        """Check if keypoints have good spatial distribution"""
        if len(keypoints) < 2:
            return 0.0
        
        # Calculate bounding box
        x_min, y_min = keypoints.min(axis=0)
        x_max, y_max = keypoints.max(axis=0)
        
        bbox_area = (x_max - x_min) * (y_max - y_min)
        
        # Calculate convex hull area
        try:
            from scipy.spatial import ConvexHull
            if len(keypoints) >= 3:
                hull = ConvexHull(keypoints)
                hull_area = hull.volume  # In 2D, volume is area
                spread_ratio = hull_area / (bbox_area + 1e-6)
            else:
                spread_ratio = 0.5
        except:
            # Fallback: use standard deviation
            std_x = np.std(keypoints[:, 0])
            std_y = np.std(keypoints[:, 1])
            spread_ratio = min(1.0, (std_x + std_y) / 100.0)
        
        return float(spread_ratio)
    
    def _check_facial_symmetry(self, keypoints):
        """Check left-right facial symmetry"""
        if len(keypoints) < 4:
            return 0.5  # Default score
        
        # Find approximate face center
        center_x = np.mean(keypoints[:, 0])
        
        # Split keypoints into left and right
        left_points = keypoints[keypoints[:, 0] < center_x]
        right_points = keypoints[keypoints[:, 0] > center_x]
        
        if len(left_points) == 0 or len(right_points) == 0:
            return 0.3  # Poor symmetry
        
        # Compare distributions
        left_y_std = np.std(left_points[:, 1]) if len(left_points) > 1 else 0
        right_y_std = np.std(right_points[:, 1]) if len(right_points) > 1 else 0
        
        symmetry_score = 1.0 - abs(left_y_std - right_y_std) / (left_y_std + right_y_std + 1e-6)
        return float(np.clip(symmetry_score, 0, 1))
    
    def _check_vertical_ordering(self, keypoints):
        """Check if facial features follow proper vertical ordering"""
        if len(keypoints) < 3:
            return 0.5
        
        # Sort by y-coordinate (top to bottom)
        sorted_points = keypoints[keypoints[:, 1].argsort()]
        
        # Check if there's reasonable vertical spread
        y_range = sorted_points[-1, 1] - sorted_points[0, 1]
        
        if y_range < 10:  # Too little vertical spread
            return 0.2
        
        # Check if points are not all clustered
        y_positions = sorted_points[:, 1]
        y_diffs = np.diff(y_positions)
        
        # Good ordering should have some variation in vertical positions
        ordering_score = min(1.0, np.std(y_diffs) / (y_range + 1e-6))
        
        return float(ordering_score)

class ProperHRNetTester:
    """Test proper HRNet implementation"""
    
    def __init__(self, model_path, config=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load config from checkpoint if available
        if config is None:
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if 'config' in checkpoint:
                    # Create a simple config object with the saved attributes
                    class SimpleConfig:
                        pass
                    
                    saved_config = checkpoint['config']
                    self.config = SimpleConfig()
                    
                    # Copy attributes from saved config
                    if hasattr(saved_config, '__dict__'):
                        for attr, value in saved_config.__dict__.items():
                            setattr(self.config, attr, value)
                    else:
                        # Fallback config values
                        self.config.input_size = [256, 256]
                        self.config.output_size = [64, 64]
                        self.config.num_keypoints = 14
                        self.config.sigma = 2.0
                else:
                    # Fallback config
                    class SimpleConfig:
                        def __init__(self):
                            self.input_size = [256, 256]
                            self.output_size = [64, 64]
                            self.num_keypoints = 14
                            self.sigma = 2.0
                    self.config = SimpleConfig()
            except Exception as e:
                print(f"Warning: Could not load config from checkpoint: {e}")
                # Fallback config
                class SimpleConfig:
                    def __init__(self):
                        self.input_size = [256, 256]
                        self.output_size = [64, 64]
                        self.num_keypoints = 14
                        self.sigma = 2.0
                self.config = SimpleConfig()
        else:
            self.config = config
        
        # Initialize model
        self.model = ProperHRNet(self.config).to(self.device)
        
        # Load weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✅ Loaded model from {model_path}")
            print(f"📊 Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
            if 'val_loss' in checkpoint:
                print(f"📈 Best validation loss: {checkpoint['val_loss']:.6f}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Initialize validator
        self.validator = AnatomicalValidator()
    
    def preprocess_image(self, image_path):
        """Preprocess image for HRNet inference"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        original_shape = image.shape
        
        # Resize to model input size
        target_size = tuple(self.config.input_size)
        image_resized = cv2.resize(image, target_size)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_resized).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.unsqueeze(0), original_shape, image_resized
    
    def heatmaps_to_keypoints(self, heatmaps):
        """Convert heatmaps to keypoint coordinates"""
        batch_size, num_joints, h, w = heatmaps.shape
        keypoints = []
        
        for b in range(batch_size):
            batch_keypoints = []
            for j in range(num_joints):
                heatmap = heatmaps[b, j]
                
                # Find maximum location
                max_val = torch.max(heatmap)
                if max_val < 0.1:  # Low confidence
                    batch_keypoints.extend([0, 0, 0])
                    continue
                
                # Get coordinates of maximum
                max_idx = torch.argmax(heatmap.flatten())
                y = max_idx // w
                x = max_idx % w
                
                # Sub-pixel refinement using weighted average around peak
                if 1 <= x < w-1 and 1 <= y < h-1:
                    # 3x3 window around peak
                    window = heatmap[y-1:y+2, x-1:x+2]
                    weights = F.softmax(window.flatten(), dim=0).view(3, 3)
                    
                    # Weighted coordinates
                    y_coords = torch.arange(y-1, y+2, dtype=torch.float32)
                    x_coords = torch.arange(x-1, x+2, dtype=torch.float32)
                    
                    refined_y = torch.sum(weights * y_coords.view(-1, 1)).item()
                    refined_x = torch.sum(weights * x_coords.view(1, -1)).item()
                else:
                    refined_x, refined_y = float(x), float(y)
                
                # Scale to input image size
                x_scaled = refined_x * self.config.input_size[0] / w
                y_scaled = refined_y * self.config.input_size[1] / h
                
                batch_keypoints.extend([x_scaled, y_scaled, float(max_val)])
            
            keypoints.append(batch_keypoints)
        
        return np.array(keypoints)
    
    def predict(self, image_path):
        """Predict keypoints for an image"""
        # Preprocess
        image_tensor, original_shape, resized_image = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            heatmaps = self.model(image_tensor)
            
            if isinstance(heatmaps, list):
                heatmaps = heatmaps[-1]  # Use final stage output
        
        # Convert to keypoints
        keypoints = self.heatmaps_to_keypoints(heatmaps.cpu())
        
        # Extract coordinates for validation
        coords = []
        for i in range(0, len(keypoints[0]), 3):
            x, y, conf = keypoints[0][i:i+3]
            if conf > 0.1:  # Only include confident predictions
                coords.append([x, y])
        
        coords = np.array(coords) if coords else np.zeros((0, 2))
        
        # Validate anatomical correctness
        validation = self.validator.validate_predictions(coords, self.config.input_size)
        
        return {
            'keypoints': keypoints[0],
            'coordinates': coords,
            'heatmaps': heatmaps[0].cpu().numpy(),
            'validation': validation,
            'original_shape': original_shape,
            'resized_image': resized_image
        }
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize predictions with anatomical validation"""
        result = self.predict(image_path)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image with keypoints
        axes[0, 0].imshow(result['resized_image'])
        if len(result['coordinates']) > 0:
            axes[0, 0].scatter(result['coordinates'][:, 0], result['coordinates'][:, 1], 
                              c='red', s=50, alpha=0.8)
            # Add keypoint numbers
            for i, (x, y) in enumerate(result['coordinates']):
                axes[0, 0].text(x+2, y+2, str(i), color='yellow', fontsize=8)
        
        axes[0, 0].set_title(f"Keypoint Predictions (n={len(result['coordinates'])})")
        axes[0, 0].axis('off')
        
        # Heatmap visualization (sum of all channels)
        heatmap_sum = np.sum(result['heatmaps'], axis=0)
        axes[0, 1].imshow(result['resized_image'], alpha=0.7)
        im = axes[0, 1].imshow(heatmap_sum, alpha=0.6, cmap='hot')
        axes[0, 1].set_title("Heatmap Overlay")
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Validation results
        validation = result['validation']
        axes[1, 0].axis('off')
        
        # Validation text
        val_text = f"ANATOMICAL VALIDATION\\n"
        val_text += f"Overall Score: {validation['scores'].get('overall', 0):.3f}\\n\\n"
        
        if validation['is_anatomically_correct']:
            val_text += "✅ ANATOMICALLY CORRECT\\n"
        else:
            val_text += "❌ ANATOMICALLY INCORRECT\\n"
        
        val_text += f"\\nScores:\\n"
        for score_name, score_val in validation['scores'].items():
            val_text += f"  {score_name}: {score_val:.3f}\\n"
        
        if validation['errors']:
            val_text += f"\\nErrors:\\n"
            for error in validation['errors']:
                val_text += f"  • {error}\\n"
        
        if validation['warnings']:
            val_text += f"\\nWarnings:\\n"
            for warning in validation['warnings']:
                val_text += f"  • {warning}\\n"
        
        axes[1, 0].text(0.05, 0.95, val_text, transform=axes[1, 0].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 0].set_title("Validation Results")
        
        # Individual heatmaps (first 6)
        axes[1, 1].axis('off')
        
        # Create mini subplot for heatmaps
        n_show = min(6, result['heatmaps'].shape[0])
        if n_show > 0:
            for i in range(n_show):
                plt.subplot(2, 4, 5 + i)
                plt.imshow(result['heatmaps'][i], cmap='hot')
                plt.title(f"KP {i}", fontsize=8)
                plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved visualization to: {save_path}")
        
        plt.show()
        
        return result

def test_proper_hrnet():
    """Test the proper HRNet implementation"""
    print("=== TESTING PROPER HRNET ===")
    
    # Check for model
    model_path = "checkpoints/proper_hrnet_best.pth"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using: python train_proper_hrnet.py")
        return
    
    # Initialize tester
    tester = ProperHRNetTester(model_path)
    
    # Find test images
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(Path('dataset/test/images').glob(ext))
    
    if not test_images:
        print("❌ No test images found in dataset/test/images/")
        return
    
    print(f"📸 Found {len(test_images)} test images")
    
    # Test on a few images
    output_dir = Path("outputs/proper_hrnet_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    anatomically_correct = 0
    total_tested = 0
    
    for i, image_path in enumerate(test_images[:5]):  # Test first 5 images
        print(f"\\n🔍 Testing image {i+1}: {image_path.name}")
        
        try:
            save_path = output_dir / f"test_{i+1}_{image_path.stem}.png"
            result = tester.visualize_prediction(str(image_path), str(save_path))
            
            total_tested += 1
            if result['validation']['is_anatomically_correct']:
                anatomically_correct += 1
                print("  ✅ Anatomically correct")
            else:
                print("  ❌ Anatomically incorrect")
                print(f"    Errors: {result['validation']['errors']}")
            
            print(f"    Overall score: {result['validation']['scores'].get('overall', 0):.3f}")
            
        except Exception as e:
            print(f"  ❌ Error testing {image_path.name}: {e}")
    
    # Summary
    if total_tested > 0:
        accuracy_rate = anatomically_correct / total_tested
        print(f"\\n📊 ANATOMICAL ACCURACY SUMMARY")
        print(f"Correct predictions: {anatomically_correct}/{total_tested} ({accuracy_rate:.1%})")
        
        if accuracy_rate >= 0.8:
            print("🎉 EXCELLENT anatomical accuracy!")
        elif accuracy_rate >= 0.6:
            print("✅ GOOD anatomical accuracy")
        elif accuracy_rate >= 0.4:
            print("⚠️  MODERATE anatomical accuracy - needs improvement")
        else:
            print("❌ POOR anatomical accuracy - major issues")
    
    print(f"\\n💾 Test results saved to: {output_dir}")

if __name__ == "__main__":
    test_proper_hrnet()

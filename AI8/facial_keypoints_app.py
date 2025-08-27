"""
Facial Keypoints Detection Web Interface
Interactive Streamlit app for facial landmark detection with metric calculations
"""

import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import pandas as pd
from pathlib import Path
import json
from scipy.spatial.distance import euclidean
import math

# Import our model classes and utilities
from train_hrnet_facial import SimpleHRNet, HRNetConfig
from model_utils import load_model_safely, get_model_info

try:
    from hrnet_model import get_pose_net, hrnet_w18_small_config
    ENHANCED_HRNET_AVAILABLE = True
except ImportError:
    ENHANCED_HRNET_AVAILABLE = False


class FacialKeypointsInference:
    """Inference class for facial keypoints detection"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.config = HRNetConfig()
        
        # Define keypoint names
        self.keypoint_names = [
            'forehead',      # kp1 - top of forehead
            'temple',        # kp2 - side of forehead/temple
            'eye_outer',     # kp3 - outer eye corner
            'eye_inner',     # kp4 - inner eye corner  
            'nose_bridge',   # kp5 - nose bridge/upper nose
            'nose_tip',      # kp6 - nose tip
            'nose_base',     # kp7 - nose base/nostril
            'lip_top',       # kp8 - upper lip
            'lip_corner',    # kp9 - mouth corner
            'lip_bottom',    # kp10 - lower lip
            'chin',          # kp11 - chin area
            'jaw_point',     # kp12 - jaw point
            'jaw_angle',     # kp13 - jaw angle/back of jaw
            'ear'            # kp14 - ear area
        ]
        
        # Define measurement pairs for orthodontic analysis
        self.measurement_pairs = {
            'Nasolabial Distance': ('nose_tip', 'lip_top'),
            'Lip Height': ('lip_top', 'lip_bottom'),
            'Lower Face Height': ('nose_tip', 'chin'),
            'Eye to Nose': ('eye_inner', 'nose_bridge'),
            'Nose Length': ('nose_bridge', 'nose_tip'),
            'Mouth Width': ('lip_corner', 'jaw_point'),
            'Face Height': ('forehead', 'chin'),
            'Jaw Width': ('jaw_point', 'jaw_angle')
        }
        
        # Define angle triplets
        self.angle_triplets = {
            'Facial Convexity': ('forehead', 'nose_tip', 'chin'),
            'Lower Facial Angle': ('nose_tip', 'lip_top', 'chin'),
            'Nasal Angle': ('nose_bridge', 'nose_tip', 'nose_base'),
            'Lip Angle': ('lip_corner', 'lip_top', 'lip_bottom')
        }
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Image preprocessing
        self.transforms = self._get_transforms()
    
    def load_model(self, model_path):
        """Load the trained model safely"""
        try:
            # Use our safe model loading utility
            model = load_model_safely(model_path, target_keypoints=14, device=self.device)
            st.info(f"Model loaded successfully")
            return model
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            # Fallback to simple model
            st.warning("Loading fallback model...")
            model = SimpleHRNet(self.config)
            model.eval()
            model.to(self.device)
            return model
    
    def _get_transforms(self):
        """Get image preprocessing transforms"""
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(tuple(self.config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image, bbox=None):
        """Preprocess image for inference. If bbox is provided, crop using the same
        bbox padding/resizing rules used during training. Returns (input_batch, processed_image).
        processed_image is the cropped & resized RGB image (np.ndarray) used for visualization.
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure RGB format
        if not (len(image.shape) == 3 and image.shape[2] == 3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # If a bbox is provided, apply the same crop+padding logic as the dataset
        if bbox is not None:
            x, y, w, h = bbox
            padding = 0.3
            x = max(0, x - w * padding / 2)
            y = max(0, y - h * padding / 2)
            w = w * (1 + padding)
            h = h * (1 + padding)

            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)

            cropped_image = image[int(y):int(y+h), int(x):int(x+w)]
            if cropped_image.size == 0:
                # Fallback to full image
                cropped_image = image.copy()

            resized_image = cv2.resize(cropped_image, tuple(self.config.input_size))
            processed_image = resized_image.copy()

            # Apply transforms to the resized image
            input_tensor = self.transforms(resized_image)
            input_batch = input_tensor.unsqueeze(0)
            return input_batch, processed_image

        # No bbox: process the full image (legacy behavior)
        # Apply transforms
        input_tensor = self.transforms(image)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        return input_batch, image
    
    def get_keypoints_from_heatmaps(self, heatmaps):
        """Extract keypoint coordinates from heatmaps"""
        batch_size, num_joints, height, width = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape(batch_size, num_joints, -1)
        
        # Find maximum values and indices
        max_vals, max_indices = torch.max(heatmaps_reshaped, dim=2)
        
        # Convert flat indices to 2D coordinates
        keypoints = torch.zeros(batch_size, num_joints, 2)
        keypoints[:, :, 0] = max_indices % width  # x coordinate
        keypoints[:, :, 1] = max_indices // width  # y coordinate
        
        # Scale keypoints to input image size
        scale_x = self.config.input_size[0] / width
        scale_y = self.config.input_size[1] / height
        
        keypoints[:, :, 0] *= scale_x
        keypoints[:, :, 1] *= scale_y
        
        # Filter out low confidence keypoints
        confidence_threshold = 0.1
        valid_keypoints = max_vals > confidence_threshold
        
        return keypoints[0].cpu().numpy(), valid_keypoints[0].cpu().numpy()
    
    def predict(self, image, bbox=None):
        """Predict keypoints for an image. If bbox is provided the image will be cropped
        using the bbox rules used in training so coordinates are returned relative to the
        cropped/resized image (which is what the app visualizes)."""
        input_batch, processed_image = self.preprocess_image(image, bbox=bbox)

        with torch.no_grad():
            input_batch = input_batch.to(self.device)
            heatmaps = self.model(input_batch)

            # If model output size doesn't match expected heatmap size, resize to output_size
            if isinstance(heatmaps, torch.Tensor) and heatmaps.dim() == 4:
                if heatmaps.shape[2:] != tuple(self.config.output_size[::-1]):
                    heatmaps = F.interpolate(heatmaps, size=tuple(self.config.output_size[::-1]), mode='bilinear', align_corners=False)

            # Get keypoint coordinates (returned in pixels relative to processed_image which is input_size)
            keypoints, confidence = self.get_keypoints_from_heatmaps(heatmaps)

        return keypoints, confidence, processed_image
    
    def calculate_distance(self, kp1, kp2):
        """Calculate Euclidean distance between two keypoints"""
        return euclidean(kp1, kp2)
    
    def calculate_angle(self, kp1, kp2, kp3):
        """Calculate angle formed by three keypoints (kp2 is the vertex)"""
        v1 = np.array(kp1) - np.array(kp2)
        v2 = np.array(kp3) - np.array(kp2)
        
        # Calculate angle in degrees
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def calculate_measurements(self, keypoints, confidence):
        """Calculate all orthodontic measurements"""
        measurements = {}
        
        # Distance measurements
        for measure_name, (kp1_name, kp2_name) in self.measurement_pairs.items():
            try:
                kp1_idx = self.keypoint_names.index(kp1_name)
                kp2_idx = self.keypoint_names.index(kp2_name)
                
                if confidence[kp1_idx] and confidence[kp2_idx]:
                    kp1 = keypoints[kp1_idx]
                    kp2 = keypoints[kp2_idx]
                    distance = self.calculate_distance(kp1, kp2)
                    measurements[measure_name] = f"{distance:.2f} px"
                else:
                    measurements[measure_name] = "N/A (low confidence)"
            except (ValueError, IndexError):
                measurements[measure_name] = "N/A (keypoint not found)"
        
        # Angle measurements
        for angle_name, (kp1_name, kp2_name, kp3_name) in self.angle_triplets.items():
            try:
                kp1_idx = self.keypoint_names.index(kp1_name)
                kp2_idx = self.keypoint_names.index(kp2_name)
                kp3_idx = self.keypoint_names.index(kp3_name)
                
                if confidence[kp1_idx] and confidence[kp2_idx] and confidence[kp3_idx]:
                    kp1 = keypoints[kp1_idx]
                    kp2 = keypoints[kp2_idx]
                    kp3 = keypoints[kp3_idx]
                    angle = self.calculate_angle(kp1, kp2, kp3)
                    measurements[angle_name] = f"{angle:.1f}°"
                else:
                    measurements[angle_name] = "N/A (low confidence)"
            except (ValueError, IndexError, ZeroDivisionError):
                measurements[angle_name] = "N/A (calculation error)"
        
        return measurements
    
    def visualize_results(self, image, keypoints, confidence, measurements):
        """Create visualization with keypoints and measurements"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image with keypoints
        ax1.imshow(image)
        ax1.set_title('Detected Facial Keypoints', fontsize=14, fontweight='bold')
        
        # Plot keypoints
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.keypoint_names)))
        for i, (kp, conf, name, color) in enumerate(zip(keypoints, confidence, self.keypoint_names, colors)):
            if conf:
                ax1.plot(kp[0], kp[1], 'o', color=color, markersize=8, alpha=0.8)
                ax1.annotate(f'{i+1}', (kp[0], kp[1]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, color='white', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
        
        # Draw connections between keypoints
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
            (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)
        ]
        
        for start_idx, end_idx in connections:
            if confidence[start_idx] and confidence[end_idx]:
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]
                ax1.plot([start_kp[0], end_kp[0]], [start_kp[1], end_kp[1]], 
                        'b-', alpha=0.5, linewidth=2)
        
        ax1.axis('off')
        
        # Measurements table
        ax2.axis('off')
        ax2.set_title('Facial Measurements', fontsize=14, fontweight='bold')
        
        # Create measurement table
        measurement_data = []
        for measure, value in measurements.items():
            measurement_data.append([measure, value])
        
        # Split measurements into distances and angles
        distance_measurements = []
        angle_measurements = []
        
        for measure, value in measurements.items():
            if '°' in value or 'Angle' in measure or 'Convexity' in measure:
                angle_measurements.append([measure, value])
            else:
                distance_measurements.append([measure, value])
        
        # Display distances
        if distance_measurements:
            table1 = ax2.table(cellText=distance_measurements,
                              colLabels=['Distance Measurements', 'Value'],
                              cellLoc='left',
                              loc='upper center',
                              bbox=[0.0, 0.55, 1.0, 0.4])
            table1.auto_set_font_size(False)
            table1.set_fontsize(10)
            table1.scale(1, 1.5)
            
            # Style the table
            for (i, j), cell in table1.get_celld().items():
                if i == 0:  # Header
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#4CAF50')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        # Display angles
        if angle_measurements:
            table2 = ax2.table(cellText=angle_measurements,
                              colLabels=['Angle Measurements', 'Value'],
                              cellLoc='left',
                              loc='lower center',
                              bbox=[0.0, 0.0, 1.0, 0.4])
            table2.auto_set_font_size(False)
            table2.set_fontsize(10)
            table2.scale(1, 1.5)
            
            # Style the table
            for (i, j), cell in table2.get_celld().items():
                if i == 0:  # Header
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#2196F3')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.tight_layout()
        return fig


def main():
    st.set_page_config(
        page_title="Facial Keypoints Detection",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔬 Facial Keypoints Detection & Analysis")
    st.markdown("Upload an image to detect facial landmarks and calculate orthodontic measurements")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    # Check for available models
    model_files = list(Path(".").glob("*.pth")) + list(Path("checkpoints").glob("*.pth"))
    model_options = [str(f) for f in model_files]
    
    if not model_options:
        st.error("No trained model found! Please ensure you have a .pth model file in the current directory or checkpoints folder.")
        st.stop()
    
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    
    # Device selection
    device_options = ["cpu"]
    if torch.cuda.is_available():
        device_options.append("cuda")
    
    selected_device = st.sidebar.selectbox("Device", device_options)
    
    # Initialize inference class
    @st.cache_resource
    def load_inference_model(model_path, device):
        return FacialKeypointsInference(model_path, device)
    
    try:
        inference = load_inference_model(selected_model, selected_device)
        st.sidebar.success(f"✅ Model loaded successfully!")
        st.sidebar.info(f"Device: {selected_device}")
        
        if ENHANCED_HRNET_AVAILABLE:
            st.sidebar.info("🚀 Using Enhanced HRNet")
        else:
            st.sidebar.info("📋 Using Simple HRNet")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a lateral face image for keypoint detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Analysis button
            if st.button("🔍 Analyze Facial Keypoints", type="primary"):
                with st.spinner("Analyzing facial keypoints..."):
                    try:
                        # Predict keypoints
                        keypoints, confidence, processed_image = inference.predict(image)
                        
                        # Calculate measurements
                        measurements = inference.calculate_measurements(keypoints, confidence)
                        
                        # Store results in session state
                        st.session_state.keypoints = keypoints
                        st.session_state.confidence = confidence
                        st.session_state.processed_image = processed_image
                        st.session_state.measurements = measurements
                        
                        st.success("✅ Analysis completed!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
    
    with col2:
        st.header("📊 Results")
        
        if hasattr(st.session_state, 'keypoints'):
            # Create and display visualization
            fig = inference.visualize_results(
                st.session_state.processed_image,
                st.session_state.keypoints,
                st.session_state.confidence,
                st.session_state.measurements
            )
            
            st.pyplot(fig)
            plt.close(fig)  # Prevent memory leaks
            
            # Additional metrics
            st.subheader("📈 Detailed Analysis")
            
            # Confidence scores
            with st.expander("Keypoint Confidence Scores"):
                conf_data = []
                for i, (name, conf) in enumerate(zip(inference.keypoint_names, st.session_state.confidence)):
                    status = "✅ High" if conf else "⚠️ Low"
                    conf_data.append([f"{i+1}. {name.replace('_', ' ').title()}", status])
                
                conf_df = pd.DataFrame(conf_data, columns=["Keypoint", "Confidence"])
                st.dataframe(conf_df, use_container_width=True)
            
            # Export results
            with st.expander("💾 Export Results"):
                # Prepare export data
                export_data = {
                    "keypoints": st.session_state.keypoints.tolist(),
                    "confidence": st.session_state.confidence.tolist(),
                    "measurements": st.session_state.measurements,
                    "keypoint_names": inference.keypoint_names
                }
                
                export_json = json.dumps(export_data, indent=2)
                
                st.download_button(
                    label="📄 Download Results (JSON)",
                    data=export_json,
                    file_name="facial_analysis_results.json",
                    mime="application/json"
                )
                
                # Create CSV for measurements
                measurements_df = pd.DataFrame(
                    list(st.session_state.measurements.items()),
                    columns=["Measurement", "Value"]
                )
                
                csv = measurements_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Measurements (CSV)",
                    data=csv,
                    file_name="facial_measurements.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("👆 Upload an image and click 'Analyze' to see results here")
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        ### Facial Keypoints Detection System
        
        This tool uses a trained HRNet model to detect 14 facial landmarks and calculate orthodontic measurements:
        
        **Detected Keypoints:**
        1. Forehead
        2. Eyebrow Outer
        3. Eyebrow Inner
        4. Eye Outer
        5. Eye Inner
        6. Nose Bridge
        7. Nose Tip
        8. Nose Bottom
        9. Lip Top
        10. Lip Corner
        11. Lip Bottom
        12. Chin Tip
        13. Jaw Mid
        14. Jaw Angle
        
        **Measurements Calculated:**
        - **Distance Measurements**: Nasolabial distance, lip height, facial dimensions
        - **Angle Measurements**: Facial convexity, nasal angles, lip angles
        
        **Usage Tips:**
        - Use lateral (side view) face images for best results
        - Ensure good lighting and clear facial features
        - Images should show the full face profile
        """)


if __name__ == "__main__":
    main()

"""
Streamlit Web Interface for HRNet Facial Keypoints Detection
Upload images and visualize keypoints with orthodontic metrics
"""

import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import pandas as pd
from pathlib import Path
import io
import base64
from scipy.spatial.distance import euclidean
import math

# Import our models
from train_hrnet_facial import HRNetConfig, SimpleHRNet, load_pretrained_hrnet
from evaluate_hrnet_facial import FacialKeypointsEvaluator

try:
    from hrnet_model import get_pose_net, hrnet_w18_small_config
    ENHANCED_HRNET_AVAILABLE = True
except ImportError:
    ENHANCED_HRNET_AVAILABLE = False


class FacialKeypointsInterface:
    """Web interface for facial keypoints detection"""
    
    def __init__(self):
        self.config = HRNetConfig()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define keypoint names and connections
        self.keypoint_names = [
            'Forehead', 'Eyebrow Outer', 'Eyebrow Inner', 'Eye Outer',
            'Eye Inner', 'Nose Bridge', 'Nose Tip', 'Nose Bottom',
            'Lip Top', 'Lip Corner', 'Lip Bottom', 'Chin Tip',
            'Jaw Mid', 'Jaw Angle'
        ]
        
        # Define connections for drawing skeleton
        self.skeleton_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Forehead to eye
            (4, 5), (5, 6), (6, 7),          # Eye to nose
            (7, 8), (8, 9), (9, 10),         # Nose to lip
            (10, 11), (11, 12), (12, 13)     # Lip to jaw
        ]
        
        # Orthodontic measurement pairs
        self.orthodontic_pairs = {
            'Nasolabial Distance': ('Nose Tip', 'Lip Top'),
            'Lip Height': ('Lip Top', 'Lip Bottom'),
            'Lower Face Height': ('Nose Tip', 'Chin Tip'),
            'Facial Width': ('Eye Outer', 'Jaw Angle'),
            'Total Face Height': ('Forehead', 'Chin Tip'),
            'Eye to Nose': ('Eye Inner', 'Nose Bridge'),
            'Nose Width': ('Nose Bridge', 'Nose Bottom'),
            'Jaw Width': ('Jaw Mid', 'Jaw Angle')
        }
        
        # Orthodontic angle triplets
        self.orthodontic_angles = {
            'Facial Convexity': ('Forehead', 'Nose Tip', 'Chin Tip'),
            'Lower Facial Angle': ('Nose Tip', 'Lip Top', 'Chin Tip'),
            'Nasal Angle': ('Nose Bridge', 'Nose Tip', 'Nose Bottom')
        }
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            if ENHANCED_HRNET_AVAILABLE:
                self.model = get_pose_net(hrnet_w18_small_config, is_train=False)
            else:
                self.model = SimpleHRNet(self.config)
            
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model = self.model.to(self.device)
                self.model.eval()
                return True
            else:
                st.error(f"Model file not found: {model_path}")
                return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for model inference"""
        # Convert PIL to numpy
        img_np = np.array(image)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Resize to model input size
        img_resized = cv2.resize(img_np, tuple(self.config.input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img_tensor = torch.FloatTensor(img_rgb).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0), img_rgb
    
    def predict_keypoints(self, image_tensor):
        """Predict keypoints from image tensor"""
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            heatmaps = self.model(image_tensor)
            
            # Get keypoints from heatmaps
            batch_size, num_joints, h, w = heatmaps.shape
            heatmaps_np = heatmaps.cpu().numpy()
            
            keypoints = []
            for i in range(num_joints):
                heatmap = heatmaps_np[0, i]
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                
                # Scale back to input image size
                x_scaled = x * self.config.input_size[0] / w
                y_scaled = y * self.config.input_size[1] / h
                
                confidence = heatmap[y, x]
                keypoints.append([x_scaled, y_scaled, confidence])
        
        return np.array(keypoints)
    
    def calculate_distance(self, kp1, kp2):
        """Calculate Euclidean distance between two keypoints"""
        return euclidean(kp1[:2], kp2[:2])
    
    def calculate_angle(self, kp1, kp2, kp3):
        """Calculate angle formed by three keypoints (kp2 is the vertex)"""
        v1 = np.array(kp1[:2]) - np.array(kp2[:2])
        v2 = np.array(kp3[:2]) - np.array(kp2[:2])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        return angle
    
    def compute_orthodontic_metrics(self, keypoints):
        """Compute orthodontic distance and angle measurements"""
        distances = {}
        angles = {}
        
        # Distance measurements
        for metric_name, (kp1_name, kp2_name) in self.orthodontic_pairs.items():
            try:
                kp1_idx = self.keypoint_names.index(kp1_name)
                kp2_idx = self.keypoint_names.index(kp2_name)
                
                distance = self.calculate_distance(keypoints[kp1_idx], keypoints[kp2_idx])
                distances[metric_name] = distance
            except (ValueError, IndexError):
                distances[metric_name] = 0.0
        
        # Angle measurements
        for angle_name, (kp1_name, kp2_name, kp3_name) in self.orthodontic_angles.items():
            try:
                kp1_idx = self.keypoint_names.index(kp1_name)
                kp2_idx = self.keypoint_names.index(kp2_name)
                kp3_idx = self.keypoint_names.index(kp3_name)
                
                angle = self.calculate_angle(keypoints[kp1_idx], keypoints[kp2_idx], keypoints[kp3_idx])
                angles[angle_name] = angle
            except (ValueError, IndexError, ZeroDivisionError):
                angles[angle_name] = 0.0
        
        return distances, angles
    
    def visualize_keypoints(self, image, keypoints, distances, angles, show_connections=True, show_labels=True):
        """Create visualization with keypoints, connections, and metrics"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Display image
        ax.imshow(image)
        
        # Plot keypoints
        for i, (kp, name) in enumerate(zip(keypoints, self.keypoint_names)):
            x, y, conf = kp
            if conf > 0.1:  # Only show confident predictions
                ax.plot(x, y, 'ro', markersize=8, alpha=0.8)
                if show_labels:
                    ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Draw skeleton connections
        if show_connections:
            for i, j in self.skeleton_connections:
                if i < len(keypoints) and j < len(keypoints):
                    x1, y1, conf1 = keypoints[i]
                    x2, y2, conf2 = keypoints[j]
                    if conf1 > 0.1 and conf2 > 0.1:
                        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.6)
        
        # Draw measurement lines for selected metrics
        measurement_colors = ['green', 'orange', 'purple', 'cyan', 'pink']
        for idx, (metric_name, distance) in enumerate(list(distances.items())[:5]):
            if metric_name in self.orthodontic_pairs:
                kp1_name, kp2_name = self.orthodontic_pairs[metric_name]
                try:
                    kp1_idx = self.keypoint_names.index(kp1_name)
                    kp2_idx = self.keypoint_names.index(kp2_name)
                    
                    x1, y1, conf1 = keypoints[kp1_idx]
                    x2, y2, conf2 = keypoints[kp2_idx]
                    
                    if conf1 > 0.1 and conf2 > 0.1:
                        color = measurement_colors[idx % len(measurement_colors)]
                        ax.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.8)
                        
                        # Add distance label
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        ax.annotate(f'{distance:.1f}px', (mid_x, mid_y), 
                                   fontsize=8, color=color, weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                except (ValueError, IndexError):
                    continue
        
        ax.set_title('Facial Keypoints Detection with Orthodontic Measurements', fontsize=14)
        ax.axis('off')
        
        return fig
    
    def create_metrics_table(self, distances, angles):
        """Create a formatted table of metrics"""
        # Distance metrics
        distance_df = pd.DataFrame([
            {'Metric': name, 'Value (pixels)': f'{value:.2f}', 'Type': 'Distance'}
            for name, value in distances.items()
        ])
        
        # Angle metrics
        angle_df = pd.DataFrame([
            {'Metric': name, 'Value (degrees)': f'{value:.1f}°', 'Type': 'Angle'}
            for name, value in angles.items()
        ])
        
        return distance_df, angle_df


def main():
    st.set_page_config(
        page_title="HRNet Facial Keypoints Detection",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 HRNet Facial Keypoints Detection Interface")
    st.markdown("Upload an image to detect facial keypoints and calculate orthodontic measurements")
    
    # Initialize interface
    interface = FacialKeypointsInterface()
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Model selection
    model_path = st.sidebar.selectbox(
        "Select Model",
        [
            "checkpoints/best_model.pth",
            "checkpoints/checkpoint_epoch_99.pth",
            "checkpoints/checkpoint_epoch_90.pth"
        ],
        help="Choose a trained model checkpoint"
    )
    
    # Load model button
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            if interface.load_model(model_path):
                st.sidebar.success("Model loaded successfully!")
                st.session_state.model_loaded = True
            else:
                st.sidebar.error("Failed to load model")
                st.session_state.model_loaded = False
    
    # Visualization options
    st.sidebar.header("Visualization Options")
    show_connections = st.sidebar.checkbox("Show Skeleton Connections", value=True)
    show_labels = st.sidebar.checkbox("Show Keypoint Labels", value=True)
    show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=False)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a lateral face image for keypoint detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process image if model is loaded
            if hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded:
                if st.button("Detect Keypoints", type="primary"):
                    with st.spinner("Detecting keypoints..."):
                        try:
                            # Preprocess image
                            image_tensor, processed_image = interface.preprocess_image(image)
                            
                            # Predict keypoints
                            keypoints = interface.predict_keypoints(image_tensor)
                            
                            # Calculate metrics
                            distances, angles = interface.compute_orthodontic_metrics(keypoints)
                            
                            # Create visualization
                            fig = interface.visualize_keypoints(
                                processed_image, keypoints, distances, angles,
                                show_connections, show_labels
                            )
                            
                            st.subheader("Detection Results")
                            st.pyplot(fig)
                            
                            # Store results in session state
                            st.session_state.keypoints = keypoints
                            st.session_state.distances = distances
                            st.session_state.angles = angles
                            st.session_state.processed_image = processed_image
                            
                        except Exception as e:
                            st.error(f"Error during detection: {e}")
            else:
                st.warning("Please load a model first using the sidebar")
    
    with col2:
        st.header("Keypoint Information")
        
        # Display keypoint names
        st.subheader("Facial Landmarks")
        for i, name in enumerate(interface.keypoint_names, 1):
            st.text(f"{i:2d}. {name}")
        
        # Display metrics if available
        if hasattr(st.session_state, 'distances') and hasattr(st.session_state, 'angles'):
            st.subheader("Orthodontic Measurements")
            
            # Distance measurements
            st.markdown("**Distance Measurements:**")
            for name, value in st.session_state.distances.items():
                st.metric(name, f"{value:.2f} px")
            
            # Angle measurements
            st.markdown("**Angle Measurements:**")
            for name, value in st.session_state.angles.items():
                st.metric(name, f"{value:.1f}°")
            
            # Download metrics
            if st.button("Download Metrics as CSV"):
                distance_df, angle_df = interface.create_metrics_table(
                    st.session_state.distances, st.session_state.angles
                )
                
                # Combine dataframes
                combined_df = pd.concat([
                    distance_df[['Metric', 'Value (pixels)']].rename(columns={'Value (pixels)': 'Value'}),
                    angle_df[['Metric', 'Value (degrees)']].rename(columns={'Value (degrees)': 'Value'})
                ], ignore_index=True)
                
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="facial_metrics.csv",
                    mime="text/csv"
                )
    
    # Additional information
    st.markdown("---")
    st.subheader("About This Tool")
    st.markdown("""
    This interface uses HRNet (High-Resolution Network) for facial keypoint detection on lateral face images.
    
    **Features:**
    - 14-point facial landmark detection
    - Real-time orthodontic measurements
    - Interactive visualization
    - Metric export capabilities
    
    **Orthodontic Measurements:**
    - Distance measurements between key facial landmarks
    - Angular measurements for facial analysis
    - Clinical-grade precision for orthodontic assessment
    
    **Usage:**
    1. Load a trained model using the sidebar
    2. Upload a lateral face image
    3. Click "Detect Keypoints" to analyze
    4. View results and download metrics
    """)
    
    # System information
    with st.expander("System Information"):
        st.write(f"Device: {interface.device}")
        st.write(f"PyTorch Version: {torch.__version__}")
        st.write(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()

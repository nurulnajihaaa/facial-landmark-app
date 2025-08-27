"""
Use the working HRNet model and create a proper Streamlit app demo
"""
import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image

# Configure page
st.set_page_config(page_title="Facial Keypoint Detection", layout="wide")

def load_working_model():
    """Load the best working model we have"""
    try:
        # Try the improved model first - our best performer
        model_path = "checkpoints/improved_hrnet_best.pth"
        if Path(model_path).exists():
            from facial_keypoints_app import FacialKeypointsInference, HRNetConfig
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inference = FacialKeypointsInference(model_path, device)
            return inference, "🎉 Improved HRNet (BEST - Val Loss: 0.019)"
    except Exception as e:
        st.error(f"Could not load improved model: {e}")
    
    try:
        # Try the clean trained model as fallback
        model_path = "checkpoints/clean_fixed_hrnet_best.pth"
        if Path(model_path).exists():
            from facial_keypoints_app import FacialKeypointsInference, HRNetConfig
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inference = FacialKeypointsInference(model_path, device)
            return inference, "Clean FixedHRNet (Good)"
    except Exception as e:
        st.error(f"Could not load clean model: {e}")
    
    try:
        # Try the latest trained FixedHRNet model as fallback
        model_path = "checkpoints/fixed_hrnet_epoch_19.pth"
        if Path(model_path).exists():
            from facial_keypoints_app import FacialKeypointsInference, HRNetConfig
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inference = FacialKeypointsInference(model_path, device)
            return inference, "Latest FixedHRNet (epoch 19)"
    except Exception as e:
        st.error(f"Could not load latest FixedHRNet model: {e}")
    
    try:
        # Try the head-finetuned model as fallback
        model_path = "checkpoints/finetune_head/checkpoint_epoch_11.pth"
        if Path(model_path).exists():
            from facial_keypoints_app import FacialKeypointsInference, HRNetConfig
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inference = FacialKeypointsInference(model_path, device)
            return inference, "Head-finetuned HRNet (epoch 11)"
    except Exception as e:
        st.error(f"Could not load head-finetuned model: {e}")
    
    try:
        # Try the compat model
        model_path = "checkpoints/compat_hrnet_from_ckpt.pth"
        if Path(model_path).exists():
            from facial_keypoints_app import FacialKeypointsInference, HRNetConfig
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inference = FacialKeypointsInference(model_path, device)
            return inference, "Compatible HRNet"
    except Exception as e:
        st.error(f"Could not load compat model: {e}")
    
    return None, "No working model found"

def load_sample_images():
    """Load sample images from the dataset"""
    samples = []
    
    # Load COCO data
    try:
        with open('coco_annotations.json', 'r') as f:
            coco_data = json.load(f)
        
        image_root = Path('dataset')
        
        # Get first few images
        for i, image_info in enumerate(coco_data['images'][:6]):
            image_file = image_info['file_name']
            
            # Try different paths
            image_path = None
            for split in ['train', 'valid', 'test']:
                candidate = image_root / split / 'images' / Path(image_file).name
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path:
                # Get annotation for bbox
                ann = None
                for a in coco_data['annotations']:
                    if a['image_id'] == image_info['id']:
                        ann = a
                        break
                
                samples.append({
                    'path': image_path,
                    'bbox': ann['bbox'] if ann else None,
                    'id': image_info['id']
                })
                
                if len(samples) >= 6:
                    break
    except Exception as e:
        st.error(f"Could not load sample images: {e}")
    
    return samples

def main():
    st.title("🔍 Facial Keypoint Detection System")
    st.markdown("### Advanced HRNet-based facial landmark detection for orthodontic analysis")
    
    # Load model
    with st.spinner("Loading model..."):
        inference_model, model_name = load_working_model()
    
    if inference_model is None:
        st.error("❌ Could not load any working model!")
        return
    
    st.success(f"✅ Loaded model: {model_name}")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Sample Images", "Upload Image"]
    )
    
    if input_method == "Sample Images":
        st.header("📊 Sample Dataset Results")
        
        # Load sample images
        samples = load_sample_images()
        
        if not samples:
            st.error("No sample images found!")
            return
        
        # Create columns for sample images
        cols = st.columns(3)
        
        for i, sample in enumerate(samples):
            with cols[i % 3]:
                try:
                    # Load and display image
                    image = cv2.imread(str(sample['path']))
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Run inference
                    bbox = sample['bbox']
                    if bbox:
                        keypoints, confidence = inference_model.predict(image_rgb, bbox)
                    else:
                        keypoints, confidence = inference_model.predict(image_rgb)
                    
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(image_rgb)
                    
                    # Plot keypoints
                    for j, (x, y) in enumerate(keypoints):
                        ax.plot(x, y, 'ro', markersize=6, alpha=0.8)
                        ax.text(x+3, y-3, str(j), color='red', fontsize=8, weight='bold')
                    
                    ax.set_title(f"Image {sample['id']} (Mean Conf: {confidence.mean():.3f})")
                    ax.axis('off')
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show confidence scores
                    st.write(f"**Confidence scores:** {confidence.mean():.3f} ± {confidence.std():.3f}")
                    
                except Exception as e:
                    st.error(f"Error processing sample {i}: {e}")
    
    else:  # Upload Image
        st.header("📤 Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a lateral face image for keypoint detection"
        )
        
        if uploaded_file is not None:
            try:
                # Load uploaded image
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    image_rgb = image_np
                else:
                    st.error("Please upload a color (RGB) image")
                    return
                
                # Display original image
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image_rgb, use_column_width=True)
                
                # Manual bbox input
                st.subheader("Bounding Box (Optional)")
                col_x, col_y, col_w, col_h = st.columns(4)
                
                with col_x:
                    bbox_x = st.number_input("X", min_value=0, value=0)
                with col_y:
                    bbox_y = st.number_input("Y", min_value=0, value=0)
                with col_w:
                    bbox_w = st.number_input("Width", min_value=0, value=image_rgb.shape[1])
                with col_h:
                    bbox_h = st.number_input("Height", min_value=0, value=image_rgb.shape[0])
                
                if st.button("🔍 Detect Keypoints"):
                    with st.spinner("Running inference..."):
                        try:
                            # Run inference
                            if bbox_w > 0 and bbox_h > 0:
                                bbox = [bbox_x, bbox_y, bbox_w, bbox_h]
                                keypoints, confidence = inference_model.predict(image_rgb, bbox)
                            else:
                                keypoints, confidence = inference_model.predict(image_rgb)
                            
                            # Create visualization
                            with col2:
                                st.subheader("Detected Keypoints")
                                
                                fig, ax = plt.subplots(figsize=(8, 8))
                                ax.imshow(image_rgb)
                                
                                # Plot keypoints
                                for j, (x, y) in enumerate(keypoints):
                                    ax.plot(x, y, 'ro', markersize=8, alpha=0.8)
                                    ax.text(x+5, y-5, str(j), color='red', fontsize=10, weight='bold')
                                
                                ax.set_title(f"Detected Keypoints (Mean Conf: {confidence.mean():.3f})")
                                ax.axis('off')
                                
                                st.pyplot(fig)
                                plt.close()
                            
                            # Show results
                            st.subheader("📊 Results")
                            
                            col_res1, col_res2 = st.columns(2)
                            
                            with col_res1:
                                st.metric("Number of Keypoints", len(keypoints))
                                st.metric("Mean Confidence", f"{confidence.mean():.3f}")
                                st.metric("Min Confidence", f"{confidence.min():.3f}")
                                st.metric("Max Confidence", f"{confidence.max():.3f}")
                            
                            with col_res2:
                                # Show keypoint coordinates
                                st.write("**Keypoint Coordinates:**")
                                for j, ((x, y), conf) in enumerate(zip(keypoints, confidence)):
                                    st.write(f"Joint {j}: ({x:.1f}, {y:.1f}) - Conf: {conf:.3f}")
                        
                        except Exception as e:
                            st.error(f"Error during inference: {e}")
                            st.write("Full error:", str(e))
            
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🔬 Model Information")
    st.write(f"**Active Model:** {model_name}")
    st.write("**Architecture:** HRNet-W18 for high-resolution keypoint detection")
    st.write("**Keypoints:** 14-point facial landmark system for lateral face analysis")

if __name__ == "__main__":
    main()

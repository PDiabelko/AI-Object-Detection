"""
YOLO Streamlit Detection App

Features included:
1) Confidence slider (adjust min detection confidence)
2) Upload image or use camera
3) Side panel showing detected classes + confidence scores
4) Table of bounding boxes (x1, y1, x2, y2, class, confidence)
5) Save output functionality
6) Camera capture with automatic detection every 2 seconds
7) Stop & Exit to clean up images

Requirements:
    pip install streamlit ultralytics pillow numpy opencv-python pandas
Run:
    streamlit run app_streamlit.py
"""

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os
import time
import cv2
import pandas as pd
import shutil
from io import BytesIO

# ---------- Configuration ----------
MODEL_PATH = "./best.pt"
IMAGES_FOLDER = "images"
# -----------------------------------

def ensure_list(x):
    try:
        return list(x)
    except Exception:
        return x

@st.cache_resource
def load_model():
    """Load YOLO model (cached)"""
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Could not load model at {MODEL_PATH}.\nError: {e}")
        st.stop()

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = []
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'annotated_image' not in st.session_state:
        st.session_state.annotated_image = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    
    # Create images folder
    os.makedirs(IMAGES_FOLDER, exist_ok=True)

def run_detection(model, image, conf_threshold):
    """Run YOLO detection on image"""
    try:
        # Save image temporarily for YOLO processing
        temp_path = os.path.join(IMAGES_FOLDER, f"temp_{int(time.time())}.jpg")
        image.save(temp_path)
        
        # Run inference
        results = model(temp_path, conf=conf_threshold, imgsz=1280)
        res0 = results[0]
        
        # Get boxes, classes, confidences
        boxes_xyxy = ensure_list(res0.boxes.xyxy.tolist()) if hasattr(res0.boxes, "xyxy") else []
        confs = ensure_list(res0.boxes.conf.tolist()) if hasattr(res0.boxes, "conf") else []
        cls_idxs = ensure_list(res0.boxes.cls.tolist()) if hasattr(res0.boxes, "cls") else []
        
        # Build detected info
        detections = []
        class_names = model.names if hasattr(model, "names") else {}
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = [int(round(v)) for v in boxes_xyxy[i]]
            cls_idx = int(cls_idxs[i])
            cls_name = class_names.get(cls_idx, str(cls_idx))
            conf_score = float(confs[i])
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class": cls_name, "conf": conf_score
            })
        
        # Get annotated image
        plotted = res0.plot()
        if isinstance(plotted, np.ndarray):
            # Convert BGR to RGB
            annotated = Image.fromarray(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
        else:
            annotated = image
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return detections, annotated
    
    except Exception as e:
        st.error(f"Detection error: {e}")
        return [], image

def create_detection_summary(detections):
    """Create summary of detected objects"""
    if not detections:
        return "No objects detected"
    
    class_count = {}
    for d in detections:
        cls = d['class']
        conf = d['conf']
        if cls not in class_count:
            class_count[cls] = []
        class_count[cls].append(conf)
    
    summary = []
    for cls, confs in class_count.items():
        avg_conf = sum(confs) / len(confs)
        summary.append(f"**{cls}**: {len(confs)} (avg conf: {avg_conf:.2f})")
    
    return "\n".join(summary)

def stop_and_exit():
    """Delete images folder and stop app"""
    try:
        if os.path.exists(IMAGES_FOLDER):
            shutil.rmtree(IMAGES_FOLDER)
            st.success(f"Deleted images folder: {IMAGES_FOLDER}")
    except Exception as e:
        st.error(f"Error deleting images folder: {e}")
    
    st.session_state.clear()
    st.stop()

def main():
    st.set_page_config(
        page_title="YOLO Object Detection",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("YOLO Object Detection App")
    
    # Initialize
    initialize_session_state()
    model = load_model()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Confidence slider
    conf_threshold = st.sidebar.slider(
        "Min Confidence",
        min_value=0.01,
        max_value=0.99,
        value=0.25,
        step=0.01
    )
    
    st.sidebar.markdown("---")
    
    # Upload or Camera selection
    input_mode = st.sidebar.radio(
        "Input Mode",
        ["Upload Image", "Camera Capture"]
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    if input_mode == "Upload Image":
        st.session_state.camera_active = False
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.current_image = image
            
            if st.sidebar.button("Run Detection", type="primary"):
                with st.spinner("Running detection..."):
                    detections, annotated = run_detection(model, image, conf_threshold)
                    st.session_state.detection_results = detections
                    st.session_state.annotated_image = annotated
            
            # Display images
            with col1:
                st.subheader("Detection Result")
                if st.session_state.annotated_image is not None:
                    st.image(st.session_state.annotated_image, width='stretch')
                else:
                    st.image(image, width='stretch')
                    st.info("Click 'Run Detection' to process the image")
            
            # Display detection info
            with col2:
                if st.session_state.detection_results is not None:
                    st.subheader("Detection Summary")
                    st.markdown(create_detection_summary(st.session_state.detection_results))
                    
                    st.subheader(f"Objects Detected: {len(st.session_state.detection_results)}")
                    
                    # Show individual detections
                    for idx, det in enumerate(st.session_state.detection_results):
                        st.write(f"{idx+1}. **{det['class']}** — {det['conf']:.2f}")
                    
                    # Bounding boxes table
                    st.subheader("Bounding Boxes")
                    df = pd.DataFrame(st.session_state.detection_results)
                    df['conf'] = df['conf'].round(3)
                    st.dataframe(df, width='stretch')
                    
                    # Save button
                    if st.session_state.annotated_image is not None:
                        buf = BytesIO()
                        st.session_state.annotated_image.save(buf, format="JPEG")
                        st.download_button(
                            label="💾 Download Annotated Image",
                            data=buf.getvalue(),
                            file_name=f"detected_{int(time.time())}.jpg",
                            mime="image/jpeg"
                        )
    
    elif input_mode == "Camera Capture":
        st.sidebar.markdown("### Camera Controls")
        
        col_start, col_stop = st.sidebar.columns(2)
        
        with col_start:
            if st.button("▶️ Start", width='stretch'):
                st.session_state.camera_active = True
                st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop", width='stretch'):
                st.session_state.camera_active = False
                st.rerun()
        
        if st.session_state.camera_active:
            st.sidebar.info("📹 Camera is active - capturing every 2 seconds")
            
            # Camera capture
            camera = cv2.VideoCapture(0)
            
            if not camera.isOpened():
                st.error("Could not access camera. Please check if it's connected.")
                st.session_state.camera_active = False
            else:
                # Display placeholders
                image_placeholder = col1.empty()
                info_placeholder = col2.empty()
                
                capture_interval = 2  # seconds
                detection_delay = 1  # seconds
                
                while st.session_state.camera_active:
                    # Capture frame
                    ret, frame = camera.read()
                    if not ret:
                        st.error("Failed to capture frame")
                        break
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Save to images folder
                    timestamp = int(time.time() * 1000)
                    image_path = os.path.join(IMAGES_FOLDER, f"capture_{timestamp}.jpg")
                    pil_image.save(image_path)
                    st.session_state.captured_images.append(image_path)
                    
                    # Display captured image
                    with image_placeholder.container():
                        st.image(pil_image, caption="Live Capture", width='stretch')
                    
                    # Wait before detection
                    time.sleep(detection_delay)
                    
                    # Run detection
                    with info_placeholder.container():
                        with st.spinner("Running detection..."):
                            detections, annotated = run_detection(model, pil_image, conf_threshold)
                            
                            # Display annotated image
                            image_placeholder.empty()
                            with image_placeholder.container():
                                st.image(annotated, caption="Detection Result", width='stretch')
                            
                            # Display results
                            st.subheader("Detection Summary")
                            st.markdown(create_detection_summary(detections))
                            
                            st.subheader(f"Objects Detected: {len(detections)}")
                            for idx, det in enumerate(detections):
                                st.write(f"{idx+1}. **{det['class']}** — {det['conf']:.2f}")
                    
                    # Wait for rest of interval
                    time.sleep(capture_interval - detection_delay)
                
                camera.release()
        else:
            with col1:
                st.info("Click 'Start' to begin camera capture")
    
    # Sidebar footer buttons
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🛑 Clear Images", type="secondary", width='stretch'):
        stop_and_exit()

if __name__ == "__main__":
    main()

"""
YOLO Streamlit Detection App

Features included:

Requirements:
    pip install streamlit ultralytics pillow numpy opencv-python pandas
Run:
    streamlit run crosswalk.py
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

def render_leds(left_status, right_status, alternating=False):
    """
    Renders two LED circles.
    
    Args:
        left_status (str): 'flash' to flash yellow, 'off' to turn off.
        right_status (str): 'flash' to flash yellow, 'off' to turn off.
        alternating (bool): If True, flashes LEDs in opposite phases.
    """
    # CSS for the LEDs
    st.markdown("""
    <style>
    .led-container {
        display: flex;
        justify_content: center;
        gap: 50px;
        padding: 20px;
        background-color: #1E1E1E;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .led {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background-color: #333; /* Default off color */
        border: 3px solid #555;
        transition: all 0.3s ease;
    }
    
    /* Flashing Yellow State */
    .led.flash {
        background-color: #FFD700; /* Gold/Yellow */
        box-shadow: 0 0 20px #FFD700, 0 0 40px #FFD700;
        border-color: #FFD700;
        animation: blink 1s infinite;
    }

    /* Alternating Delay */
    .led.flash.delay {
        animation-delay: 0.5s; /* Half of the 1s duration */
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; transform: scale(1); background-color: #FFD700; }
        50% { opacity: 0.2; transform: scale(0.95); background-color: #333; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Determine classes based on status
    left_class = "flash" if left_status == "flash" else ""
    right_class = "flash" if right_status == "flash" else ""
    
    # Apply delay to right LED if alternating mode is on
    if alternating and right_class == "flash":
        right_class += " delay"
    
    # HTML structure
    st.markdown(f"""
    <div class="led-container">
        <div style="text-align: center;">
            <div class="led {left_class}"></div>
            <p style="margin-top: 5px; color: #888;">Left</p>
        </div>
        <div style="text-align: center;">
            <div class="led {right_class}"></div>
            <p style="margin-top: 5px; color: #888;">Right</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

def clear_images():
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
        page_title="Crosswalk Detection App",
        page_icon="🚸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("AI Detection Crosswalk App")

    # Initialize
    initialize_session_state()
    model = load_model()

    # Initialize session state for LEDs if not exists
    if 'left_led' not in st.session_state:
        st.session_state.left_led = 'off'
    if 'right_led' not in st.session_state:
        st.session_state.right_led = 'off'
    if 'alternating_mode' not in st.session_state:
        st.session_state.alternating_mode = False

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
    st.sidebar.markdown("### Camera Controls")

    col_start, col_stop = st.sidebar.columns(2)
    
    with col_start:
        if st.button("▶️ Start", width='stretch'):
            st.session_state.camera_active = True
            st.rerun()
    
    with col_stop:
        if st.button("⏹️ Stop", width='stretch'):
            st.session_state.camera_active = False
            st.session_state.right_led = 'off'
            st.session_state.left_led = 'off'
            st.session_state.alternating_mode = False
            st.rerun()

    if st.sidebar.button("🛑 Clear Images", type="secondary", width='stretch'):
        clear_images()

    # Main Layout
    col1, col2 = st.columns([2, 1])

    # LED Display (Right Column)
    with col2:
        st.subheader("LED Status")
        led_placeholder = st.empty()
        # Render current state
        with led_placeholder.container():
            render_leds(
                st.session_state.left_led, 
                st.session_state.right_led, 
                alternating=st.session_state.alternating_mode
            )
        
        st.markdown("---")
        st.subheader("Manual Controls")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Turn Off", width='stretch'):
                st.session_state.right_led = 'off'
                st.session_state.left_led = 'off'
                st.session_state.alternating_mode = False
                st.rerun()
        with c2:
            if st.button("Alternating", width='stretch'):
                st.session_state.left_led = 'flash'
                st.session_state.right_led = 'flash'
                st.session_state.alternating_mode = True
                st.rerun()
        
        stats_placeholder = st.empty()

    # Camera Feed (Left Column)
    with col1:
        st.subheader("Camera Feed")
        image_placeholder = st.empty()
        
        if not st.session_state.camera_active:
            st.info("Click 'Start' in the sidebar to begin camera capture")
            # Optional: Show a static image or placeholder
            image_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Camera Off", channels="RGB")

    if st.session_state.camera_active:
        # Camera capture loop
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            st.error("Could not access camera. Please check if it's connected.")
            st.session_state.camera_active = False
        else:
            while st.session_state.camera_active:
                # Capture frame
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Run detection
                detections, annotated_img = run_detection(model, pil_image, conf_threshold)
                
                # Update Image
                image_placeholder.image(annotated_img, caption="Live Feed", width='stretch')
                
                # Update Stats
                with stats_placeholder.container():
                    st.subheader("Detection Summary")
                    st.markdown(create_detection_summary(detections))
                
                # Check for person detection to trigger LEDs
                person_detected = any(d['class'] == "person" for d in detections)
                
                if person_detected and not st.session_state.alternating_mode:
                    st.session_state.left_led = 'flash'
                    st.session_state.right_led = 'flash'
                    st.session_state.alternating_mode = True
                    # Update LED placeholder immediately
                    with led_placeholder.container():
                        render_leds('flash', 'flash', alternating=True)

                elif not person_detected and st.session_state.alternating_mode:
                    st.session_state.left_led = 'off'
                    st.session_state.right_led = 'off'
                    st.session_state.alternating_mode = False
                    # Update LED placeholder immediately
                    with led_placeholder.container():
                        render_leds('off', 'off', alternating=False)
                
                # Small delay to prevent UI freezing
                time.sleep(0.05)
            
            camera.release()

if __name__ == "__main__":
    main()
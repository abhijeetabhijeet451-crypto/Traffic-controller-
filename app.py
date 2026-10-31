import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time

st.set_page_config(page_title="AI Smart Traffic Controller", layout="wide", page_icon="🚦")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Welcome Page", "Live Simulation"])

# ==========================================
# PAGE 1: NORMAL WELCOME PAGE
# ==========================================
if page == "Welcome Page":
    st.title("🚦 AI-Powered Smart Traffic Controller")
    st.write("An Intelligent Real-Time Traffic Density Management & Emergency Vehicle Priority System")
    
    st.divider()

    st.header("Project Overview")
    st.write(
        "This prototype uses computer vision and deep learning (YOLOv8) to analyze traffic video feeds "
        "in real-time. It automatically optimizes traffic signal durations based on vehicle density, "
        "detects emergency vehicles for priority green corridors, and flags stray animal hazards on the road."
    )

    st.header("Key Features")
    st.markdown("- **Dynamic Signal Timing:** Automatically calculates green light duration based on vehicle count.")
    st.markdown("- **Emergency Vehicle Override:** Detects ambulances and fire trucks to trigger a 60-second green corridor.")
    st.markdown("- **Stray Animal Detection:** Identifies cows, dogs, or other animals to alert authorities of road hazards.")

    st.header("Tech Stack")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "YOLOv8 Nano")
    col2.metric("Processing", "OpenCV")
    col3.metric("Framework", "Streamlit")
    col4.metric("Dataset", "COCO")

    st.info("To test the system, select 'Live Simulation' from the sidebar menu and upload a traffic video clip.")

# ==========================================
# PAGE 2: LIVE SIMULATION
# ==========================================
elif page == "Live Simulation":
    st.title("Live Traffic Simulation & Analytics")
    st.write("Upload a traffic video clip to run object detection and view real-time signal logic.")

    @st.cache_resource
    def load_model():
        return YOLO("yolov8n.pt")

    model = load_model()

    st.sidebar.header("Controls")
    video_source = st.sidebar.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Video Feed")
        video_placeholder = st.empty()

    with col2:
        st.subheader("Signal Status")
        status_card = st.empty()
        timer_card = st.empty()
        stats_card = st.empty()

    VEHICLE_CLASSES = [2, 3, 5, 7]
    EMERGENCY_CLASSES = [7]
    ANIMAL_CLASSES = [15, 16, 17, 18, 19, 20]

    if video_source is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_source.read())
        cap = cv2.VideoCapture(tfile.name)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            boxes = results[0].boxes

            vehicle_count = 0
            emergency_detected = False
            animal_detected = False

            for box in boxes:
                cls = int(box.cls[0])
                if cls in VEHICLE_CLASSES:
                    vehicle_count += 1
                if cls in EMERGENCY_CLASSES:
                    emergency_detected = True
                if cls in ANIMAL_CLASSES:
                    animal_detected = True

            annotated_file = results[0].plot()

            if emergency_detected:
                signal = "GREEN: EMERGENCY CORRIDOR OVERRIDE"
                advice = "Emergency vehicle detected. Corridor cleared."
                allocated_time = 60
            elif animal_detected:
                signal = "YELLOW: ANIMAL HAZARD CAUTION"
                advice = "Stray animal detected on road."
                allocated_time = 15
            elif vehicle_count > 4:
                signal = "GREEN: HIGH DENSITY TRAFFIC"
                advice = "Dynamic timing active for high traffic."
                allocated_time = min(20 + (vehicle_count * 3), 60)
            else:
                signal = "RED: STANDARD INTERVAL"
                advice = "Low traffic density."
                allocated_time = 15

            video_placeholder.image(cv2.cvtColor(annotated_file, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            status_card.info(f"**Status:** {signal}\n\n**Action:** {advice}")
            timer_card.metric(label="Signal Timer", value=f"{allocated_time}s")
            stats_card.write(f"Vehicles: {vehicle_count} | Emergency: {'Yes' if emergency_detected else 'No'} | Animal: {'Yes' if animal_detected else 'No'}")
            time.sleep(0.03)

        cap.release()
    else:
        st.warning("Please upload a video file from the sidebar to begin.")

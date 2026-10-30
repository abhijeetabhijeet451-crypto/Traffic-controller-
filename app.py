import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

st.set_page_config(page_title="AI Smart Traffic Controller", layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

if st.session_state.page == 'welcome':
    st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>AI-Powered Next-Gen Traffic Management System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gray;'>Smart Traffic Density Optimization and Emergency Corridor System</h3>", unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Welcome to the Hackathon Project. Click the button below to start the live traffic simulation")
        if st.button("Start Traffic Simulation", use_container_width=True):
            st.session_state.page = 'simulation'
            st.rerun()

elif st.session_state.page == 'simulation':
    if st.button("Back to Home"):
        st.session_state.page = 'welcome'
        st.rerun()

    st.title("AI Smart Traffic Controller - Live Dashboard")
    st.caption("Vehicle Density Optimization | Emergency Vehicle Detection | Stray Animal Protection")

    st.sidebar.header("Simulation Controls")
    video_source = st.sidebar.file_uploader("Upload Traffic Video Clip", type=["mp4", "avi", "mov"])

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Video Feed and AI Bounding Boxes")
        video_placeholder = st.empty()

    with col2:
        st.subheader("Real-Time Signal Analytics")
        status_card = st.empty()
        timer_card = st.empty()
        stats_card = st.empty()

    VEHICLE_CLASSES = [2, 3, 5, 7]
    EMERGENCY_CLASSES = [7]
    ANIMAL_CLASSES = [15, 16, 17, 18, 19, 20]

    if video_source is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_source.read())
        cap = cv2.VideoCapture(tfile.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

        vehicle_count = 0
        emergency_detected = False
        animal_detected = False
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 3 != 0:
                continue

            results = model(frame, verbose=False)
            boxes = results[0].boxes

            for box in boxes:
                cls = int(box.cls[0])
                if cls in VEHICLE_CLASSES:
                    vehicle_count += 1
                if cls in EMERGENCY_CLASSES:
                    emergency_detected = True
                if cls in ANIMAL_CLASSES:
                    animal_detected = True

            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        video_placeholder.video(out_file.name)

        if emergency_detected:
            signal = "GREEN (EMERGENCY CORRIDOR OVERRIDE)"
            color = "#28a745"
            advice = "Emergency Vehicle Detected. Green corridor opened automatically"
            allocated_time = 60
        elif animal_detected:
            signal = "YELLOW (ANIMAL CAUTION HAZARD)"
            color = "#ffc107"
            advice = "Stray Animal Detected on Road. Hazard Alert Active"
            allocated_time = 15
        elif vehicle_count > 4:
            signal = "GREEN (HIGH DENSITY)"
            color = "#28a745"
            advice = "Heavy Traffic Cleared Dynamically"
            allocated_time = min(20 + (vehicle_count * 3), 60)
        else:
            signal = "RED (LOW TRAFFIC / STANDARD)"
            color = "#dc3545"
            advice = "Standard Interval Active"
            allocated_time = 15

        status_card.markdown(
            f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 8px; color: black; font-weight: bold; text-align: center; font-size: 20px;">
                {signal}
            </div>
            <p style="margin-top: 8px; font-weight: bold;">Notice: {advice}</p>
            """,
            unsafe_allow_html=True
        )

        timer_card.metric(label="Calculated Signal Timer", value=f"{allocated_time}s")
        stats_card.write(f"**Vehicles:** {vehicle_count} | **Emergency:** {'Yes 🚨' if emergency_detected else 'None'} | **Animal:** {'Yes ⚠️' if animal_detected else 'None'}")

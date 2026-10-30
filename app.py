import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

st.set_page_config(page_title="AI Smart Traffic Controller", layout="wide")

st.title("🚦 AI Smart Traffic Controller")
st.caption("Vehicle Density Optimization | Emergency Vehicle Detection | Stray Animal Protection")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.sidebar.header("Simulation Controls")
video_source = st.sidebar.file_uploader("Upload Traffic Video Clip", type=["mp4", "avi", "mov"])

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Processed Traffic Video Feed")
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
    with st.spinner("Processing video... Please wait."):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_source.read())
        cap = cv2.VideoCapture(tfile.name)

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360

        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))

        vehicle_count = 0
        emergency_detected = False
        animal_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            boxes = results[0].boxes

            current_vehicles = 0
            for box in boxes:
                cls = int(box.cls[0])
                if cls in VEHICLE_CLASSES:
                    current_vehicles += 1
                if cls in EMERGENCY_CLASSES:
                    emergency_detected = True
                if cls in ANIMAL_CLASSES:
                    animal_detected = True

            if current_vehicles > vehicle_count:
                vehicle_count = current_vehicles

            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

    # Native video player (smooth play/pause/scrub controls)
    video_placeholder.video(output_file.name)

    if emergency_detected:
        signal = "🟢 GREEN (EMERGENCY CORRIDOR OVERRIDE)"
        color = "#28a745"
        advice = "Emergency Vehicle Detected! Green corridor opened automatically."
        allocated_time = 60
    elif animal_detected:
        signal = "🟡 YELLOW (ANIMAL CAUTION HAZARD)"
        color = "#ffc107"
        advice = "Stray Animal (Cow/Dog) Detected on Road! Hazard Alert Active."
        allocated_time = 15
    elif vehicle_count > 4:
        signal = "🟢 GREEN (HIGH DENSITY)"
        color = "#28a745"
        advice = "Heavy Traffic Cleared Dynamically."
        allocated_time = min(20 + (vehicle_count * 3), 60)
    else:
        signal = "🔴 RED (LOW TRAFFIC / STANDARD)"
        color = "#dc3545"
        advice = "Standard Interval Active."
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

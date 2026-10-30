import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time

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
    st.subheader("Live Video Feed & AI Bounding Boxes")
    video_placeholder = st.empty()

with col2:
    st.subheader("Real-Time Signal Analytics")
    status_card = st.empty()
    timer_card = st.empty()
    stats_card = st.empty()

VEHICLE_CLASSES = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
EMERGENCY_CLASSES = [7]          # Truck (Emergency vehicle simulation)
ANIMAL_CLASSES = [15, 16, 17, 18, 19, 20]  # Cat, Dog, Horse, Sheep, Cow, Elephant

if video_source is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_source.read())
    cap = cv2.VideoCapture(tfile.name)

    while cap.isOpened():
        ret, frame = cap.read()
        # Frame ka size chota karke fast karna
frame = cv2.resize(frame, (640, 360))
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

        video_placeholder.image(cv2.cvtColor(annotated_file, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

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
        time.sleep(0.03)

    cap.release()

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time

st.set_page_config(page_title="AI Smart Traffic Controller", layout="wide", page_icon="🚦")

# --- Sidebar Navigation ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Welcome Page", "🚦 Live AI Simulation"])

# ==========================================
# PAGE 1: WELCOME / FRONT PAGE
# ==========================================
if page == "🏠 Welcome Page":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 35px; border-radius: 15px; color: white; text-align: center; margin-bottom: 25px;">
            <h1 style="font-size: 42px; margin-bottom: 10px;">🚦 AI-Powered Next-Gen Smart Traffic Controller</h1>
            <p style="font-size: 19px; opacity: 0.95;">
                An Intelligent Real-Time Traffic Density Management & Emergency Vehicle Priority System
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🌟 Key Prototype Capabilities")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style="background-color: #1f242d; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
                <h3>🚗 Dynamic Signal Timing</h3>
                <p>Computer vision counts real-time vehicles and dynamically sets optimal green light intervals to clear congestion.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="background-color: #1f242d; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
                <h3>🚨 Emergency Priority</h3>
                <p>Detects ambulances, fire engines, and police vehicles immediately to open automatic green corridors (60s override).</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div style="background-color: #1f242d; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107;">
                <h3>⚠️ Stray Animal Alert</h3>
                <p>Identifies stray animals (cows, dogs) on roads to trigger hazard caution signals and prevent fatal accidents.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("🛠️ Technical Architecture")
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    tech_col1.metric("Object Detection", "YOLOv8 Nano")
    tech_col2.metric("Computer Vision", "OpenCV")
    tech_col3.metric("Framework", "Streamlit UI")
    tech_col4.metric("Dataset Classes", "COCO Vehicles & Animals")

    st.info("👉 **How to Test:** Select **'🚦 Live AI Simulation'** from the left sidebar and upload a traffic video clip to view the live detector in action!")

# ==========================================
# PAGE 2: LIVE SIMULATION
# ==========================================
elif page == "🚦 Live AI Simulation":
    st.title("🚦 Live AI Simulation & Analytics")
    st.caption("Vehicle Density Optimization | Emergency Corridor Override | Stray Animal Protection")

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
    else:
        st.warning("Please upload a traffic video clip from the sidebar to start the simulation.")

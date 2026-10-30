if video_source is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_source.read())
        cap = cv2.VideoCapture(tfile.name)

        frame_skip = 5  # Har 5th frame process hoga taki lag bilkul khatam ho jaye
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # Frame ka size chota karne se AI processing 3x fast ho jati hai
            frame = cv2.resize(frame, (640, 360))

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

        cap.release()

import streamlit as st
import cv2
import numpy as np
from utils.face_engine import register_employee, get_embedding

def show():
    st.title("📝 Register New Employee")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Full Name", placeholder="e.g. Rahul Sharma")
        department = st.selectbox(
            "Department",
            ["Security", "HR", "IT", "Sales", "Operations", "Management"]
        )

    st.markdown("### 📸 Capture Face Photos")
    st.info("We will capture 5 photos automatically. Look straight at camera.")

    if st.button("🎥 Start Capture", type="primary"):
        if not name:
            st.error("Please enter employee name first")
            return

        frames = []
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Camera not found. Check connection.")
            return

        placeholder = st.empty()
        progress = st.progress(0)
        status = st.empty()

        count = 0
        total = 5
        frame_skip = 0

        while count < total:
            ret, frame = cap.read()
            if not ret:
                break

            frame_skip += 1

            # Show live feed
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(rgb, channels="RGB", use_column_width=True)

            # Capture every 15 frames to get varied photos
            if frame_skip % 15 == 0:
                emb, bbox = get_embedding(frame)
                if emb is not None:
                    # Draw green box
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    frames.append(frame.copy())
                    count += 1
                    progress.progress(count / total)
                    status.success(f"✅ Photo {count}/{total} captured")
                else:
                    status.warning("⚠️ No face detected — adjust position")

        cap.release()
        placeholder.empty()

        if len(frames) < 3:
            st.error("Could not capture enough photos. Try again with better lighting.")
            return

        # Register
        with st.spinner("🧠 Processing face data..."):
            success, result = register_employee(name, department, frames)

        if success:
            st.success(f"✅ Employee registered successfully!")
            st.balloons()
            col1, col2, col3 = st.columns(3)
            col1.metric("Employee ID", result)
            col2.metric("Name", name)
            col3.metric("Department", department)
        else:
            st.error(f"Registration failed: {result}")
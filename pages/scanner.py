import streamlit as st
import cv2
import numpy as np
import time
from utils.face_engine import recognize_face, log_attendance, get_employee_visit_count

def show():
    st.title("📷 Live Face Scanner")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Camera Feed")
        run = st.checkbox("▶️ Start Scanner", value=False)
        placeholder = st.empty()

    with col2:
        st.markdown("### Last Detection")
        result_box = st.empty()
        log_box = st.empty()

    if not run:
        placeholder.info("Check the box above to start the scanner")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera not found.")
        return

    last_logged = {}  # emp_id -> last log timestamp (prevent duplicate logs)
    COOLDOWN = 5      # seconds between logs for same person

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        employee, score, bbox = recognize_face(frame)
        display = frame.copy()

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            if employee:
                # Green box — known person
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"{employee['name']} ({score*100:.1f}%)"
                cv2.putText(display, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Log with cooldown to avoid spam
                emp_id = employee['id']
                now = time.time()
                last_time = last_logged.get(emp_id, 0)

                if now - last_time > COOLDOWN:
                    event = log_attendance(employee)
                    last_logged[emp_id] = now
                    ins, outs = get_employee_visit_count(emp_id)

                    result_box.success(
                        f"✅ **{employee['name']}**\n\n"
                        f"Department: {employee['department']}\n\n"
                        f"Event: **{event}**\n\n"
                        f"Confidence: {score*100:.1f}%\n\n"
                        f"Today — IN: {ins} | OUT: {outs}"
                    )

            else:
                # Red box — unknown
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(display, "UNKNOWN PERSON", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                result_box.error(
                    "🚨 **Unknown Person Detected**\n\n"
                    f"Confidence: {score*100:.1f}%\n\n"
                    "Not registered in system"
                )

        else:
            result_box.info("👀 Scanning for faces...")

        # Show frame
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        placeholder.image(rgb, channels="RGB", use_column_width=True)

        # Stop if checkbox unchecked
        run = st.session_state.get("▶️ Start Scanner", True)

    cap.release()
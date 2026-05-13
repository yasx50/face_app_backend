import streamlit as st
import pandas as pd
from datetime import datetime
from utils.face_engine import load_attendance, load_employees

def show():
    st.title("📊 Attendance Dashboard")
    st.markdown("---")

    attendance = load_attendance()
    employees = load_employees()

    if len(attendance) == 0:
        st.warning("No attendance records yet. Use the Scanner to record attendance.")
        return

    df = pd.DataFrame(attendance)
    today = datetime.now().strftime("%Y-%m-%d")

    # ── Top metrics ──────────────────────────────────────────────
    today_df = df[df['date'] == today]
    total_employees = len(employees)
    present_today = today_df['employee_id'].nunique()
    total_events = len(today_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", total_employees)
    col2.metric("Present Today", present_today)
    col3.metric("Total Events Today", total_events)
    col4.metric("Absent Today", total_employees - present_today)

    st.markdown("---")

    # ── Filter ───────────────────────────────────────────────────
    st.markdown("### 🔍 Filter Records")
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_date = st.date_input("Select Date", datetime.now())

    with col2:
        dept_list = ["All"] + list(set(df['department'].tolist()))
        selected_dept = st.selectbox("Department", dept_list)

    with col3:
        event_filter = st.selectbox("Event Type", ["All", "IN", "OUT"])

    # Apply filters
    filtered = df[df['date'] == str(selected_date)]

    if selected_dept != "All":
        filtered = filtered[filtered['department'] == selected_dept]

    if event_filter != "All":
        filtered = filtered[filtered['event'] == event_filter]

    st.markdown("---")

    # ── Today's IN/OUT count per employee ────────────────────────
    st.markdown("### 👥 Employee Visit Summary")

    if len(today_df) > 0:
        summary = today_df.groupby(
            ['employee_id', 'employee_name', 'department', 'event']
        ).size().reset_index(name='count')

        pivot = summary.pivot_table(
            index=['employee_name', 'department'],
            columns='event',
            values='count',
            fill_value=0
        ).reset_index()

        st.dataframe(pivot, use_container_width=True)
    else:
        st.info("No records for today yet.")

    st.markdown("---")

    # ── Full log table ───────────────────────────────────────────
    st.markdown("### 📋 Full Attendance Log")

    if len(filtered) > 0:
        display_df = filtered[[
            'employee_name', 'department', 'event', 'date', 'time'
        ]].sort_values('time', ascending=False)

        # Color event column
        def color_event(val):
            color = 'green' if val == 'IN' else 'red'
            return f'color: {color}; font-weight: bold'

        styled = display_df.style.applymap(color_event, subset=['event'])
        st.dataframe(styled, use_container_width=True)

        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"attendance_{selected_date}.csv",
            mime="text/csv"
        )
    else:
        st.info("No records found for selected filters.")
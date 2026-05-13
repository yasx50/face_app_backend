import streamlit as st

st.set_page_config(
    page_title="FaceTrack Attendance",
    page_icon="🎯",
    layout="wide"
)

st.sidebar.title("🎯 FaceTrack Pro")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📝 Register Employee", "📷 Scanner", "📊 Dashboard"]
)

if page == "🏠 Home":
    st.title("🎯 Welcome to FaceTrack Pro")
    st.markdown("### AI-Powered Face Detection Attendance System")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 📝 Register\nAdd new employees with face capture")
    with col2:
        st.success("### 📷 Scanner\nLive face recognition at entry gate")
    with col3:
        st.warning("### 📊 Dashboard\nView attendance logs and reports")

    st.markdown("---")
    st.markdown("#### How to use:")
    st.markdown("1. Go to **Register Employee** — add employees with face photos")
    st.markdown("2. Go to **Scanner** — start live camera for attendance")
    st.markdown("3. Go to **Dashboard** — view who came in and out")

elif page == "📝 Register Employee":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from pages.register import show
    show()

elif page == "📷 Scanner":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from pages.scanner import show
    show()

elif page == "📊 Dashboard":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from pages.dashboard import show
    show()
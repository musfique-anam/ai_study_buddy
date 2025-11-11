# app.py
import streamlit as st
import cv2
from focus_detector import FocusEstimator
import time
import base64
import streamlit.components.v1 as components
import db
from datetime import datetime

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="AI Study Buddy", layout="wide")
st.title("🧠 AI Study Buddy – Pundra Students")

# --- Initialize Database ---
db.init_db()

# ---------------------------
# Session State (To hold data across reruns)
# ---------------------------
if 'session_running' not in st.session_state:
    st.session_state.session_running = False
if 'estimator' not in st.session_state:
    st.session_state.estimator = FocusEstimator()
if 'pomodoro_running' not in st.session_state:
    st.session_state.pomodoro_running = False
if 'pomodoro_start_time' not in st.session_state:
    st.session_state.pomodoro_start_time = None
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = None
if 'total_alerts' not in st.session_state:
    st.session_state.total_alerts = 0
if 'cap' not in st.session_state:
    st.session_state.cap = None # Camera will be initialized later

# ---------------------------
# Instructions / Features
# ---------------------------
with st.expander("ℹ️ Instructions & Features", expanded=True):
    st.markdown("""
- **Step 1:** Click 'Start Study Session'.
- **Step 2:** Click 'Start Pomodoro' if you want a timed session.
- **Step 3:** Click 'Stop Session' to finish and save your data.
- Alerts for drowsiness, yawning, and poor posture.
- Session data is saved and displayed below.
""")

# ---------------------------
# Placeholders for Streamlit UI
# ---------------------------
frame_placeholder = st.image([])
timing_placeholder = st.empty()
alert_placeholder = st.empty()
pomodoro_placeholder = st.empty()

# ---------------------------
# Function to play sound instantly in browser
# ---------------------------
def play_sound(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    audio_html = f"""
    <audio autoplay="true">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    components.html(audio_html, height=0)

# ---------------------------
# Session and Pomodoro Controls
# ---------------------------
col1, col2, col3 = st.columns(3)
with col1:
    start_session = st.button(
        "▶️ Start Study Session", 
        key="start_session", 
        disabled=st.session_state.session_running
    )
with col2:
    stop_session = st.button(
        "⏹ Stop Session", 
        key="stop_session", 
        disabled=not st.session_state.session_running
    )
with col3:
    start_pomodoro = st.button(
        "⏱ Start Pomodoro (25 min focus)", 
        key="start_pomodoro", 
        disabled=not st.session_state.session_running or st.session_state.pomodoro_running
    )

pomodoro_duration = 25 * 60  # 25 minutes

# ---------------------------
# Alert cooldown (seconds)
# ---------------------------
alert_cooldown = {"blink": 0, "yawn": 0}
COOLDOWN_TIME = 5  # seconds

# ---------------------------
# Session Logic
# ---------------------------
if start_session:
    st.session_state.cap = cv2.VideoCapture(0) # <-- CAMERA IS OPENED HERE
    if not st.session_state.cap.isOpened():
        st.warning("Cannot access webcam. Please check permissions or try another index (e.g., 1 instead of 0).")
        st.session_state.cap = None
    else:
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        st.session_state.session_running = True
        st.session_state.session_start_time = datetime.now()
        # Reset counters
        st.session_state.estimator = FocusEstimator()
        st.session_state.total_alerts = 0
        st.rerun()

if stop_session:
    st.session_state.session_running = False
    st.session_state.pomodoro_running = False
    
    if st.session_state.cap is not None:
        st.session_state.cap.release() # <-- CAMERA IS RELEASED HERE
        st.session_state.cap = None
    
    # --- SAVE SESSION TO DATABASE ---
    session_data = {
        "username": "pundra_student", # Placeholder username
        "start_time": st.session_state.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "focused_seconds": int(st.session_state.estimator.focused_seconds),
        "distracted_seconds": int(st.session_state.estimator.distracted_seconds),
        "drowsy_seconds": int(st.session_state.estimator.drowsy_seconds),
        "alerts": st.session_state.total_alerts
    }
    db.insert_session(session_data)
    
    st.success("Session stopped and saved!")
    # Clear the image placeholder
    frame_placeholder.empty()
    st.rerun()


# ---------------------------
# Main Loop
# ---------------------------
if st.session_state.session_running and st.session_state.cap is not None:
    while True: # Loop until stop is pressed
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.warning("Webcam feed lost.")
            st.session_state.session_running = False
            break

        frame = cv2.flip(frame, 1)
        estimator = st.session_state.estimator
        annotated_frame = estimator.process_frame_light(frame)

        # Display frame
        frame_placeholder.image(
            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

        # Live counters
        total_seconds = estimator.focused_seconds + estimator.drowsy_seconds + estimator.distracted_seconds
        focus_percent = (estimator.focused_seconds / max(1, total_seconds)) * 100
        timing_placeholder.text(
            f"Focused: {estimator.focused_seconds:.1f}s | "
            f"Drowsy (Blinks): {estimator.drowsy_seconds:.1f}s | "
            f"Distracted (Yawn/Posture): {estimator.distracted_seconds:.1f}s | "
            f"Focus %: {focus_percent:.1f}%"
        )

        # Current time for cooldown
        current_time = time.time()
        alert_text = ""

        # Blink alert
        if estimator.blink_alert and current_time - alert_cooldown["blink"] > COOLDOWN_TIME:
            play_sound("alert.wav")
            alert_cooldown["blink"] = current_time
            alert_text += "⚠️ Drowsy! "
            st.session_state.total_alerts += 1

        # Yawn alert
        if estimator.yawn_alert and current_time - alert_cooldown["yawn"] > COOLDOWN_TIME:
            play_sound("alert.wav")
            alert_cooldown["yawn"] = current_time
            alert_text += "⚠️ Yawning! "
            st.session_state.total_alerts += 1

        # Posture alert
        if estimator.posture_alert:
            alert_text += "⚠️ Sit upright! "

        alert_placeholder.text(alert_text)

        # Pomodoro start
        if start_pomodoro and not st.session_state.pomodoro_running:
            st.session_state.pomodoro_start_time = time.time()
            st.session_state.pomodoro_running = True

        # Pomodoro countdown
        if st.session_state.pomodoro_running:
            elapsed = time.time() - st.session_state.pomodoro_start_time
            remaining = max(0, pomodoro_duration - elapsed)
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            pomodoro_placeholder.markdown(f"⏳ Pomodoro Timer: {minutes:02d}:{seconds:02d}")

            if remaining <= 0:
                play_sound("alert.wav") # Play sound on completion
                st.balloons()
                st.success("✅ Pomodoro complete! Take a short break.")
                st.session_state.pomodoro_running = False
                start_pomodoro = False # Reset button state
                st.rerun()

        # ---------------------------
        # Limit FPS to stabilize Streamlit
        # ---------------------------
        time.sleep(0.03)  # ~30 FPS
        
        # Check if stop has been pressed
        if not st.session_state.session_running:
            break
            
# ---------------------------
# Clean up if cap is still open (e.g., on script stop)
# ---------------------------
if st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None

cv2.destroyAllWindows()

# ---------------------------
# Show Past Sessions
# ---------------------------
with st.expander("📊 Past Sessions", expanded=True):
    sessions = db.fetch_all()
    if sessions:
        display_data = []
        for s in sessions:
            display_data.append({
                "ID": s[0],
                "Start Time": s[2],
                "End Time": s[3],
                "Focused (s)": s[4],
                "Distracted (s)": s[5],
                "Drowsy (s)": s[6],
                "Alerts": s[7],
            })
        st.dataframe(display_data, use_container_width=True)
    else:
        st.write("No study sessions recorded yet.")
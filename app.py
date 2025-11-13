import streamlit as st
import cv2
from focus_detector import FocusEstimator 
from pro_detector import ProEstimator 
import time
import base64
import streamlit.components.v1 as components
import db
from datetime import datetime
import pandas as pd

# CUSTOM CSS FUNCTION 
def load_css(file_name):
    with open(file_name) as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Page Setup
st.set_page_config(page_title="AI Study Buddy", layout="wide")
st.title("AI Study Buddy")

load_css("style.css") 

#  Initialize Database 
db.init_db()

# Session State (To hold data across reruns)

if 'session_running' not in st.session_state:
    st.session_state.session_running = False
if 'estimator' not in st.session_state:
    st.session_state.estimator = None
if 'mode' not in st.session_state:
    st.session_state.mode = "Normal" 
if 'pomodoro_running' not in st.session_state:
    st.session_state.pomodoro_running = False
if 'pomodoro_start_time' not in st.session_state:
    st.session_state.pomodoro_start_time = None
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = None
if 'total_alerts' not in st.session_state:
    st.session_state.total_alerts = 0
if 'cap' not in st.session_state:
    st.session_state.cap = None 
if 'last_annotated_frame' not in st.session_state:
    st.session_state.last_annotated_frame = None

# Function to play sound
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


tab1, tab2 = st.tabs(["Live Session", " Dashboard"])

# TAB 1: LIVE SESSION 

with tab1:
    # --- Top Control Panel ---
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            # --- Mode Selector ---
            st.radio(
                "Select Study Mode",
                ("Normal", "Pomodoro", "Pro"),
                key="mode_select",
                horizontal=True,
                disabled=st.session_state.session_running 
            )
            mode = st.session_state.mode_select
        
        with c2:
 
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                start_session = st.button(
                    "▶️ Start Study Session", 
                    key="start_session", 
                    disabled=st.session_state.session_running,
                    use_container_width=True
                )
            with sc2:
                stop_session = st.button(
                    "⏹ Stop Session", 
                    key="stop_session", 
                    disabled=not st.session_state.session_running,
                    use_container_width=True
                )
            
            if mode == "Pomodoro":
                with sc3:
                    start_pomodoro = st.button(
                        "⏱ Start Pomodoro (25 min)", 
                        key="start_pomodoro", 
                        disabled=not st.session_state.session_running or st.session_state.pomodoro_running,
                        use_container_width=True
                    )
            else:
                start_pomodoro = False 

    st.markdown("---") 

    # --- Layout for Video and Live Stats ---
    col1, col2 = st.columns([2, 1]) 

    with col1:
        frame_placeholder = st.image([])

    with col2:
       
        if st.session_state.session_running:
            with st.container(border=True):
                st.markdown("### Live Stats")
                focus_percent_placeholder = st.progress(0, text="Focus %") 

                # Placeholders for Pomodoro and Alerts
                pomodoro_placeholder = st.empty()
                alert_placeholder = st.empty()
                
                st.divider()
                st.markdown("### Live Counters")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    focus_placeholder = st.empty()
                with metric_col2:
                    drowsy_placeholder = st.empty()
                with metric_col3:
                    distract_placeholder = st.empty()
        

    pomodoro_duration = 25 * 60  # 25 minutes
    alert_cooldown = {"blink": 0, "yawn": 0, "emotion": 0}
    COOLDOWN_TIME = 5 

    # --- Session Logic ---
    if start_session:
        st.session_state.cap = cv2.VideoCapture(0)
        if not st.session_state.cap.isOpened():
            st.warning("Cannot access webcam.")
            st.session_state.cap = None
        else:
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            st.session_state.session_running = True
            st.session_state.session_start_time = datetime.now()
            st.session_state.mode = mode 
            
            try:
                if st.session_state.mode == "Pro":
                    with st.spinner("Loading Pro Model (this may take a moment)..."):
                        st.session_state.estimator = ProEstimator()
                else: 
                    with st.spinner("Loading Detector..."):
                        st.session_state.estimator = FocusEstimator()
                
                st.session_state.total_alerts = 0
                st.session_state.last_annotated_frame = None
                st.rerun()
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.error("Did you place 'fer_model.h5' and 'shape_predictor_68_face_landmarks.dat' in the folder?")
                st.session_state.session_running = False
                if st.session_state.cap:
                    st.session_state.cap.release()
                st.session_state.cap = None

    if stop_session:
        st.session_state.session_running = False
        st.session_state.pomodoro_running = False
        
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        
        estimator = st.session_state.estimator
        if estimator is not None:
            session_data = {
                "username": "pundra_student",
                "start_time": st.session_state.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "focused_seconds": int(estimator.focused_seconds),
                "distracted_seconds": int(estimator.distracted_seconds),
                "drowsy_seconds": int(estimator.drowsy_seconds),
                "alerts": st.session_state.total_alerts
            }
            db.insert_session(session_data)
            st.success("Session stopped and saved!")
        
        frame_placeholder.empty()
        st.session_state.last_annotated_frame = None
        st.session_state.estimator = None
        st.rerun()

    # --- Main Loop ---
    frame_counter = 0
    estimator = st.session_state.estimator 

    if st.session_state.session_running and st.session_state.cap is not None and estimator is not None:
        while True: 
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.warning("Webcam feed lost.")
                st.session_state.session_running = False
                break

            frame = cv2.flip(frame, 1)
            
            # Optimization
            frame_counter += 1
            annotated_frame = None
            
            if frame_counter % 6 == 0:
                annotated_frame = estimator.process_frame(frame) 
                st.session_state.last_annotated_frame = annotated_frame
            else:
                if st.session_state.last_annotated_frame is not None:
                    annotated_frame = st.session_state.last_annotated_frame
                else:
                    annotated_frame = frame 
            
            frame_placeholder.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )

            # --- Update Live Dashboard ---
            total_seconds = estimator.focused_seconds + estimator.drowsy_seconds + estimator.distracted_seconds
            focus_percent = (estimator.focused_seconds / max(1, total_seconds))
            focus_percent_text = f"Focus: {focus_percent*100:.1f}%"
            
            focus_percent_placeholder.progress(float(focus_percent), text=focus_percent_text) 
            
            # Show hundredths of a second (like milliseconds)
            focus_placeholder.metric("🎯 Focus", f"{estimator.focused_seconds:.2f}s")
            drowsy_placeholder.metric("😴 Drowsy", f"{estimator.drowsy_seconds:.2f}s")
            distract_placeholder.metric("😵 Distracted", f"{estimator.distracted_seconds:.2f}s")


            current_time = time.time()
            alert_text = ""

            if estimator.yawn_alert and current_time - alert_cooldown["yawn"] > COOLDOWN_TIME:
                play_sound("alert.wav")
                alert_cooldown["yawn"] = current_time
                alert_text += "⚠️ Yawning! "
                st.session_state.total_alerts += 1

            if estimator.blink_alert and current_time - alert_cooldown["blink"] > COOLDOWN_TIME:
                play_sound("alert.wav")
                alert_cooldown["blink"] = current_time
                alert_text += "⚠️ Drowsy! "
                st.session_state.total_alerts += 1

            if estimator.posture_alert:
                alert_text += "⚠️ Sit upright! "

            if st.session_state.mode == "Pro":
                
                if estimator.emotion_alert and current_time - alert_cooldown["emotion"] > current_time:
                    alert_cooldown["emotion"] = current_time
                    alert_text += f"⚠️ Distracted ({estimator.current_emotion})! "
                    st.session_state.total_alerts += 1

            if alert_text:
                alert_placeholder.error(alert_text)
            else:
                alert_placeholder.empty()

            if start_pomodoro and not st.session_state.pomodoro_running:
                st.session_state.pomodoro_start_time = time.time()
                st.session_state.pomodoro_running = True
                start_pomodoro = False 
                st.rerun()

            if st.session_state.mode == "Pomodoro" and st.session_state.pomodoro_running:
                elapsed = time.time() - st.session_state.pomodoro_start_time
                remaining = max(0, pomodoro_duration - elapsed)
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                
                pomodoro_placeholder.markdown(f"### ⏳ Pomodoro: {minutes:02d}:{seconds:02d}")

                if remaining <= 0:
                    play_sound("alert.wav") 
                    st.balloons()
                    st.success("✅ Pomodoro complete! Take a short break.")
                    st.session_state.pomodoro_running = False
                    st.rerun()
            
            if not st.session_state.session_running:
                break
                
    # --- End of While Loop ---

# --- Cleanup ---
if st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None
cv2.destroyAllWindows()

# --- TAB 2: DASHBOARD ---

with tab2:
    st.header("Your Study Dashboard")
    st.write("Review your past performance and see your focus trends over time.")

    sessions = db.fetch_all()
    
    if not sessions:
        st.info("You don't have any saved study sessions yet. Complete a session in the 'Live Session' tab to see your stats here!")
    else:
        # --- Convert DB data to Pandas DataFrame ---
        df = pd.DataFrame(
            sessions,
            columns=[
                'id', 'username', 'start_time', 'end_time', 
                'focused_seconds', 'distracted_seconds', 'drowsy_seconds', 'alerts'
            ]
        )
        # Convert times to datetime objects for plotting
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        # --- Create new metrics for the dashboard ---
        df['total_seconds'] = df['focused_seconds'] + df['distracted_seconds'] + df['drowsy_seconds']
        df['total_minutes'] = (df['total_seconds'] / 60).round(1)
        df['focus_percent'] = (df['focused_seconds'] / df['total_seconds'] * 100).round(1)
        
      
        df_chart = df.set_index('start_time')

        # --- Show Key Metrics ---
        st.subheader("All-Time Stats")
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Sessions", len(df))
            col2.metric("Total Study Time", f"{df['total_minutes'].sum():.1f} min")
            col3.metric("Avg. Focus %", f"{df['focus_percent'].mean():.1f}%")

        st.divider()

        # --- Show Charts ---
        st.subheader("Focus % Over Time")
        st.line_chart(df_chart['focus_percent'])

        st.subheader("Alerts Per Session")
        st.bar_chart(df_chart['alerts'])

        with st.expander("Show Raw Session Data"):
           
            df_display = df[[
                'start_time', 'end_time', 'total_minutes', 
                'focus_percent', 'alerts'
            ]].copy()
            
            df_display.columns = [
                'Start Time', 'End Time', 'Duration (min)', 
                'Focus %', 'Alerts'
            ]
            
            df_display.index = pd.RangeIndex(start=1, stop=len(df_display) + 1)
            df_display.index.name = "Session"
            
            # 4. Display the clean DataFrame
            st.dataframe(df_display, use_container_width=True)
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Sign Language System", layout="wide")

# ================= STATE =================
if "run" not in st.session_state:
    st.session_state.run = False

if "sentence" not in st.session_state:
    st.session_state.sentence = []

# ================= UI =================
st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'> Sign Language Recognition</h1>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Controls")

mode = st.sidebar.selectbox("Mode", ["WORD", "ALPHABET"])

col1, col2 = st.sidebar.columns(2)
if col1.button("▶ Start"):
    st.session_state.run = True

if col2.button("⏹ Stop"):
    st.session_state.run = False

if st.sidebar.button("🧹 Clear Sentence"):
    st.session_state.sentence = []

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    rf = joblib.load("gesture_model.pkl")
    lstm = load_model("model.h5", compile=False)
    return rf, lstm

rf_model, lstm_model = load_models()

alphabet_actions = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ================= LAYOUT =================
colA, colB = st.columns([2, 1])
frame_placeholder = colA.empty()
info_placeholder = colB.empty()

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if st.session_state.run:

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        prev_time = time.time()
        sequence = []

        # ===== stability variables =====
        last_char = ""
        char_start_time = 0
        CHAR_DELAY = 1.0

        while st.session_state.run:

            ret, frame = cap.read()
            if not ret:
                st.error("Camera error")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture_text = "..."
            confidence = 0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # ===== KEYPOINTS =====
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.append([lm.x, lm.y, lm.z])

                keypoints = np.array(keypoints)
                keypoints = keypoints - keypoints[0]
                keypoints = keypoints.flatten()

                # ================= WORD MODE =================
                if mode == "WORD":
                    probs = rf_model.predict_proba(keypoints.reshape(1, -1))[0]
                    confidence = max(probs)

                    if confidence > 0.6:
                        gesture_text = rf_model.classes_[np.argmax(probs)]

                        if len(st.session_state.sentence) == 0 or \
                           st.session_state.sentence[-1] != gesture_text:
                            st.session_state.sentence.append(gesture_text)

                # ================= ALPHABET MODE =================
                else:
                    sequence.append(keypoints)
                    sequence = sequence[-15:]

                    if len(sequence) == 15:
                        res = lstm_model.predict(np.expand_dims(sequence, axis=0))[0]
                        confidence = max(res)

                        if confidence > 0.8:
                            predicted_char = alphabet_actions[np.argmax(res)]

                            # ===== stability logic =====
                            if predicted_char == last_char:
                                if time.time() - char_start_time > CHAR_DELAY:
                                    st.session_state.sentence.append(predicted_char)
                                    last_char = ""
                            else:
                                last_char = predicted_char
                                char_start_time = time.time()

                            gesture_text = predicted_char
                        else:
                            gesture_text = "..."

            # ===== FPS =====
            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time)) if prev_time else 0
            prev_time = curr_time

            # ===== DISPLAY =====
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)

            info_placeholder.markdown(f"""
            ### 🧠 Prediction
            **Gesture:** `{gesture_text}`  
            **Confidence:** `{confidence:.2f}`  
            **FPS:** `{fps}`  

            ---
            ### 📝 Sentence
            **{" ".join(st.session_state.sentence)}**
            """)

cap.release()
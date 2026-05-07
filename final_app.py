import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from tensorflow.keras.models import load_model

print("🚀 Starting System...")

# -------------------- LOAD MODELS --------------------
rf_model = joblib.load("gesture_model.pkl")
lstm_model = load_model("model.h5", compile=False)

print("✅ Models Loaded")

# -------------------- CONFIG --------------------
word_actions = ["YES", "NO", "OK", "STOP", "PEACE", "ROCK"]
alphabet_actions = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

mode = "WORD"

sentence = []
last_word = ""

current_gesture = ""
gesture_start_time = 0
GESTURE_DELAY = 1.0

sequence = []

# FPS
prev_time = 0

# -------------------- MEDIAPIPE --------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera Error")
    exit()

print("✅ Camera Started")

# -------------------- MAIN LOOP --------------------
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame error")
            continue

        frame = cv2.flip(frame, 1)

        key = cv2.waitKey(1) & 0xFF

        # -------- MODE SWITCH --------
        if key == ord('1'):
            mode = "WORD"
            sequence = []
            sentence = []
            last_word = ""

        elif key == ord('2'):
            mode = "ALPHABET"
            sequence = []
            sentence = []
            last_word = ""

        elif key == ord('c'):
            sentence = []

        # -------- MEDIAPIPE --------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture_text = ""
        confidence = 0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])

            keypoints = np.array(keypoints)

            # 🔥 NORMALIZATION
            keypoints = keypoints - keypoints[0]
            keypoints = keypoints.flatten()

            # ================= WORD MODE =================
            if mode == "WORD":
                probs = rf_model.predict_proba(keypoints.reshape(1, -1))[0]
                confidence = max(probs)

                if confidence > 0.9:
                    predicted = word_actions[np.argmax(probs)]

                    if predicted == current_gesture:
                        if time.time() - gesture_start_time > GESTURE_DELAY:

                            if predicted != last_word:
                                sentence.append(predicted)
                                last_word = predicted

                            current_gesture = ""
                    else:
                        current_gesture = predicted
                        gesture_start_time = time.time()
                else:
                    current_gesture = ""

                gesture_text = predicted if confidence > 0.9 else ""

            # ================= ALPHABET MODE =================
            elif mode == "ALPHABET":
                sequence.append(keypoints)
                sequence = sequence[-15:]

                if len(sequence) == 15:
                    res = lstm_model.predict(np.expand_dims(sequence, axis=0))[0]
                    confidence = max(res)

                    if confidence > 0.8:
                        predicted = alphabet_actions[np.argmax(res)]

                        if predicted == current_gesture:
                            if time.time() - gesture_start_time > GESTURE_DELAY:

                                if predicted != last_word:
                                    sentence.append(predicted)
                                    last_word = predicted

                                current_gesture = ""
                        else:
                            current_gesture = predicted
                            gesture_start_time = time.time()
                    else:
                        current_gesture = ""

                    gesture_text = predicted if confidence > 0.8 else ""

        # -------------------- FPS --------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # -------------------- UI PANEL --------------------
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 120), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Mode
        cv2.putText(frame, f"Mode: {mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Gesture
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Confidence
        cv2.putText(frame, f"Conf: {confidence:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (300, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Sentence box
        cv2.rectangle(frame, (0, 400), (640, 480), (50, 50, 50), -1)

        cv2.putText(frame, "Sentence:", (10, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, " ".join(sentence[-6:]), (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # Instructions
        cv2.putText(frame, "1:WORD  2:ALPHABET  C:Clear  Q:Quit",
                    (10, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # -------------------- SHOW --------------------
        cv2.imshow("🖐️ Sign Language System", frame)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
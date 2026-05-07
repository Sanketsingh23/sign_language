import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from tensorflow.keras.models import load_model

print("STARTING SCRIPT...")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

print("Loading RF model...")
rf_model = joblib.load("gesture_model.pkl")
print("RF model loaded")

print("Loading LSTM model...")
lstm_model = load_model("model.h5", compile=False)
print("LSTM model loaded")


#word_actions = ["YES", "NO", "OK", "HELLO", "STOP", "PEACE", "CALL_ME", "ROCK"]
word_actions =["YES", "NO", "OK", "HELLO", "STOP"]
alphabet_actions = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

sentence = []
last_word = ""

current_gesture = ""
gesture_start_time = 0
GESTURE_DELAY = 1.0

sequence = []
mode = "WORD"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera not opening")
    exit()
else:
    print("Camera opened successfully")

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            mode = "WORD"
            sequence = []

        elif key == ord('2'):
            mode = "ALPHABET"
            sequence = []

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

            # 🔥 SAME NORMALIZATION (CRITICAL)
            keypoints = keypoints - keypoints[0]
            keypoints = keypoints.flatten()

            if mode == "WORD":
                probs = rf_model.predict_proba(keypoints.reshape(1, -1))[0]
                confidence = max(probs)

                if confidence > 0.9:
                    gesture_text = word_actions[np.argmax(probs)]
                else:
                    gesture_text = ""

                if gesture_text != "":
                    if gesture_text == current_gesture:
                        if time.time() - gesture_start_time > GESTURE_DELAY:

                            if gesture_text != last_word:
                                sentence.append(gesture_text)
                                last_word = gesture_text

                            current_gesture = ""
                    else:
                        current_gesture = gesture_text
                        gesture_start_time = time.time()
                else:
                    current_gesture = ""

            elif mode == "ALPHABET":
                sequence.append(keypoints)
                sequence = sequence[-15:]

                if len(sequence) == 15:
                    res = lstm_model.predict(np.expand_dims(sequence, axis=0))[0]
                    confidence = max(res)
                    gesture_text = alphabet_actions[np.argmax(res)]

        cv2.putText(frame, f"Mode: {mode}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, gesture_text, (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.putText(frame, f"Conf: {confidence:.2f}", (50, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.putText(frame, " ".join(sentence), (50, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Gesture System", frame)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
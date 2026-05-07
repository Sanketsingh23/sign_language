import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands   

DATA_PATH = "dataset"
#actions = ["YES", "NO", "OK", "HELLO", "STOP", "PEACE", "CALL_ME", "ROCK"]
actions = ["OK"]

no_sequences = 30

# 🔥 Ensure base dataset folder exists
os.makedirs(DATA_PATH, exist_ok=True)   

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

current_action = 0
sequence = 0

with mp_hands.Hands(max_num_hands=1) as hands:

    while current_action < len(actions):

        action = actions[current_action]

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, f"Action: {action} | Sample: {sequence}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, "Press S to capture | N next | Q quit", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.append([lm.x, lm.y, lm.z])

                keypoints = np.array(keypoints)

                # 🔥 NORMALIZATION (CRITICAL FIX)
                keypoints = keypoints - keypoints[0]

                keypoints = keypoints.flatten()

                os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

                np.save(os.path.join(DATA_PATH, action, str(sequence)), keypoints)

                print(f"Saved {action} sample {sequence}")
                sequence += 1

            else:
                print("No hand detected")

        elif key == ord('n'):
            current_action += 1
            sequence = 0

        elif key == ord('q'):
            break

        if sequence >= no_sequences:
            current_action += 1
            sequence = 0

cap.release()
cv2.destroyAllWindows()
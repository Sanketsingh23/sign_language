import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def detect_gesture(landmarks):
    tips = [8, 12, 16, 20]
    fingers = []

    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    total = sum(fingers)

    # 👍 YES (STRICT + FIRST)
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    thumb_up = thumb_tip.y < thumb_ip.y - 0.02  # margin added

    index_down = landmarks[8].y > landmarks[6].y
    middle_down = landmarks[12].y > landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y

    if thumb_up and index_down and middle_down and ring_down and pinky_down:
        return "YES"

    # ✋ HELLO
    if total == 4:
        return "HELLO"

    # ✌️ CLEAR
    if total == 2:
        return "CLEAR"

    # ✊ NO (after YES)
    if total == 0:
        return "NO"

    # 👌 OK
    index_tip = landmarks[8]
    thumb_tip = landmarks[4]

    distance = math.sqrt(
        (index_tip.x - thumb_tip.x) ** 2 +
        (index_tip.y - thumb_tip.y) ** 2
    )

    if distance < 0.05:
        return "OK"

    return ""


def format_sentence(words):
    if not words:
        return ""

    text = " ".join(words)
    text = text.capitalize()

    if text.startswith("Hello"):
        return text + "!"

    return text + "."


# 🔹 Sentence storage
sentence = []
last_word = ""


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera not opening")
    exit()

print("Camera started. Press 'q' to exit.")

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Frame not received")
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture_text = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                gesture_text = detect_gesture(hand_landmarks.landmark)

                # 🔥 Sentence logic
                if gesture_text == "CLEAR":
                    sentence = []
                    last_word = ""
                elif gesture_text != "" and gesture_text != last_word:
                    sentence.append(gesture_text)
                    last_word = gesture_text

                print("Sentence:", sentence)

        else:
            print("No hand")

        # 🔹 Show current gesture
        cv2.putText(
            frame,
            gesture_text,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            3
        )

        # 🔹 Show formatted sentence
        cv2.putText(
            frame,
            format_sentence(sentence),
            (50, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.imshow("Hand Gesture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break


cap.release()
cv2.destroyAllWindows()
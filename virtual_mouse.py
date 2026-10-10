import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

# Camera
cap = cv2.VideoCapture(0)

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# Screen size
screen_w, screen_h = pyautogui.size()

# Cursor smoothing
prev_x = 0
prev_y = 0
smooth = 0.3

# Click states
left_click = False
right_click = False

# Scroll
prev_scroll_y = None
scroll_threshold = 20

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    h, w, c = frame.shape

    # BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        # Landmarks
        thumb = hand.landmark[4]
        index = hand.landmark[8]
        middle = hand.landmark[12]

        # Pixel coordinates
        thumb_x = int(thumb.x * w)
        thumb_y = int(thumb.y * h)

        index_x = int(index.x * w)
        index_y = int(index.y * h)

        middle_x = int(middle.x * w)
        middle_y = int(middle.y * h)

        # Finger states
        index_up = (
            hand.landmark[8].y <
            hand.landmark[6].y
        )

        middle_up = (
            hand.landmark[12].y <
            hand.landmark[10].y
        )

        ring_up = (
            hand.landmark[16].y <
            hand.landmark[14].y
        )

        pinky_up = (
            hand.landmark[20].y <
            hand.landmark[18].y
        )

        # Distances
        thumb_index = math.hypot(
            index_x - thumb_x,
            index_y - thumb_y
        )

        thumb_middle = math.hypot(
            middle_x - thumb_x,
            middle_y - thumb_y
        )

        # All fingers closed
        all_closed = (
            not index_up and
            not middle_up and
            not ring_up and
            not pinky_up
        )

        # -----------------------------
        # SCROLL
        # -----------------------------
        if all_closed:

            if prev_scroll_y is not None:

                movement = prev_scroll_y - index_y

                if movement > scroll_threshold:
                    pyautogui.scroll(2)

                elif movement < -scroll_threshold:
                    pyautogui.scroll(-2)

            prev_scroll_y = index_y

            left_click = False
            right_click = False

        # -----------------------------
        # LEFT CLICK
        # -----------------------------
        elif index_up and not middle_up and thumb_index < 45:

            if not left_click:
                pyautogui.click()
                left_click = True

            right_click = False
            prev_scroll_y = None

        # -----------------------------
        # RIGHT CLICK
        # -----------------------------
        elif middle_up and not index_up and thumb_middle < 45:

            if not right_click:
                pyautogui.rightClick()
                right_click = True

            left_click = False
            prev_scroll_y = None

        # -----------------------------
        # MOVE MOUSE
        # -----------------------------
        elif index_up:

            # Camera to screen
            mouse_x = np.interp(
                index_x,
                [0, w],
                [0, screen_w]
            )

            mouse_y = np.interp(
                index_y,
                [0, h],
                [0, screen_h]
            )

            # Smooth movement
            smooth_x = (
                prev_x +
                (mouse_x - prev_x) * smooth
            )

            smooth_y = (
                prev_y +
                (mouse_y - prev_y) * smooth
            )

            pyautogui.moveTo(
                smooth_x,
                smooth_y
            )

            prev_x = smooth_x
            prev_y = smooth_y

            left_click = False
            right_click = False
            prev_scroll_y = None

        else:

            left_click = False
            right_click = False
            prev_scroll_y = None

        # Draw hand
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Index finger point
        cv2.circle(
            frame,
            (index_x, index_y),
            8,
            (0, 0, 255),
            -1
        )

    # Project name
    cv2.putText(
        frame,
        "AI VIRTUAL MOUSE",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # Show camera
    cv2.imshow(
        "AI Virtual Mouse",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close everything
cap.release()
cv2.destroyAllWindows()
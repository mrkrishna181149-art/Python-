import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Get screen resolution
width, height = pyautogui.size()
print(width, height)
pyautogui.FAILSAFE = False
# Get current cursor position
current_x, current_y = pyautogui.position()
# print(current_x, current_y)

# pyautogui.moveTo(580, 780, duration=1)

# Ai Model Initilize
mp_hand = mp.solutions.hands

hand = mp_hand.Hands(
        max_num_hands = 1,
        min_detection_confidence=0.3,
        min_tracking_confidence = 0.3
)

# Caputure Video
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read() # Reading Frames

    if not success:
        break

    # Coverting BGR Image into RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hand.process(rgb)

    if results.multi_hand_landmarks:
        
        for handlmks in results.multi_hand_landmarks:
            # print(handlmks)

            h, w, _ = frame.shape

            lmkList = []
            for inde, lmk in enumerate(handlmks.landmark):
                # print(lmk.x, lmk.y)
                cx = int((lmk.x*w))
                cy = int((lmk.y *h ))

                lmkList.append((cx, cy))
            # print(lmkList)

            indexFing = lmkList[8]

            x,y = indexFing
            screen_x = np.interp(x, (0, w), (0, width))

            screen_y = np.interp(y, (0, h), (0, height))

            print(screen_x, screen_y)

            

            pyautogui.moveTo(screen_x,screen_y)

            cv2.circle(frame, indexFing,10, [0,0,255], -5)

    cv2.imshow("Mouse Controller",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
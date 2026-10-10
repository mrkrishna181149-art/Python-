# Import all required libs
import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
print(mp.__file__)

# Initialize Audio 


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None) 

volume = cast(interface, POINTER(IAudioEndpointVolume))
volmin, volmax = volume.GetVolumeRange()[:2]


# Initilize Mediapipe AI Hand Detection Model

mp_hand = mp.solutions.hands

hand = mp_hand.Hands(
        max_num_hands = 2,
        min_detection_confidence=0.5,
        min_tracking_confidence = 0.5
        )


# Capture Video
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)


    if not success:
        break

    # Converting BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processing Hand
    results = hand.process(rgb)

    if results.multi_hand_landmarks:
        #print(results.multi_hand_landmarks)

        for handlmks in results.multi_hand_landmarks:
            # print(handlmks)

            h, w, c = rgb.shape

        lmsList = []
        for ind, lm in enumerate(handlmks.landmark):
            cx = int(lm.x * w)
            cy = int(lm.y * h)

            lmsList.append((cx, cy))

        # print(lmsList)

        if len(lmsList) != 0:
            thumb = lmsList[4]
            indexFing = lmsList[8]
            
            x1,y1 = thumb
            x2,y2 = indexFing

            # print(f"(x1 : {x1}, x2 : {x2}), (y1 : {y1}, y2:{y2})")

            length = math.hypot(x2 - x1, y2 - y1)
            print("length", int(length))

            volumeRange = np.interp(length,[20, 200], [0, 100])
            print("volumeRange " +  str(volumeRange) + "%") 
            vol = np.interp(length,[2, 200], [volmin, volmax])
            volume.SetMasterVolumeLevel(vol, None)
            cv2.putText(frame, f"volume:{int(volumeRange)}%", (20,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,255,0), 2)



            #print("start points", thumb)
            # print("end points",indexFing)

            cv2.circle(frame, thumb,10,[255,0,255], -5)
            cv2.circle(frame, indexFing,10,[255,0,255], -5)

            cv2.line(frame, thumb, indexFing, [0, 255, 0], 5)
    cv2.imshow("Volume Controller", frame)

    if cv2.waitKey(1) & 0xFF== ord('p'):
        break

cap.release()
cv2.destroyAllWindows()





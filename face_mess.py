import cv2
import mediapipe as mp


# Model Configuration
mp_face_mess = mp.solutions.face_mesh

face_mess = mp_face_mess.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Capture Video
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mess.process(rgb)
    
    if results.multi_face_landmarks:

        for facelmk in results.multi_face_landmarks:
            print(facelmk.landmark)

            h, w, _ = frame.shape

            allFaceMess = []
            for flmk in facelmk.landmark:
                cx = int(flmk.x*w)
                cy = int(flmk.y*h)

                allFaceMess.append((cx, cy))
                cv2.circle(frame, (cx,cy), 2, [0,0,255], -2)
    cv2.imshow("Face Mess Window", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
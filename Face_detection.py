import cv2 
import mediapipe as mp

# Initializing Mediapipe Model
mp_face = mp.solutions.face_detection

face = mp_face.FaceDetection(
    model_selection=0, 
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0) # default value 0


while True:
    success, frame = cap.read()

    if not success:
        break

    # Flip Layout
    # frame = cv2.flip(frame, 1)
    # Converting BRG to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h,w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    cv2.imshow("Camera window", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
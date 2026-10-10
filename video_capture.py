import cv2

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break
    cv2.imshow("Video",frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
       break

cap.release()

cv2.destroyAllWindows()
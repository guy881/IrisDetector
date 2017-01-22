import cv2
import time

# face_cascade = cv2.CascadeClassifier('haar\\haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haar\\haarcascade_eye.xml')
eye_cascade = cv2.CascadeClassifier('haar\\haarcascade_eye_tree_eyeglasses.xml')
# eye_cascade = cv2.CascadeClassifier('haar\\haarcascade_lefteye_2splits.xml')

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (ex, ey, ew, eh) in eyes:
        # cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh)imp, (0, 255, 0), 2)
        # cv2.rectangle(frame, (ex, ey), (ex + 2, ey + 2), (0, 255, 0), 2)
        cv2.circle(frame, (int(ex + float(ew/2)), int(ey + float(eh/2))), 2, (0, 255, 0), 2)
    cv2.imshow('img', frame)
    time.sleep(0.06)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

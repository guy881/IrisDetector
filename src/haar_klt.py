import cv2
import numpy


def current_frame(capture, gray=False):
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    if not gray:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


eye_cascade = cv2.CascadeClassifier('haar\\haarcascade_eye_tree_eyeglasses.xml')
capture = cv2.VideoCapture(0)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

old_frame = current_frame(capture, gray=True)
eyes = []

while len(eyes) < 2:
    old_frame = current_frame(capture)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(old_gray, 1.3, 5)  # if not empty - eyes detected
    cv2.imshow('Trying to localize your eyes..', old_frame)
    print "waiting"
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
irises = []
for (ex, ey, ew, eh) in eyes:
    iris_w = int(ex + float(ew / 2))
    iris_h = int(ey + float(eh / 2))
    irises.append([numpy.float32(iris_w), numpy.float32(iris_h)])

irises = numpy.array(irises)

while True:
    frame = current_frame(capture)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, irises, None, **lk_params)

    print str(st) + " " + str(err)
    irises = []
    for w, h in p1:
        cv2.circle(frame, (w, h), 2, (0, 255, 0), 2)
        irises.append([w, h])
    irises = numpy.array(irises)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    old_gray = gray.copy()

cv2.destroyAllWindows()
capture.release()

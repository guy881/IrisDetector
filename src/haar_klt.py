import cv2
import numpy

eye_cascade = cv2.CascadeClassifier('haar\\haarcascade_eye_tree_eyeglasses.xml')
capture = cv2.VideoCapture(0)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
blinks = 0


def current_frame(capture, gray=False):
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    if not gray:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def get_irises_location():
    eyes = []
    while len(eyes) < 2:
        old_gray = current_frame(capture, gray=True)
        eyes = eye_cascade.detectMultiScale(old_gray, 1.3, 5)  # if not empty - eyes detected

    irises = []
    for (ex, ey, ew, eh) in eyes:
        iris_w = int(ex + float(ew / 2))
        iris_h = int(ey + float(eh / 2))
        irises.append([numpy.float32(iris_w), numpy.float32(iris_h)])

    return numpy.array(irises)

old_frame = current_frame(capture)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
cv2.imshow('Trying to localize your eyes..', old_frame)
k = cv2.waitKey(1) & 0xff
irises = get_irises_location()
cv2.destroyAllWindows()

while True:
    frame = current_frame(capture)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, irises, None, **lk_params)

    error_threshold = 9
    if st[0][0] == 0 or st[1][0] == 0:  # lost track on eyes
        irises = get_irises_location()
        blink_in_previous = False
    elif err[0][0] > error_threshold or err[1][0] > error_threshold:  # high error rate in klt tracking
        irises = get_irises_location()
        if not blink_in_previous:
            blinks += 1
            blink_in_previous = True
    else:
        blink_in_previous = False
        irises = []
        for w, h in p1:
            cv2.circle(frame, (w, h), 2, (0, 255, 0), 2)
            irises.append([w, h])
        irises = numpy.array(irises)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(err), (20, 450), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'blinks: ' + str(blinks), (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESC
        break

    old_gray = gray.copy()

cv2.destroyAllWindows()
capture.release()

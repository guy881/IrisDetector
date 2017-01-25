import cv2
import numpy

eye_cascade = cv2.CascadeClassifier('haar\\haarcascade_eye_tree_eyeglasses.xml')
capture = cv2.VideoCapture(0)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def current_frame(capture, gray=False):
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    if not gray:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def get_irises_location(frame_gray):
    eyes = eye_cascade.detectMultiScale(frame_gray, 1.3, 5)  # if not empty - eyes detected
    irises = []

    for (ex, ey, ew, eh) in eyes:
        iris_w = int(ex + float(ew / 2))
        iris_h = int(ey + float(eh / 2))
        irises.append([numpy.float32(iris_w), numpy.float32(iris_h)])

    return numpy.array(irises)


def show_image_with_data(frame, blinks, irises, err=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if err:
        cv2.putText(frame, str(err), (20, 450), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'blinks: ' + str(blinks), (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    for w, h in irises:
        cv2.circle(frame, (w, h), 2, (0, 255, 0), 2)
    cv2.imshow('Eyeris detector', frame)


def run_iris_detector():
    irises = []
    blink_in_previous = False
    blinks = 0
    k = cv2.waitKey(30) & 0xff

    while k != 27:  # ESC
        frame = current_frame(capture)
        error_threshold = 9
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        if len(irises) >= 2:  # irises detected, track eyes
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, irises, None, **lk_params)
            if st[0][0] == 0 or st[1][0] == 0:  # lost track on eyes
                irises = get_irises_location(gray)
                blink_in_previous = False
            elif err[0][0] > error_threshold or err[1][0] > error_threshold:  # high error rate in klt tracking
                irises = get_irises_location(gray)
                if not blink_in_previous:
                    blinks += 1
                    blink_in_previous = True
            else:
                blink_in_previous = False
                irises = []
                for w, h in p1:
                    irises.append([w, h])
                irises = numpy.array(irises)
        else:  # cannot track for some reason -> find irises
            irises = get_irises_location(gray)

        show_image_with_data(frame, blinks, irises)
        k = cv2.waitKey(30) & 0xff
        old_gray = gray.copy()

    cv2.destroyAllWindows()
    capture.release()

run_iris_detector()


# import the necessary packages
from helpers import pyramid
from helpers import sliding_window
import time
import cv2

eye = cv2.imread('..\\bazy\\Webcam\\eyes\\3.jpg', 0)
(winW, winH) = (103, 31)
orb = cv2.ORB_create()
kp_eye, eye_descriptor = orb.detectAndCompute(eye, None)  # find key points and computer descriptor

face = cv2.imread('..\\bazy\\Webcam\\6.jpg', 0)
matches = []
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# loop over the image pyramid
for resized in pyramid(face, scale=1.5):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=15, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # clone = resized.copy()

        kp_window, windows_descriptor = orb.detectAndCompute(window, None)  #
        match = matcher.match(eye_descriptor, windows_descriptor)
        # matches.append(match)
        # match = sorted(match, key=lambda x: x.distance)

        print match
        #
        # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        # cv2.imshow("Window", window)
        # cv2.waitKey(1)
        # time.sleep(0.025)


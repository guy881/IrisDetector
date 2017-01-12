import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../bazy/BioID-FaceDatabase-V1.2/lenna.png', 0)
# img = cv2.imread('../bazy/BioID-FaceDatabase-V1.2/BioID_0292.pgm',0)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector(45)

# find and draw the keypoints
kp = fast.detect(img, None)
img2 = cv2.drawKeypoints(img, kp, color=(255, 0, 0))

# Print all default params
# print "Threshold: ", fast.getInt('threshold')
# print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
# print "neighborhood: ", fast.getInt('type')
# print "Total Keypoints with nonmaxSuppression: ", len(kp)

cv2.imwrite('fast_true.png', img2)

# # Disable nonmaxSuppression
# fast.setBool('nonmaxSuppression',0)
# kp = fast.detect(img,None)
#
# print "Total Keypoints without nonmaxSuppression: ", len(kp)
#
# img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))
#
# cv2.imwrite('fast_false.png',img3)

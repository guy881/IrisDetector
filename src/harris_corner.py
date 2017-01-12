#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np

# filename = 'G:\\zrenice\\bazy\\Webcam\\eyes\\1.jpg'
# filename = 'G:\\zrenice\\bazy\\Webcam\\eyes\\2.jpg'
filename = 'G:\\zrenice\\bazy\\Webcam\\eyes\\3.jpg'
# filename = 'G:\\zrenice\\bazy\\Webcam\\eyes\\4.jpg'
# filename = 'G:\\zrenice\\bazy\\Webcam\\eyes\\5.jpg'
# filename = 'G:\\zrenice\\bazy\\Webcam\\eyes\\6.jpg'
# img = cv2.imread('../bazy/BioID-FaceDatabase-V1.2/BioID_0293.pgm')
# img = cv2.imread('../bazy/BioID-FaceDatabase-V1.2/lenna2.jpg')
# img = cv2.imread('../bazy/eye.jpeg')
img = cv2.imread(filename)
# img = cv2.imread("G:\\zrenice\\bazy\\Webcam\\2016-12-17-175909.jpg", 0)
# img = cv2.imread('/home/stevens/Obrazy/Webcam/2016-10-27-220504.jpg')
# img = cv2.imread('/home/stevens/Obrazy/Webcam/2016-10-27-220018.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,11,0.1)

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

without_extension = filename[:-4]
extension = filename[-4:]
cv2.imshow('dst',img)
cv2.imwrite(without_extension + 'harris' + extension, img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
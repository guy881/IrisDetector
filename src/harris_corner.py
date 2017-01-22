#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np

# filename = '..\\bazy\\Webcam\\eyes\\1.jpg'
# filename = '..\\bazy\\Webcam\\eyes\\2.jpg'
filename = '..\\bazy\\Webcam\\eyes\\3.jpg'
# filename = '..\\bazy\\Webcam\\eyes\\4.jpg'
# filename = '..\\bazy\\Webcam\\eyes\\5.jpg'
# filename = '..\\bazy\\Webcam\\eyes\\6.jpg'
# img = cv2.imread('../bazy/BioID-FaceDatabase-V1.2/BioID_0293.pgm')
# img = cv2.imread('../bazy/BioID-FaceDatabase-V1.2/lenna2.jpg')
# img = cv2.imread('../bazy/eye.jpeg')
img = cv2.imread(filename)
# img = cv2.imread("..\\bazy\\Webcam\\2016-12-17-175909.jpg", 0)
# img = cv2.imread('/home/stevens/Obrazy/Webcam/2016-10-27-220504.jpg')
# img = cv2.imread('/home/stevens/Obrazy/Webcam/2016-10-27-220018.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,11,0.1)

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# create keypoints from dst
keypoints = np.argwhere(dst > 0.01 * dst.max())
keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints]

# now I use ORB to create descriptor from harris keypoints
orb = cv2.ORB_create()
addas, descriptor = orb.compute(img, keypoints)

img2 = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)
cv2.imshow("harris", img2)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


def get_harris_orb_descriptor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 11, 0.1)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # create keypoints from dst
    keypoints = np.argwhere(dst > 0.01 * dst.max())
    keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints]

    # now I use ORB to create descriptor from harris keypoints
    orb = cv2.ORB_create()
    addas, descriptor = orb.compute(img, keypoints)

    return descriptor


def save_harris_on_disk(img, filename):
    without_extension = filename[:-4]  # save on the disk
    extension = filename[-4:]

    cv2.imshow('dst', img)
    cv2.imwrite(without_extension + 'harris' + extension, img)



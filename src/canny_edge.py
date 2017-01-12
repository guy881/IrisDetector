#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread('../bazy/BioID-FaceDatabase-V1.2/BioID_0293.pgm', 0)
img = cv2.imread('../bazy/BioID-FaceDatabase-V1.2/lenna2.jpg', 0)
# img = cv2.imread('/home/stevens/Obrazy/Webcam/2016-10-27-220504.jpg', 0)
# img = cv2.imread('/home/stevens/Obrazy/Webcam/2016-10-27-220018.jpg', 0)
edges = cv2.Canny(img, 80, 120)

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detector'), plt.xticks([]), plt.yticks([])

plt.show()

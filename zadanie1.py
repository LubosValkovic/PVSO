import sys

import cv2
from ximea import xiapi

img = cv2.imread(cv2.samples.findFile(""))

if img is None:
    sys.exit("Could not read the image.")
cv2.imshow("Display window", img)
k = cv2.waitKey(0)

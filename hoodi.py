import math
import cv2
import time
import handTrackingModule as htm
import numpy as np

cap = cv2.VideoCapture(0)
detector = htm.handTracker(detectionConf=0.7)


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmarksList = detector.findPosition(img, draw=False)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
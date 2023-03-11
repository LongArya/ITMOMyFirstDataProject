import cv2
import numpy as np


def demo_web_cam():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        cv2.imshow("frame", frame)
        cv2.waitKey(1)


demo_web_cam()

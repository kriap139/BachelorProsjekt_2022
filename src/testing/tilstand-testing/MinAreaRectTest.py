import cv2 as cv2
import numpy as np
from numbers import Number
from src.Backend.StateDetection.Methods.constants import RAD_TO_DEG, DEG_TO_RAD
from operator import mul


window_name = 'MinaAreaRect Test'
trackbar_name = "Trackbar Rot Deg"

img = np.zeros((600, 600, 3), dtype=np.uint8)
rect = np.array(((150, 100), (450, 150), (150, 500), (450, 500)))
rotAngle = 0
rotAngleMax = 360
offset = (300, 300)


def rotatePoint(p: tuple, theta: Number):
    cos, sin = np.cos(theta), np.sin(theta)

    return p[0] * cos - p[1] * sin, p[0] * sin + p[1] * cos


def translatePoint(p: tuple, offset):
    return p[0] + offset[0], p[1] + offset[1]


def rotateRect(angle: int):
    global rotAngle
    global rect
    global offset

    rotAngle = angle
    angle *= DEG_TO_RAD

    rotated = np.zeros((4, 2), dtype=np.int64)
    offsetNeg = tuple(map(mul, offset, (-1, -1)))

    for i, p in enumerate(rect):
        ptc = translatePoint(p, offsetNeg)
        rot = rotatePoint(ptc, angle)
        rotated[i] = translatePoint(rot, offset)

    rect = rotated
    cv2.setTrackbarPos(trackbar_name, window_name, rotAngle)


cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 639, 639)

cv2.createTrackbar(trackbar_name, window_name, rotAngle, rotAngleMax, rotateRect)

while True:
    frame = img.copy()

    mar = cv2.minAreaRect(rect)
    ((cx, cy), (w, h), angle) = mar
    print(f"MAR: w={round(w, 1)}, h={round(h, 1)}, angle={round(angle, 1)}")

    hw, hh = w * 0.5, h * 0.5

    cv2.rectangle(frame, rect[0], rect[3], (0, 255, 0), 6)
    cv2.circle(frame, (300, 300), 3, (255, 0, 0), 3)
    cv2.rectangle(frame, (int(cx - w * 0.5), int(cy - h * 0.5)), (int(cx + w * 0.5), int(cy + h * 0.5)), (0, 0, 255), 3)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(30)

    if key == ord('q') or key == 27:
        break

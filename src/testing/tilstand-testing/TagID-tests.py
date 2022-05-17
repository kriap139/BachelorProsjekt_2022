import cv2
import numpy as np

def calcTagLine(frame: np.ndarray, bbox: tuple, draw: bool = True):
    x, y, w, h = bbox
    points = (x, y), (x + w, y), (x + w, y + h), (x, y + h)
    cnt = np.array(points).reshape((-1, 1, 2)).astype(np.int32)

    # Find rotation of bbox
    mar = cv2.minAreaRect(cnt)
    ((cx, cy), (w, h), angle) = mar

    # angle in (0 , 180, 360)  -> MAR: w < h, angle=90
    # angle in (90, 270)       -> MAR: w > h, angle=90
    # if MAR was almost square in vertical position the angle would become -0!

    if w < h:
        angle += 90

    rad = np.radians(angle)

    print(angle)
from src.Backend.StateDetection.Methods.TypeDefs import TYDisplay
from src.Backend.StateDetection.Methods.constants import ValveState, ReturnType, RAD_TO_DEG
from src.Backend.StateDetection.Methods.filter_and_sorting import filterContours
from src.Util.Logging import Logging
from src.Backend.Valve import Valve
from typing import Union, Tuple
import numpy as np
import os
import cv2


def findMaskContours(hsv: np.ndarray, colorUpper: tuple, colorLower: tuple):
    # Find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)

    # Deleting noises which are in area of mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, mask


def pickMarking(img: np.ndarray, bbox: Tuple[int, int, int, int], contours_p: np.ndarray, contour_v: np.ndarray) -> np.ndarray:
    # If method is to be used in practise, piking the right mark in a cluster of valves will be critical
    # (Futher logic required)
    return contours_p[0]


def pipeMarking(img: np.ndarray, bbox: Tuple[int, int, int, int], v: Valve, display: TYDisplay) \
        -> Tuple[ReturnType, Union[ValveState, float]]:

    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    contours_p, mask_p = findMaskContours(hsv, (50, 100, 100), (70, 255, 255))
    display("Mask-Pipe", mask_p)

    contours_v, mask_v = findMaskContours(hsv, v.colorUpper, v.colorLower)
    display("Mask-Valve", mask_v)

    len_cp, len_cv = len(contours_p), len(contours_v)

    if len_cp and len_cv:
        filterContours(contours_p, filterThreshold=0.90)

        cv = max(contours_v, key=cv2.contourArea)
        cp = contours_p[0] if len_cp == 1 else pickMarking(img, bbox, contours_p, cv)

        rows, cols = img.shape[:2]

        (vx, vy, x, y) = cv2.fitLine(cv, cv2.DIST_L12, 0, 0.01, 0.01)
        lefty, righty = int((-x * vy / vx) + y), int(((cols - x) * vy / vx) + y)

        cv2.arrowedLine(img, (0, lefty), (cols - 1, righty), (150, 141, 184), thickness=6, tipLength=0.03)
        vec_v = np.array((vx, vy))

        (vx, vy, x, y) = cv2.fitLine(cp, cv2.DIST_L12, 0, 0.01, 0.01)
        lefty, righty = int((-x * vy / vx) + y), int(((cols - x) * vy / vx) + y)

        cv2.arrowedLine(img, (0, lefty), (cols - 1, righty), (150, 141, 184), thickness=6, tipLength=0.03)
        vec_p = np.array((vx, vy))

        angle, = np.arccos(np.dot(vec_p, vec_v)) * RAD_TO_DEG

        display("Processed image", img)

        return ReturnType.ANGLE, angle

    return ReturnType.STATE, ValveState.UNKNOWN


from src.Backend.StateMethods.constants import TYDisplay, Number
from src.Backend.Valve import ValveState, Valve
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, Union, Dict
from src.Config import Config
from src.Backend.DataClasses import BBoxData
from src.Backend.Tools import checkForEdgeIntersections


def filterContours(conts, requiredLength=19, filterThreshold=0.16):
    """fiilters contours based on their arcLength. Contours Smaller then the requiredLength(in pixels)
    parameter gets removed. Additionally any Contour that are to small commpared to the biggest countour in the list,
    will also be removed. The cantors minimum length (compared to the longest in the list),
    is given by the filterThreshold param in precent[0-1]"""

    if not conts.__len__():
        return tuple()

    lengths = tuple(cv2.arcLength(cont, True) for cont in conts)

    longest = np.max(lengths)
    filtered = []

    for i in range(conts.__len__()):
        if lengths[i] > requiredLength and (lengths[i]/longest > filterThreshold):
            filtered.append(conts[i])

    return filtered


class ColorStateDetector:
    def __init__(self, valveClasses=None, angleClosedThreshDeg: float = 70, angleOpenThreshDeg: float = 19):

        self.angleClosedThreshDeg = angleClosedThreshDeg
        self.angleOpenThreshDeg = angleOpenThreshDeg

        if valveClasses is not None:
            self.valveClasses = valveClasses
        else:
            self.valveClasses = Config.loadValveInfoData().valveClasses

    @staticmethod
    def map(val: Number, inMin: Number, inMax: Number, outMin: Number, outMax: Number) -> float:
        return ((val - inMin) * (outMax - outMin) / float(inMax - inMin)) + outMin

    def calcState(self, angle: float) -> ValveState:
        deg = np.abs(np.degrees(angle))

        if deg > 90:
            newDeg = self.map(val=deg, inMin=90, inMax=180, outMin=90, outMax=0)
        else:
            newDeg = deg

        #print(f"angle={np.degrees(angle)}, newAngle={newDeg}")

        if newDeg > self.angleClosedThreshDeg:
            return ValveState.CLOSED
        elif newDeg < self.angleOpenThreshDeg:
            return ValveState.OPEN
        else:
            return ValveState.UNKNOWN

    @staticmethod
    def calcTagLine(frame: np.ndarray, tagBox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = tagBox
        wh, hh = int(w * 0.5), int(h * 0.5)
        cx, cy = x + wh, y + hh

        # find the smallest Symmetry line to tagBox
        if w > h:
            p1 = (x + wh, y)
            p2 = (x + wh, y + h)
        else:
            p1 = (x, y + hh)
            p2 = (x + w, y + hh)

        # find normalized heading vector
        vec = np.array((p1[0] - p2[0], p1[1] - p2[1]))
        vec = vec / np.linalg.norm(vec)

        return vec

    def stateDetect(self, frame: np.ndarray, data: BBoxData, draw: bool = True) -> ValveState:

        x, y, w, h = data.box
        img = frame[y:y + h, x:x + w]

        vec_p = self.calcTagLine(frame, data.tagBox)
        cls = self.valveClasses.get(data.classID)

        if (vec_p is None) or (cls is None) or (len(cls.colorLower) != 3) or (len(cls.colorUpper) != 3) \
                or (img.shape[0] == 0) or (img.shape[1] == 0):
            return ValveState.UNKNOWN

        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(hsv, np.array(cls.colorLower), np.array(cls.colorUpper))

        # Deleting noises which are in area of mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # define kernel size
        kernel = np.ones((5, 5), np.uint8)

        # Remove unnecessary noise from mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours_h, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_h) > 0:
            c = max(contours_h, key=cv2.contourArea)

            rows, cols = img.shape[:2]
            (vx, vy, x, y) = cv2.fitLine(c, cv2.DIST_L12, 0, 0.01, 0.01)

            lefty, righty = int((-x * vy / vx) + y), int(((cols - x) * vy / vx) + y)

            fh, fw, _ = frame.shape

            if draw:
                if (np.abs(lefty) < fh) and (np.abs(righty) < fh):
                    cv2.arrowedLine(img, (0, lefty), (cols - 1, righty), (0, 223, 255), thickness=6, tipLength=0.03)

                # cv2.imshow("BBox crop", img)
                # cv2.imshow("mask", mask)

            vec_v = np.array((vx, vy))

            dot = np.dot(np.transpose(vec_v), vec_p)
            angle, = np.arccos(dot)

            return self.calcState(angle)
        return ValveState.UNKNOWN

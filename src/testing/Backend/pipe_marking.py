from src.Backend.StateMethods.Methods import TYDisplay
from src.Backend.StateMethods.Methods import ReturnType
from src.Backend import ValveState
from src.Backend.StateMethods.Methods import filterContours
from src.Backend.Valve import Valve
from typing import Union, Tuple
import numpy as np
import cv2
import numbers


def findMaskContours(hsv: np.ndarray,  colorLower: tuple, colorUpper: tuple,):
    # Find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)

    # Deleting noises which are in area of mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, mask


def pickMarking(img: np.ndarray, bbox: Tuple[int, int, int, int], contours_p: np.ndarray, contour_v: np.ndarray) -> np.ndarray:
    # If method is to be used in practise, piking the right mark in a cluster of valves will be critical
    # (Futher logic required)
    return max(contours_p, key=cv2.contourArea)


# old Pipe calculation didn't work correctly!
def calcPipeMARVec(img: np.ndarray, marPipe, errMargin: numbers.Number = 16) -> Union[Tuple[tuple, tuple], None]:

    ((cx, cy), (w, h), angle) = marPipe
    # print(f"MAR: w={round(w, 1)}, h={round(h, 1)}, angle={round(angle, 1)}", end=',   ')

    # Make a diagonal line through the MAR of the pipe
    points = (
        (int(cx), int(cy - h * 0.5)), (int(cx), int(cy + h * 0.5)),
        (int(cx - w * 0.5), int(cy)), (int(cx + w * 0.5), int(cy))
    )

    # Vec from p1 to p2: (p2x - p1x, p2y - p1y)
    vectors = [
        (points[0][0] - points[1][0], points[0][1] - points[1][1]),
        (points[2][0] - points[3][0], points[2][1] - points[3][1])
    ]

    vectors[0] = vectors[0] / np.linalg.norm(vectors[0])
    vectors[1] = vectors[1] / np.linalg.norm(vectors[1])

    iw, ih, _ = img.shape
    iwm, ihm = int(iw * 0.5), int(ih * 0.5)

    # Points vec_ix: (iwm, 0), (iw, ihm)
    # Points vec_iy: (0, ihm), (iwm, ih)
    vec_ix = np.array((iw - iwm, ihm))
    vec_ix = vec_ix / np.linalg.norm(vec_ix)

    vec_iy = np.array((iwm, ih - ihm))
    vec_iy = vec_iy / np.linalg.norm(vec_iy)

    dot = np.dot(vectors[0], vec_ix)
    angle_ix = np.degrees(np.arccos(dot))

    dot = np.dot(vectors[0], vec_iy)
    angle_iy = np.degrees(np.arccos(dot))

    if angle_ix <= angle_iy:
        index = 0
        ip1, ip2 = 0, 1
    else:
        index = 1
        ip1, ip2 = 2, 3

    # print(f"Points: p1={points[ip1]}, p2={points[ip2]}, Angles: angle_ix={angle_ix}, angle_iy={angle_iy}")
    cv2.arrowedLine(img, points[ip2], points[ip1], (0, 0, 255), thickness=6, tipLength=0.06)

    return vectors[index]


# Old had problems
def calcPipeVec(img: np.ndarray, cp: np.ndarray) -> np.ndarray:
    rows, cols, _ = img.shape
    (vx, vy, x, y) = cv2.fitLine(cp, cv2.DIST_L12, 0, 0.01, 0.01)
    lefty, righty = int((-x * vy / vx) + y), int(((cols - x) * vy / vx) + y)

    cv2.arrowedLine(img, (0, lefty), (cols - 1, righty), (150, 141, 184), thickness=6, tipLength=0.03)
    return np.array((vx, vy))


def calcPipeVec2(img: np.ndarray, cp: np.ndarray, display: TYDisplay):
    mar_pipe = cv2.minAreaRect(cp)
    ((cx, cy), (w, h), angle) = mar_pipe

    box_pipe = np.int64(cv2.boxPoints(mar_pipe))
    cv2.drawContours(img, [box_pipe], -1, (0, 0, 255), 3)
    display("MAR-Pipe", img)

    # angle in (0 , 180, 360)  -> MAR: w < h, angle=90
    # angle in (90, 270)       -> MAR: w > h, angle=90
    # if MAR was almost square in vertical position the angle would become -0!

    if w < h:
        angle += 90

    rad = np.radians(angle)
    vec_p = (np.cos(rad), np.sin(rad))

    return vec_p, mar_pipe, box_pipe


def pipeMarking(img: np.ndarray, bbox: Tuple[int, int, int, int], v: Valve, display: TYDisplay) \
        -> Tuple[ReturnType, Union[ValveState, float]]:

    x, y, w, h = bbox
    img = img[y:y+h, x:x+w]

    display("OG_image", img)

    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    display("hsv", hsv)

    contours_p, mask_p = findMaskContours(hsv, colorLower=(51, 100, 78), colorUpper=(65, 255, 255))
    display("Mask-Pipe", mask_p)

    contours_v, mask_v = findMaskContours(hsv, v.cls.colorLower, v.cls.colorUpper)
    display("Mask-Valve", mask_v)

    len_cp, len_cv = len(contours_p), len(contours_v)

    if len_cp and len_cv:
        filterContours(contours_p, filterThreshold=0.60)

        cv = max(contours_v, key=cv2.contourArea)
        cp = contours_p[0] if (len_cp == 1) else pickMarking(img, bbox, contours_p, cv)

        rows, cols = img.shape[:2]
        (vx, vy, x, y) = cv2.fitLine(cv, cv2.DIST_L12, 0, 0.01, 0.01)
        lefty, righty = int((-x * vy / vx) + y), int(((cols - x) * vy / vx) + y)

        cv2.arrowedLine(img, (0, lefty), (cols - 1, righty), (150, 141, 184), thickness=6, tipLength=0.03)
        vec_v = np.array((vx, vy))

        vec_p, mar, mar_points = calcPipeVec2(img, cp, display)

        dot = np.dot(np.transpose(vec_p), vec_v)
        angle, = np.degrees(np.arccos(dot))

        # Have to check if Rectangle is placed vertical (with the pipe) or horizontal (placed normal on the pipe).
        # All rectangles had to be placed horizontal with valveClass 0, du to the shape of the valve.
        # if this method were to be used in practise (not just for a proof of consept), a better solution should be developed

        if v.cls.classID == 0:
            angle = np.abs(angle - 90)

        display("Processed image", img)
        return ReturnType.ANGLE, angle

    return ReturnType.STATE, ValveState.UNKNOWN


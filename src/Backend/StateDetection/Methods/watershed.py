import numbers
from typing import Union, Tuple
from src.Backend.StateDetection.Methods.TypeDefs import TYDisplay
from src.Backend.Valve import Valve
from rembg.bg import remove
from src.Backend.StateDetection.Methods.constants import *
from PIL import Image
import numpy as np
import cv2


def watershed(img: np.ndarray, display: TYDisplay) -> np.ndarray:
    # To apply median Blur to imag for removing the unnecessary details from the image
    img = cv2.medianBlur(img, 35)  # kernel size is 25 or 35

    # Convert the image to Grayscale
    gray_p = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Threshold (Inverse Binary with OTSU method as well)  in order to make it black and white
    ret, thresh = cv2.threshold(gray_p, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    display("thresh", thresh, cmap='gray')

    # To define a kernel size 3X3
    kernel = np.ones((3, 3), np.uint8)

    # To remove unnecessary noise by applying morphologyEx
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # To grab the sure Background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    display("sure_bg", sure_bg, cmap='gray')

    # To finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    display("dist_TFM", dist_transform, cmap='gray')

    # 0.7*dist_transform.max()
    ret, sure_fg = cv2.threshold(dist_transform, 0.998 * dist_transform.max(), 255, 0)
    display("sure_fg", sure_fg, cmap='gray')

    # Finding unknown region in the image
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    display("unknown", unknown, cmap='gray')

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # mark the region of unknown with zero
    markers[unknown == 255] = 0
    display("markers", markers, cmap='gray')

    # Applying Watershed Algorithm to find Markers
    markers = cv2.watershed(img, markers)

    return markers


def removeBKG(img: np.ndarray) -> np.ndarray:
    """ remove background from 'img'. The original img is not altered!"""

    result: np.array = remove(img)
    img = Image.fromarray(result).convert("RGBA")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def calcPipeMARVec(img: np.ndarray, marPipe, errMargin: numbers.Number = 16) -> Union[Tuple[tuple, tuple], None]:

    ((cx, cy), (w, h), angle) = marPipe

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

    # print(f"MAR: w={round(w, 1)}, h={round(h, 1)}, angle={round(angle, 1)}", end=',   ')
    # print(f"Points: p1={points[ip1]}, p2={points[ip2]}, Angles: angle_ix={angle_ix}, angle_iy={angle_iy}")

    cv2.arrowedLine(img, points[ip2], points[ip1], (0, 0, 255), thickness=6, tipLength=0.06)

    return vectors[index]


def watershedVec(img: np.ndarray, bbox: Tuple[int, int, int, int], v: Valve, display: TYDisplay) \
            -> Tuple[ReturnType, Union[ValveState, float]]:

    display("OG_image", img)

    blurred = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)
    display("blurred", blurred)

    rembg_img = removeBKG(blurred)
    display("Rem_BKG", rembg_img)

    markers = watershed(rembg_img, display)
    display("Markers_final", markers)

    # Finding Contours on Markers
    # cv2.RETR_EXTERNAL:Only extracts external contours
    # cv2.RETR_CCOMP: Extracts both internal and external contours organized in a two-level hierarchy
    # cv2.RETR_TREE: Extracts both internal and external contours organized in a  tree graph
    # cv2.RETR_LIST: Extracts all contours without any internal/external relationship
    contours_p, hierarchy_p = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours left-to-right
    sorted_contours_p = sorted(contours_p, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # filtered Contours
    filter_arr = sorted_contours_p[1]
    for i in range(len(filter_arr)):
        # draw the rquaired contour
        cv2.drawContours(rembg_img, filter_arr, i, (0, 255, 0), 20)

    mar_pipe = cv2.minAreaRect(filter_arr)
    box_pipe = np.int64(cv2.boxPoints(mar_pipe))
    cv2.drawContours(rembg_img, [box_pipe], -1, (255, 0, 0), 5)

    vec_p = calcPipeMARVec(rembg_img, mar_pipe)

    if vec_p is None:
        return ReturnType.STATE, ValveState.UNKNOWN

    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    display("hsv", hsv)

    # Find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(hsv, v.colorLower, v.colorUpper)

    # Deleting noises which are in area of mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    display("Mask_valve", mask)

    contours_h, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_h) > 0:
        c = max(contours_h, key=cv2.contourArea)

        rows, cols = rembg_img.shape[:2]
        (vx, vy, x, y) = cv2.fitLine(c, cv2.DIST_L12, 0, 0.01, 0.01)
        lefty, righty = int((-x * vy / vx) + y), int(((cols - x) * vy / vx) + y)

        cv2.arrowedLine(rembg_img, (0, lefty), (cols - 1, righty), (150, 141, 184), thickness=6, tipLength=0.03)

        vec_v = np.array((vx, vy))

        dot = np.dot(vec_p, vec_v)
        angle, = np.degrees(np.arccos(dot))

        display("Processed image", rembg_img)
        return ReturnType.ANGLE, angle

    return ReturnType.STATE, ValveState.UNKNOWN


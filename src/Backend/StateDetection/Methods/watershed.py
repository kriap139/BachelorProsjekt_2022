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


def calcPipeMARVec(img: np.ndarray, cx_p, cy_p, w_p, h_p, angle_p) -> Tuple[tuple, tuple]:

    # Make a diagonal line through the MAR of the pipe
    print("MAR: ", w_p, h_p, angle_p * RAD_TO_DEG)

    if w_p > h_p:
        if angle_p == -0:
            p_p1 = (int(cx_p - w_p * 0.5), int(cy_p))
            p_p2 = (int(cx_p + w_p * 0.5), int(cy_p))
        else:
            p_p1 = (int(cx_p), int(cy_p - h_p * 0.5))
            p_p2 = (p_p1[0], int(cy_p + h_p * 0.5))
    elif w_p < h_p:
        if angle_p == -0:
            p_p1 = (int(cx_p), int(cy_p - h_p * 0.5))
            p_p2 = (p_p1[0], int(cy_p + h_p * 0.5))
        else:
            p_p1 = (int(cx_p - w_p * 0.5), int(cy_p))
            p_p2 = (int(cx_p + w_p * 0.5), int(cy_p))
    else:
        p_p1 = (int(cx_p), int(cy_p - h_p * 0.5))
        p_p2 = (p_p1[0], int(cy_p + h_p * 0.5))

    # create a normalized vector through the MAR
    vec_p = np.array((p_p2[0] - p_p1[0], p_p2[1] - p_p1[1]))
    vec_p = vec_p / np.linalg.norm(vec_p)

    cv2.arrowedLine(img, p_p2, p_p1, (218, 165, 32), thickness=6, tipLength=0.1)

    return vec_p


def watershedVec(img: np.ndarray, bbox: Tuple[int, int, int, int], v: Valve, display: TYDisplay) \
            -> Tuple[ReturnType, Union[ValveState, float]]:

    display("OG_image", img)

    rembg_img = removeBKG(img)
    display("Rem_BKG", rembg_img)

    # getting the markers by using whatersheld from the rembg_img
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

    # Find Minimum Area Rect
    mar_pipe = cv2.minAreaRect(filter_arr)

    box_pipe = np.int64(cv2.boxPoints(mar_pipe))
    cv2.drawContours(rembg_img, [box_pipe], -1, (255, 0, 0), 5)

    ((cx_p, cy_p), (w_p, h_p), angle_p) = mar_pipe
    vec_p = calcPipeMARVec(rembg_img, cx_p, cy_p, w_p, h_p, angle_p)

    # blur the orginal image to remove the noise
    blurred = cv2.GaussianBlur(img, (11, 11), 0)

    # Convert the image to HSV colorspace
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    display("hsv", hsv)

    # Find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(hsv, v.colorLower, v.colorUpper)

    # Deleting noises which are in area of mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    display("Mask_valve", mask)

    # Find contours from the mask
    contours_h, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours_h) > 0:
        # get max contour
        c = max(contours_h, key=cv2.contourArea)

        rows, cols = rembg_img.shape[:2]
        (vx, vy, x, y) = cv2.fitLine(c, cv2.DIST_L12, 0, 0.01, 0.01)
        lefty, righty = int((-x * vy / vx) + y), int(((cols - x) * vy / vx) + y)

        cv2.arrowedLine(rembg_img, (0, lefty), (cols - 1, righty), (150, 141, 184), thickness=6, tipLength=0.03)

        vec_v = np.array((vx, vy))
        angle, = np.arccos(np.dot(vec_p, vec_v)) * RAD_TO_DEG

        display("Processed image", rembg_img)
        return ReturnType.ANGLE, angle

    return ReturnType.STATE, ValveState.UNKNOWN


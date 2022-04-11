from src.Backend.StateDetection.Methods.TypeDefs import TYDisplay
from src.Backend.StateDetection.Methods.constants import ValveState, ReturnType
from src.Util.Logging import Logging
from src.Backend.Valve import Valve
from typing import Union, Tuple
import numpy as np
import os
import cv2
import glob


class SIFTImageHandler:
    SIFT_IMAGES_PATH = os.path.abspath(
        os.path.join("resources", "testing", "ventil-tilstand", "tilstand-test", "sift", "cropped")
    )

    VALVE_COMP_IMAGES = {}

    @classmethod
    def fetchImages(cls, v: Valve):

        if v.classID in cls.VALVE_COMP_IMAGES.keys():
            return cls.VALVE_COMP_IMAGES[v.classID]["OPEN"], cls.VALVE_COMP_IMAGES[v.classID]["CLOSED"]

        img_open_path = os.path.join(cls.SIFT_IMAGES_PATH, str(v.classID), "OPEN")
        img_closed_path = os.path.join(cls.SIFT_IMAGES_PATH, str(v.classID), "CLOSED")

        if not os.path.exists(img_open_path):
            Logging.print(f"SIFT image Dir doesn't exist: {img_open_path}")
            return None, None
        if not os.path.exists(img_closed_path):
            Logging.print(f"SIFT image Dir doesn't exist: {img_open_path}")
            return None, None

        img_open = []
        img_closed = []

        cls.VALVE_COMP_IMAGES[v.classID] = {"OPEN": img_open, "CLOSED": img_closed}

        for f in glob.iglob(f"{img_open_path}{os.path.sep}*"):
            image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
            img_open.append(image)

        for f in glob.iglob(f"{img_closed_path}{os.path.sep}*"):
            image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
            img_closed.append(image)

        return img_open, img_closed


def sift(img: np.ndarray, bbox: Tuple[int, int, int, int], v: Valve, display: TYDisplay) \
        -> Tuple[ReturnType, ValveState]:

    comp_open, comp_closed = SIFTImageHandler.fetchImages(v)

    if comp_open is None or (comp_closed is None):
        return ReturnType.STATE, ValveState.UNKNOWN

    sft = cv2.SIFT_create()
    blurred = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    kp, des = sft.detectAndCompute(blurred, None)

    num_kp_arr = [0, 0]

    for i, comp_images in enumerate((comp_open, comp_closed)):

        num_kp = 0
        comp_good = []

        for comp in comp_images:
            com_kp, comp_des = sft.detectAndCompute(comp, None)

            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des, comp_des, k=2)

            # Apply ratio test
            for m, n in matches:

                if m.distance < 0.75 * n.distance:
                    comp_good.append([m])

                if len(kp) >= len(com_kp):
                    num_kp = len(kp)
                else:
                    num_kp = len(com_kp)

            num_kp_arr[i] += len(comp_good) / num_kp * 100

    sim_open = num_kp_arr[0]
    sim_closed = num_kp_arr[1]

    return (ReturnType.STATE, ValveState.OPEN) if sim_open > sim_closed else (ReturnType.STATE, ValveState.CLOSED)













from src.Backend.StateMethods.constants import TYDisplay
from src.Backend import ValveState
from src.Config import Config
from src.Backend.Logging import LOG, LOGType
from src.Backend.Valve import Valve, ValveClass
from typing import Tuple, Iterable, Dict, List, Union, Optional
import numpy as np
import os
import cv2
import glob
from multiprocessing.shared_memory import SharedMemory
from src.Backend.DataClasses import SharedImage, BBoxData, SiftRefImg


class SIFTImageHandler:
    SIFT_IMAGES_PATH = Config.getSiftRefsDir()
    VALVE_COMP_IMAGES: Dict[int, Dict[str, List[SiftRefImg]]] = {}

    @classmethod
    def loadAllSiftRefImages(cls):
        for classID in Config.loadValveInfoData().valveClasses.keys():
            cls.fetchImages(classID)

    @classmethod
    def fetchImages(cls, classID: int) \
            -> Union[Tuple[List[SiftRefImg], List[SiftRefImg]], Tuple[None, None]]:

        if classID in cls.VALVE_COMP_IMAGES.keys():
            return cls.VALVE_COMP_IMAGES[classID]["OPEN"], cls.VALVE_COMP_IMAGES[classID]["CLOSED"]

        img_open_path = os.path.join(cls.SIFT_IMAGES_PATH, str(classID), "OPEN")
        img_closed_path = os.path.join(cls.SIFT_IMAGES_PATH, str(classID), "CLOSED")

        if not os.path.exists(img_open_path):
            # LOG(f"SIFT image Dir doesn't exist: {img_open_path}")
            return None, None
        if not os.path.exists(img_closed_path):
            # LOG(f"SIFT image Dir doesn't exist: {img_open_path}")
            return None, None

        img_open = []
        img_closed = []
        sft = cv2.SIFT_create()

        cls.VALVE_COMP_IMAGES[classID] = {"OPEN": img_open, "CLOSED": img_closed}

        for f in glob.iglob(f"{img_open_path}{os.path.sep}*"):
            image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
            kp, des = sft.detectAndCompute(image, None)

            img_open.append(SiftRefImg(image, kp, des))

        for f in glob.iglob(f"{img_closed_path}{os.path.sep}*"):
            image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
            kp, des = sft.detectAndCompute(image, None)

            img_closed.append(SiftRefImg(image, kp, des))

        return img_open, img_closed


class SIFTImageHandlerSHM:
    SIFT_IMAGES_PATH = Config.getSiftRefsDir()
    SHM_MAPPINGS: Dict[int, Dict[str, List[SharedImage]]] = {}

    @classmethod
    def loadSIFTImagesToSHM(cls, valveClasses: Iterable[ValveClass]):
        for vc in valveClasses:
            img_open_path = os.path.join(cls.SIFT_IMAGES_PATH, str(vc.classID), "OPEN")
            img_closed_path = os.path.join(cls.SIFT_IMAGES_PATH, str(vc.classID), "CLOSED")

            if not os.path.exists(img_open_path):
                LOG(f"SIFT Ref open image Dir for classID {vc.classID} doesn't exist: {img_open_path}")
                continue
            if not os.path.exists(img_closed_path):
                LOG(f"SIFT Ref closed image Dir for classID {vc.classID} doesn't exist: {img_open_path}")
                continue

            cls.SHM_MAPPINGS[vc.classID] = {"OPEN": [], "CLOSED": []}

            shmOpen: list = cls.SHM_MAPPINGS[vc.classID]["OPEN"]
            shmClosed: list = cls.SHM_MAPPINGS[vc.classID]["CLOSED"]

            for f in glob.iglob(f"{img_open_path}{os.path.sep}*"):
                image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                image: np.ndarray = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)

                shm = SharedMemory(create=True, size=image.nbytes)
                copy = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
                copy[:] = image[:]

                shmOpen.append(SharedImage(shm.name, copy.dtype, copy.shape))
                shm.close()

            for f in glob.iglob(f"{img_closed_path}{os.path.sep}*"):
                image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)

                shm = SharedMemory(create=True, size=image.nbytes)
                copy = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
                copy[:] = image[:]

                shmClosed.append(SharedImage(shm.name, copy.dtype, copy.shape))
                shm.close()

    @classmethod
    def freeSIFTImagesFromSHM(cls):
        for dct in cls.SHM_MAPPINGS.values():
            for arr in dct.values():
                for sim in arr:
                    shm = SharedMemory(sim.memName)
                    shm.close()
                    shm.unlink()

    @classmethod
    def fetchImages(cls, classID) -> Union[Tuple[list, list], Tuple[None, None]]:

        imgOpen = []
        imgClosed = []

        dct = cls.SHM_MAPPINGS.get(classID, default=None)

        if dct is None:
            return None, None

        shImgOpen = dct["OPEN"]
        shImgClosed = dct["CLOSED"]

        for data in shImgOpen:
            sim: SharedImage = data

            shm = SharedMemory(sim.memName)
            img = np.ndarray(shape=sim.shape, dtype=sim.dType, buffer=shm.buf)

            imgOpen.append(img)

        for data in shImgClosed:
            sim: SharedImage = data

            shm = SharedMemory(sim.memName)
            img = np.ndarray(shape=sim.shape, dtype=sim.dType, buffer=shm.buf)

            imgClosed.append(img)

        return imgOpen, imgClosed



class SiftStateDetector:
    def __init__(self):
        self.valveClasses = Config.loadValveInfoData().valveClasses

    @classmethod
    def sift(cls, img: np.ndarray, data: BBoxData) -> ValveState:

        x, y, w, h = data.box
        img = img[y:y + h, x:x + w]

        if (img.shape[0] == 0) or (img.shape[1] == 0):
            return ValveState.UNKNOWN

        comp_open, comp_closed = SIFTImageHandler.fetchImages(data.classID)

        if comp_open is None or (comp_closed is None):
            return ValveState.UNKNOWN

        sft = cv2.SIFT_create()
        blurred = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
        kp, des = sft.detectAndCompute(blurred, None)

        bf = cv2.BFMatcher()
        num_kp_arr = [0, 0]

        # sim, comp_kp, comp_good, comp_img, state (0=OPEN, 1=CLOSED)
        highest = [0, None, None, None, -1]

        for i, comp_img_data in enumerate((comp_open, comp_closed)):

            for cid in comp_img_data:
                cid: SiftRefImg = cid

                num_kp = 0
                comp_good = []
                # com_kp, comp_des = sft.detectAndCompute(comp, None)

                # BFMatcher with default params
                matches = bf.knnMatch(des, cid.des, k=2)

                # Apply ratio test
                for m, n in matches:

                    if m.distance < 0.75 * n.distance:
                        comp_good.append([m])

                    if len(kp) >= len(cid.kp):
                        num_kp = len(kp)
                    else:
                        num_kp = len(cid.kp)

                sim = (len(comp_good) / num_kp * 100) if (num_kp != 0) else 0

                if sim > highest[0]:
                    highest[0] = sim
                    highest[1] = cid.kp
                    highest[2] = comp_good
                    highest[3] = cid.img
                    highest[4] = i

                num_kp_arr[i] += sim

        # draw kp of image with the highest match
        # his = highest[3]
        # if (his is not None) and (img.shape[1] != 0) and (img.shape[0] != 0):
        #     draw = cv2.drawMatchesKnn(img, kp, highest[3], highest[1], highest[2], None,
        #                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #
        #    path = Config.createAppDataPath("images", "results", "SDST", "sift", "RefResult")
        #    cv2.imwrite(os.path.join(path, f"{SIFTImageHandler.getFrameID()}.jpg"), draw)

        sim_open = num_kp_arr[0]
        sim_closed = num_kp_arr[1]

        return ValveState.OPEN if sim_open > sim_closed else ValveState.CLOSED













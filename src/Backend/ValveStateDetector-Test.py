from src.Backend.Processes import (
    UnifiedStateDetectProcess, PreStateDetectArgs, PreStateDetectProcess, freeShm
)

from src.Backend.DataClasses import ImageData, BBoxData
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import signal
from queue import Empty
import time
from src.Backend.Valve import ValveState
from src.Config import Config

mainExitEvent = mp.Event()

process = UnifiedStateDetectProcess(mainExitEvent)
process.start()

import numpy as np
import cv2 as cv


def mainExitEventHandler(signum, frame):
    mainExitEvent.set()
    process.shutdown()

    try:
        process.join()
    except ValueError as e:
        print(e.args.__str__())


signal.signal(signal.SIGINT, mainExitEventHandler)
signal.signal(signal.SIGTERM, mainExitEventHandler)


class ValveStateDetectorTest:
    TAG_BOX_COLOR = (155, 103, 60)

    VALVE_COLORS_BGRA = {
        ValveState.UNKNOWN: [2, 210, 238, 1],
        ValveState.CLOSED: [10, 34, 224, 1],
        ValveState.OPEN: [0, 230, 17, 1]
    }

    def __init__(self):
        self.queue = process.getPostStateDetectQueue()

    def run(self):
        while not mainExitEvent.is_set():
            try:
                imgData = self.queue.get(block=False)

                if imgData is not None:
                    sm = imgData.sharedImg

                    shm = SharedMemory(name=sm.memName)
                    img = np.ndarray(shape=sm.shape, dtype=sm.dType, buffer=shm.buf)

                    self.draw(img, imgData)
                    cv.imshow("result", img)
                    cv.waitKey(1)

                    freeShm(shm)
            except ValueError as e:
                pass
            except Empty as e:
                time.sleep(0.3)

    def draw(self, img: np.ndarray, imgData: ImageData):
        font = cv.FONT_HERSHEY_PLAIN

        for bbox in imgData.bboxes:
            x, y, w, h = bbox.box

            color = self.VALVE_COLORS_BGRA.get(bbox.valveState)
            label = f"id={bbox.valveID}, cls={bbox.classID}"

            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.rectangle(img, (x, y), (x + w, y + 30), color, -1)
            cv.putText(img, label, (x, y + 30), font, 2, (255, 255, 255), 3)

            tagBox = bbox.tagBox

            if tagBox is not None:
                label = "Tag"
                color = self.TAG_BOX_COLOR
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv.rectangle(img, (x, y), (x + w, y + 30), color, -1)
                cv.putText(img, label, (x, y + 30), font, 2, (255, 255, 255), 2)


if __name__ == "__main__":
    video = Config.createAppDataPath("video", fName="al.mp4")
    process.activate(PreStateDetectArgs(video, "test_stream"))

    tester = ValveStateDetectorTest()
    tester.run()











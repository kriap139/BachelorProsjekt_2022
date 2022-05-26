from src.Backend.Processes import (
    UnifiedStateDetectProcess, PreStateDetectArgs, PreStateDetectProcess, freeShm, freeShmFromImageData
)

from src.Backend.DataClasses import ImageData, BBoxData
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import signal
from queue import Empty
import time
from src.Backend.Valve import ValveState
from src.Config import Config
from typing import Iterable, Dict, Tuple, Optional
import os
import time

mainExitEvent = mp.Event()
process = PreStateDetectProcess(mainExitEvent, numSDProcs=3)
process.start()

import numpy as np
import cv2 as cv
import PIL
from PIL import Image


class ValveStateDetectorTest:
    TAG_BOX_COLOR = (155, 103, 60)

    VALVE_COLORS_BGRA = {
        ValveState.UNKNOWN: [2, 210, 238, 1],
        ValveState.CLOSED: [10, 34, 224, 1],
        ValveState.OPEN: [0, 230, 17, 1]
    }

    def __init__(self, frameSavePath: str, finalFrameSavePath: str, saveFrames: bool = True):
        self.queue = process.getPostStateDetectQueue()
        self.font = cv.FONT_HERSHEY_PLAIN
        self.frameSavePath = frameSavePath
        self.finalFrameSavePath = finalFrameSavePath
        self.saveFrames: bool = saveFrames
        self.queuedFrames: Dict[int, ImageData] = {}
        self.currFrameID: int = 1

    def run(self):
        start = time.time()
        shm, frame = None, None

        while not mainExitEvent.is_set():
            try:
                imgData: ImageData = self.queue.get(block=False)
                print(imgData)

                if imgData is None:
                    shm, frame = self.checkQueuedFrames()
                elif imgData.frameID == self.currFrameID:
                    shm, frame = self.handleFrame(imgData)
                elif imgData.frameID != self.currFrameID:
                    self.queuedFrames[imgData.frameID] = imgData
                    shm, frame = self.checkQueuedFrames()

            except ValueError as e:
                pass
            except Empty as e:
                # print(f"Queue empty")
                msg = process.getPipe().recv() if process.getPipe().poll() else ""

                if msg == "finished":
                    self.shutdown()
                    break
                time.sleep(0.1)

                shm, frame = self.checkQueuedFrames()

            if cv.waitKey(1) == ord('q'):
                self.shutdown()
                break

            if (shm is not None) and (frame is not None):
                cv.imshow("result", frame)
                cv.waitKey(1)
                freeShm(shm)
                shm, frame = None, None

            print(f"currFrameID={self.currFrameID - 1}, queued={len(self.queuedFrames)}")

        endTime = time.time() - start
        fps = self.currFrameID / endTime

        print(f"time={round(endTime, 3)}, FPS={round(fps, 3)}, TotalFrames={self.currFrameID - 1}")

    def checkQueuedFrames(self) -> Tuple[Optional[SharedMemory], Optional[np.ndarray]]:
        if len(self.queuedFrames) > 0:
            imgData: ImageData = self.queuedFrames.pop(self.currFrameID, None)

            if imgData is not None:
                return self.handleFrame(imgData)

        return None, None

    def handleFrame(self, imgData: ImageData) -> Tuple[SharedMemory, np.ndarray]:
        sm = imgData.sharedImg

        shm = SharedMemory(name=sm.memName)
        frame = np.ndarray(shape=sm.shape, dtype=sm.dType, buffer=shm.buf)

        if self.saveFrames:
            cv.imwrite(os.path.join(self.frameSavePath, f"{imgData.frameID}.jpg"), frame)

        # Drawing -----------------------------------------------------------------
        unassignedTags = (imgData.tagsData[i] for i in imgData.unassignedIndexes)
        for td in unassignedTags:
            self.drawTag(frame, td.tagBox, td.tagID, color=(0, 0, 0))

        self.draw(frame, imgData.bboxes)

        # for line in tagLines:
        #    cv.line(frame, line[0].toTuple(), line[1].toTuple(), (0, 255, 0), 3)

        resized = self.resizeImage(frame, width=600)
        cv.putText(resized, f"Frame {self.currFrameID}", (3, 30), self.font, 2, color=(0, 255, 0), thickness=2)
        # -------------------------------------------------------------------------

        if self.saveFrames:
            cv.imwrite(os.path.join(self.finalFrameSavePath, f"{imgData.frameID}.jpg"), resized)

        self.currFrameID += 1

        return shm, resized

    def draw(self, img: np.ndarray, bboxes: Iterable[BBoxData]):
        for bbox in bboxes:
            x, y, w, h = bbox.box

            color = self.VALVE_COLORS_BGRA.get(bbox.valveState)
            label = f"ID={bbox.valveID}, cls={bbox.classID}"

            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.rectangle(img, (x, y), (x + w, y + 30), color, -1)
            cv.putText(img, label, (x, y + 30), self.font, 2, (255, 255, 255), 3)

            tagBox = bbox.tagBox

            if tagBox is not None:
                x, y, w, h = tagBox

                label = f"ID={bbox.valveID}"
                color = self.TAG_BOX_COLOR

                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv.rectangle(img, (x, y), (x + w, y + 30), color, -1)
                cv.putText(img, label, (x, y + 30), self.font, 2, (255, 255, 255), 2)

    def drawTag(self, img: np.ndarray, tagBox: tuple, tagId: str = None, color: tuple = (255, 255, 255)):
        x, y, w, h = tagBox
        color = self.TAG_BOX_COLOR

        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.rectangle(img, (x, y), (x + w, y + 30), color, -1)

        if tagId is not None:
            cv.putText(img, f"ID={tagId}", (x, y + 30), self.font, 2, (255, 255, 255), 2)

    @classmethod
    def resizeImage(cls, img, width: int = 600) -> np.ndarray:
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        pim = Image.fromarray(rgb)

        wPct = (width / float(pim.size[0]))
        hsize = int((float(pim.size[1]) * float(wPct)))

        resized = pim.resize((width, hsize), PIL.Image.ANTIALIAS)
        return cv.cvtColor(np.asarray(resized), cv.COLOR_RGB2BGR)

    def shutdown(self):
        for imgData in self.queuedFrames.values():
            freeShmFromImageData(imgData)

        mainExitEvent.set()
        process.shutdown()

        try:
            process.join(process.maxTimeoutSec)
        except ValueError as e:
            print(e.args.__str__())


def mainExitEventHandler(signum, frame):
    global tester
    tester.shutdown()


signal.signal(signal.SIGINT, mainExitEventHandler)
signal.signal(signal.SIGTERM, mainExitEventHandler)

if __name__ == "__main__":
    fn1, fn2 = "MOV_0368.mp4", "MOV_0375.mp4"
    fn = fn1
    baseName, ext = os.path.splitext(fn)

    video = Config.createAppDataPath("video", "new-videos", fName=fn)
    finalSavePath = Config.createAppDataPath("images", "results", "SDST", baseName, "final")
    rawSavePath = Config.createAppDataPath("images", "results", "SDST", baseName, "raw")

    process.activate(PreStateDetectArgs(video, "test_stream"))

    tester = ValveStateDetectorTest(rawSavePath, finalSavePath)
    tester.run()











import numbers
from typing import Tuple, Callable, Union, List
from src.Backend.DataClasses import ImageData, BBoxData, SharedImage
from src.Backend.Processes.ProcessFuncs import freeAllShmInImageDataQueue
from src.Backend.DataClasses import PreStateDetectData, PreStateDetectArgs, StateDetectArgs
from src.Backend.Processes.StateDetectProcess import StateDetectProcess
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from collections.abc import Iterable
from src.Config import Config, LOG
import time
import os
import signal


class UnifiedStateDetectProcess(mp.Process):
    def __init__(self, mainExitEvent: mp.Event(), maxTimeoutSec=10):
        super().__init__()

        self.maxTimeoutSec = maxTimeoutSec

        data = Config.getModelPaths()
        self.cfg = data.valveCfgSrc
        self.weights = data.valveWeightsSrc

        self.frontPipe, self.backPipe = mp.Pipe()
        self.comsQueue = mp.Queue()
        self.mainExitEvent = mainExitEvent

        self.resultQueue = mp.Queue()

    # Backend

    def run(self) -> None:
        import cv2 as cv

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        net = cv.dnn_DetectionModel(self.cfg, self.weights)
        net.setInputSize(416, 416)
        net.setInputScale((1.0 / 255))
        net.setInputSwapRB(True)

        data = PreStateDetectData()
        data.valveModel = net

        while data.mainActive:
            self.listenFrontend__(data)

    def listenFrontend__(self, data: PreStateDetectData):
        msg = self.backPipe.recv() if self.backPipe.poll() else ""

        # print(f"PreStateDetect: msg{msg}")

        if (msg == "shutdown") or self.mainExitEvent.is_set() or data.shutdownFlag:
            self.shutdown__(data)

        elif data.finishedFlag and data.dfsActive:
            data.dfsActive = False
            data.finishedFlag = False
            self.backPipe.send("finished")

        elif msg == "deactivate":
            data.dfsActive = False
            data.activateFlag = False
            self.backPipe.send("deactivated")

        elif msg == "flush":
            self.flush__(data)
            self.backPipe.send("flushed")

        elif (msg == "activate") and not data.dfsActive or (data.activateFlag and data.dfsActive is False):
            args: PreStateDetectArgs = self.comsQueue.get()

            if type(args) == PreStateDetectArgs:
                data.args = args
                data.dfsActive = True
                data.activateFlag = False
                self.backPipe.send("activated")
                self.detectFromStream__(data)
            else:
                data.activateFlag = True

        elif not data.dfsActive and not data.activateFlag:
            time.sleep(0.3)

    def flush__(self, data: PreStateDetectData):
        freeAllShmInImageDataQueue(self.resultQueue)

    def shutdown__(self, data: PreStateDetectData):
        if data.mainActive:
            data.dfsActive = False
            data.mainActive = False
            print(f"UnifiedStateDetectProcess: Shutting down QueueSize={self.resultQueue.qsize()}")
            freeAllShmInImageDataQueue(self.resultQueue)
            print(f"UnifiedStateDetectProcess: shm freed QueueSize={self.resultQueue.qsize()}")

    def detectFromStream__(self, data: PreStateDetectData):
        import numpy as np
        import cv2 as cv
        from src.Backend.IDMethods.TagID import assignTagsToBBoxes
        from src.Backend.StateMethods import sift

        cap = cv.VideoCapture(data.args.streamPath)

        while data.dfsActive:
            _, frame = cap.read()

            if frame is None:
                data.finishedFlag = True
                break

            classes, confidences, boxes = data.valveModel.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
            bboxes = []
            tagIdx = []

            for i, (classID, confidence, box) in enumerate(zip(classes, confidences, boxes)):
                # tag class
                if (classID == 7) and (confidence > data.args.confidTagThresh):
                    tagIdx.append(i)
                elif confidence > data.args.confidValveThresh:
                    bboxes.append(BBoxData(classID, box))

            if len(bboxes) > 0:

                tagBoxes = [boxes[i] for i in tagIdx]

                if len(tagBoxes) > 0:
                    assignTagsToBBoxes(tagBoxes, bboxes)

                    # Create a shared memory buffer for image for accessing across Processes
                    shm = SharedMemory(create=True, size=frame.nbytes)
                    copy = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
                    copy[:] = frame[:]
                    # ----------------------------------------------------------------------

                    imgData = ImageData(data.args.streamID, SharedImage(shm.name, copy.dtype, copy.shape), bboxes)

                    #for boxData in imgData.bboxes:
                    #    retType, state = sift(frame, boxData)
                    #    boxData.valveState = state

                    shm.close()
                    self.resultQueue.put(imgData)

            self.listenFrontend__(data)
        cap.release()

    # Frontend

    def activate(self, data: PreStateDetectArgs):
        if type(data) != PreStateDetectArgs:
            LOG(f"data arg is of type {type(data)}, not PreStateDetectArgs")
            return
        elif not os.path.exists(data.streamPath):
            LOG(f"streamPath doesn't exist")

        self.frontPipe.send("activate")
        self.comsQueue.put(data)

    def flush(self):
        self.frontPipe.send("flush")

    def deactivate(self):
        self.frontPipe.send("deactivate")

    def sendShutdownSignal(self):
        self.frontPipe.send("shutdown")

    def shutdown(self):
        self.frontPipe.send("shutdown")
        self.join(self.maxTimeoutSec)
        self.close()

    def getPipe(self):
        return self.frontPipe

    def getPostStateDetectQueue(self):
        return self.resultQueue


if __name__ == "__main__":
    pass



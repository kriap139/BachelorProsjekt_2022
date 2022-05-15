import numbers
import time
from typing import Tuple, Callable, Union, List
from src.Backend.DataClasses import ImageData, BBoxData, SharedImage
import multiprocessing as mp
from src.Backend.Processes.ProcessFuncs import freeAllShmInImageDataQueue, freeShmFromImageData
from multiprocessing.shared_memory import SharedMemory
from collections.abc import Iterable
from src.Backend.DataClasses import StateDetectData
from src.Backend.Logging import LOG
from queue import Empty, Full
import os
import signal


class StateDetectProcess(mp.Process):
    def __init__(self, mainExitEvent: mp.Event, preSDQueue: mp.Queue, postSDQueue: mp.Queue, maxTimeoutSec=10):
        super().__init__()

        self.maxTimeoutSec = maxTimeoutSec
        self.mainExitEvent = mainExitEvent
        self.comsQueue = mp.Queue()
        self.preSDQueue = preSDQueue
        self.postSDQueue = postSDQueue
        self.frontPipe, self.backPipe = mp.Pipe()

    def run(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        data = StateDetectData()
        while data.mainActive:
            self.listenFrontend__(data)

    def listenFrontend__(self, data: StateDetectData):
        msg = self.backPipe.recv() if self.backPipe.poll() else ""

        if (msg == "shutdown") or self.mainExitEvent.is_set():
            data.mainActive = False
            data.stActive = False
            self.flush__(data)

        elif data.finishedFlag and data.stActive:
            data.finishedFlag = False
            data.stActive = False
            self.backPipe.send("finished")

        elif msg == "deactivate":
            data.stActive = False
            data.activateFlag = False
            self.backPipe.send("deactivated")

        elif msg == "flush":
            self.flush__(data)
            self.backPipe.send("flushed")

        elif (msg == "activate") and not data.stActive:
            self.backPipe.send("activated")
            data.stActive = True
            self.stateDetection__(data)

    def flush__(self, data: StateDetectData):
        freeAllShmInImageDataQueue(self.postSDQueue)

    def stateDetection__(self, data: StateDetectData):
        from src.Backend.StateMethods import SIFTImageHandler, sift
        import numpy as np
        import cv2 as cv

        while data.stActive:
            try:
                imgData: ImageData = self.preSDQueue.get(block=False)

                if imgData is None:
                    self.postSDQueue.put(None)
                    data.finishedFlag = True
                    break

                sharedImg = imgData.sharedImg
                shm = SharedMemory(name=sharedImg.memName)
                img = np.ndarray(shape=sharedImg.shape, dtype=sharedImg.dType, buffer=shm.buf)

                for boxData in imgData.bboxes:
                    retType, state = sift(img, boxData)
                    boxData.valveState = state

                shm.close()

                #print(imgData)

                self.postSDQueue.put(imgData)
                self.listenFrontend__(data)
            except Empty as e:
                time.sleep(0.3)
                self.listenFrontend__(data)

    # Frontend

    def activate(self):
        self.frontPipe.send("activate")

    def deactivate(self):
        self.frontPipe.send("deactivate")

    def flush(self):
        self.frontPipe.send("flush")

    def sendShutdownSignal(self):
        self.frontPipe.send("shutdown")

    def shutdown(self):
        self.frontPipe.send("shutdown")
        self.join(self.maxTimeoutSec)
        self.close()

    def getPipe(self):
        return self.frontPipe

    def getPostSDQueue(self) -> mp.Queue:
        return self.postSDQueue


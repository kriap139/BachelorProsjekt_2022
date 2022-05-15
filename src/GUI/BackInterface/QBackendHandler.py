from src.Backend.Processes import (
    PreStateDetectProcess, freeShmFromImageData, freeShm, PreStateDetectArgs, UnifiedStateDetectProcess
)
from src.Backend.DataClasses import ImageData, SharedImage, ValveStateData, BBoxData
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import select
import os
from dataclasses import dataclass, field
import uuid
import platform
import time
from typing import Dict, Tuple, List
from queue import Empty

mainExitEvent = mp.Event()

# Create All processes first, before any threads (or packages that use threads) are activated!
NUM_PROCS = 1  # Could be set depending on available cpu cores
BACKEND_PROCS = []

for i in range(NUM_PROCS):
    prc = PreStateDetectProcess(mainExitEvent)
    prc.start()
    BACKEND_PROCS.append(prc)
# ! -------------------------------------------------------------------------------------------


from src.Config import Config
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, QMutex
from PyQt6.QtGui import QImage
import numpy as np
import cv2 as cv
from src.Backend.Logging import LOG
from src.GUI.QUtils import toQImage
from src.GUI.ValveGUI import ValveGUI


@dataclass
class ProcessWrapper:
    streamID: str
    process: UnifiedStateDetectProcess
    args: PreStateDetectArgs
    timeAdded: int = time.time()


class QBackendHandler(QThread):
    TAG_BOX_COLOR = (155, 103, 60)

    # Signals
    updateValveStates = pyqtSignal(list)
    updateImageFeed = pyqtSignal(QImage)
    # !Signal

    def __init__(self, parent=None, maxTimeoutSec=10):
        super(QBackendHandler, self).__init__(parent)

        data = Config.getModelPaths()
        self.valveWeightsSrc = data.valveWeightsSrc
        self.valveCfgSrc = data.valveCfgSrc
        self.maxTimeoutSec = maxTimeoutSec
        self.currStreamID = None

        self.frontPipe, self.backPipe = mp.Pipe()
        self.procWrapperByPipe: Dict[mp.Pipe, ProcessWrapper] = {}
        self.coms = []
        self.comsLock = QMutex()

    def run(self) -> None:
        active = True

        while active:
            rlist = [pipe for pipe, proc in self.procWrapperByPipe.items()]
            rlist.append(self.backPipe)

            r = mp.connection.wait(rlist, timeout=1)

            #print(f"QBackendHandler: run method active={active}")

            if len(r) > 0:
                for pipe in r:
                    if pipe == self.backPipe:
                        msg = pipe.recv()

                        #print(f"QBackendHandler: own pipe msg={msg}")

                        if msg == "com":
                            self.comsLock.lock()
                            comStr, obj = self.coms.pop()

                            if comStr == "addProcess":
                                self.addProcess__(obj)
                            self.comsLock.unlock()

                        elif msg == "shutdown":
                            self.shutdown__()
                            active = False
                            continue
                    else:
                        pw: ProcessWrapper = self.procWrapperByPipe[pipe]
                        msg = pipe.recv()

                        #print(f"QBackendHandler: process pipe msg={msg}")

                        if msg == "finished":
                            pw.process.deactivate()
                            self.comsLock.lock()
                            BACKEND_PROCS.append(pw.process)
                            self.comsLock.unlock()

                        elif msg == "deactivated":
                            pass
                        elif msg == "flushed":
                            pass

            self.handleResults__()

    def shutdown__(self):
        trigExitEvent = True
        for pipe, pw in self.procWrapperByPipe.items():
            if trigExitEvent:
                pw.process.mainExitEvent.set()
                trigExitEvent = False

            pw.process.sendShutdownSignal()

        for pipe, pw in self.procWrapperByPipe.items():
            pw.process.join(self.maxTimeoutSec)
            pw.process.shutdown()

    def handleResults__(self):
        self.comsLock.lock()
        streamID = self.currStreamID
        self.comsLock.unlock()

        updatedValveStates: List[ValveStateData] = []
        imgDataForCurrStream: ImageData = None

        for pipe, dw in self.procWrapperByPipe.items():
            queue: mp.Queue = dw.process.getPostStateDetectQueue()

            try:
                imageData = queue.get(block=False)

                for bb in imageData.bboxes:
                    bbox: BBoxData = bb

                    if bbox.valveID is not None:
                        vsd = ValveStateData(bbox.valveID, bbox.valveState, bbox.angle)
                        updatedValveStates.append(vsd)

                if imageData.streamID == streamID:
                    imgDataForCurrStream = imageData
            except ValueError as e:
                pass
            except Empty as e:
                pass

        if len(updatedValveStates) > 0:
            self.updateValveStates.emit(updatedValveStates)

        if imgDataForCurrStream is not None:
            self.updateImageStream__(imgDataForCurrStream)

    def updateImageStream__(self, imageData: ImageData):
        """Draw bboxes and valveInfo on sharedImage to the currently selected stream"""
        sm = imageData.sharedImg

        shm = SharedMemory(name=sm.memName)
        img = np.ndarray(shape=sm.shape, dtype=sm.dType, buffer=shm.buf)

        font = cv.FONT_HERSHEY_PLAIN

        for bbox in imageData.bboxes:
            x, y, w, h = bbox.box

            color = ValveGUI.VALVE_COLORS_BGRA.get(bbox.valveState)
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

        rgbImg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        qim = toQImage(rgbImg)

        self.updateImageFeed.emit(qim)
        freeShm(shm)

    def addProcess__(self, pw: ProcessWrapper):
        """Register process"""
        pw.process.activate(pw.args)
        self.procWrapperByPipe[pw.process.getPipe()] = pw

    # Frontend

    def addProcess(self, pw: ProcessWrapper):
        """tell Thread to activate process"""

        self.comsLock.lock()
        self.coms.append(
            ("addProcess", pw)
        )
        self.comsLock.unlock()

        self.frontPipe.send("com")

    @pyqtSlot(str)
    def addStream(self, streamPath: str):
        if len(BACKEND_PROCS) == 0:
            # Here one could deactivate running streams if necessary or add path to a stream Queue etc...
            LOG(f"All StreamProcesses are busy")
        elif not os.path.exists(streamPath):
            LOG(f"StreamPath doesn't exist")
        else:
            self.comsLock.lock()
            prc = BACKEND_PROCS.pop()
            self.comsLock.unlock()

            args = PreStateDetectArgs(streamPath, self.createStreamID(streamPath))
            pw = ProcessWrapper(args.streamID, prc, args)

            if self.currStreamID is None:
                self.currStreamID = pw.streamID
            self.addProcess(pw)

    @pyqtSlot(str)
    def setCurrentImageFeed(self, streamID: str):
        with self.comsLock:
            self.currStreamID = streamID

    def shutdown(self):
        self.frontPipe.send("shutdown")
        self.wait()
        self.exit()

    @classmethod
    def createStreamID(cls, streamPath) -> str:
        return f"{os.path.basename(streamPath)}_{uuid.uuid1().hex}"

















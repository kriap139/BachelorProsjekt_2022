from src.Backend.DataClasses import ImageData, BBoxData, SharedImage
from src.Backend.Processes.ProcessFuncs import freeAllShmInImageDataQueue
from src.Backend.DataClasses import PreStateDetectData, PreStateDetectArgs
from src.Backend.Processes.StateDetectProcess import StateDetectProcess
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from src.Config import Config, LOG
import time
import os
import signal
from typing import List
from queue import Empty


class PreStateDetectProcess(mp.Process):
    def __init__(self, mainExitEvent: mp.Event(), maxTimeoutSec=10, numSDProcs: int = 2):
        super().__init__()

        self.maxTimeoutSec = maxTimeoutSec

        data = Config.getModelPaths()
        self.cfg = data.valveCfgSrc
        self.weights = data.valveWeightsSrc
        self.tagModelPath = data.tagIdCNNPath

        self.frontPipe, self.backPipe = mp.Pipe()
        self.comsQueue = mp.Queue()
        self.mainExitEvent = mainExitEvent

        self.preSDetectQueue = mp.Queue()
        self.postSDetectQueue = mp.Queue()

        self.numSDProcs = numSDProcs
        self.stateDProcs: List[StateDetectProcess] = None

    # Backend

    def run(self) -> None:
        import cv2 as cv
        from src.Backend.IDMethods import createCNNModel

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.stateDProcs = []
        for _ in range(self.numSDProcs):
            sdp = StateDetectProcess(self.mainExitEvent, self.preSDetectQueue, self.postSDetectQueue, self.maxTimeoutSec)
            sdp.start()
            sdp.activate()
            self.stateDProcs.append(sdp)

        tagModel = createCNNModel(self.tagModelPath)

        net = cv.dnn.readNet(self.weights, self.cfg)
        layerNames = net.getLayerNames()
        outputLayers = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

        data = PreStateDetectData()
        data.valveModel = net
        data.vmOutputLayers = outputLayers
        data.tagIdModel = tagModel
        data.finishedSDFlags = [False for _ in range(self.numSDProcs)]

        while data.mainActive:
            self.listenFrontend__(data)

    def listenFrontend__(self, data: PreStateDetectData):
        msg = self.backPipe.recv() if self.backPipe.poll() else ""

        # print(f"PreStateDetect: msg{msg}")

        if (msg == "shutdown") or self.mainExitEvent.is_set() or data.shutdownFlag:
            self.shutdown__(data)

        elif data.finishedFlag:
            if data.dfsActive:
                data.dfsActive = False

            for i, sdp in enumerate(self.stateDProcs):
                msgSDP = sdp.getPipe().recv() if sdp.getPipe().poll() else ""
                if msgSDP == "finished":
                    data.finishedSDFlags[i] = True

            # print(f"PreStateDetect: finishedFlags={data.finishedSDFlags}")

            if all(data.finishedSDFlags):
                print("PreStateDetectFinished: Finished")
                self.backPipe.send("finished")
                data.finishedFlag = False
                data.finishedSDFlags = [False for _ in data.finishedSDFlags]
            elif not data.sdActive:
                self.stateDetection__(data)

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

    def stateDetection__(self, data: PreStateDetectData):
        from src.Backend.StateMethods import SIFTImageHandler, SiftStateDetector
        import numpy as np

        while data.sdActive:
            try:
                imgData: ImageData = self.postSDetectQueue.get(block=False)

                if imgData is None:
                    data.sdActive = False
                    continue

                sharedImg = imgData.sharedImg
                shm = SharedMemory(name=sharedImg.memName)
                img = np.ndarray(shape=sharedImg.shape, dtype=sharedImg.dType, buffer=shm.buf)

                for boxData in imgData.bboxes:
                    boxData.valveState = SiftStateDetector.sift(img, boxData)

                shm.close()
                self.postSDetectQueue.put(imgData)
                self.listenFrontend__(data)
            except Empty as e:
                data.sdActive = False

    def flush__(self, data: PreStateDetectData):
        for sdp in self.stateDProcs:
            sdp.flush()

        freeAllShmInImageDataQueue(self.preSDetectQueue)

    def shutdown__(self, data: PreStateDetectData):
        if data.dfsActive:
            data.dfsActive = False
            for sdp in self.stateDProcs:
                sdp.sendShutdownSignal()

        if data.mainActive:
            data.mainActive = False
            print("PreStateDetect: Shutting down")
            freeAllShmInImageDataQueue(self.preSDetectQueue)

            for sdp in self.stateDProcs:
                sdp.shutdown()

    def activateSDProcs__(self):
        for proc in self.stateDProcs:
            proc.activate()

    def deactivateSDProcs__(self):
        for proc in self.stateDProcs:
            proc.deactivate()

    def finishSDProcsWhenEmpty__(self):
        for proc in self.stateDProcs:
            proc.finishIfEmpty()

    def detectFromStream__(self, data: PreStateDetectData):
        import numpy as np
        import cv2 as cv
        from src.Backend.IDMethods import identifyTags, assignTagsToBBoxes

        cap = cv.VideoCapture(data.args.streamPath)
        tagClassID = data.args.tagClassID
        frameID = 0

        while data.dfsActive:
            _, frame = cap.read()

            if frame is None:
                self.finishSDProcsWhenEmpty__()
                data.finishedFlag = True
                break

            frameID += 1

            height, width, channels = frame.shape
            blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            data.valveModel.setInput(blob)
            outs = data.valveModel.forward(data.vmOutputLayers)

            classes = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.2:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append((x, y, w, h))
                        confidences.append(float(confidence))
                        classes.append(class_id)

            indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

            tags = []
            bboxes = []

            for i, box in enumerate(boxes):
                if i in indexes:
                    if classes[i] == tagClassID:
                        tags.append(box)
                    else:
                        bboxes.append(BBoxData(classes[i], box))

            tagsData = identifyTags(data.tagIdModel, frame, tags)
            uIdx = assignTagsToBBoxes(tagsData, bboxes)

            if len(bboxes) > 0:
                # Create a shared memory buffer for image for accessing across Processes
                shm = SharedMemory(create=True, size=frame.nbytes)
                copy = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
                copy[:] = frame[:]
                shm.close()
                # ----------------------------------------------------------------------

                imgData = ImageData(data.args.streamID, frameID, SharedImage(shm.name, copy.dtype, copy.shape), bboxes,
                                    tagsData, uIdx)

                self.preSDetectQueue.put(imgData)
            self.listenFrontend__(data)

        cap.release()
        self.listenFrontend__(data)

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
        return self.postSDetectQueue


if __name__ == "__main__":
    pass

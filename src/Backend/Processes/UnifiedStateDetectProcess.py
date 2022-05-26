from src.Backend.DataClasses import ImageData, BBoxData, SharedImage
from src.Backend.Processes.ProcessFuncs import freeAllShmInImageDataQueue
from src.Backend.DataClasses import PreStateDetectData, PreStateDetectArgs
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
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
        self.tagModelPath = data.tagIdCNNPath

        self.frontPipe, self.backPipe = mp.Pipe()
        self.comsQueue = mp.Queue()
        self.mainExitEvent = mainExitEvent

        self.resultQueue = mp.Queue()

    # Backend

    def run(self) -> None:
        import cv2 as cv
        from src.Backend.IDMethods import createCNNModel
        from src.Backend.StateMethods import ColorStateDetector

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        tagModel = createCNNModel(self.tagModelPath)

        net = cv.dnn.readNet(self.weights, self.cfg)
        layerNames = net.getLayerNames()
        outputLayers = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

        data = PreStateDetectData()
        data.valveModel = net
        data.vmOutputLayers = outputLayers
        data.tagIdModel = tagModel

        self.colorDetector = ColorStateDetector()

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
        from src.Backend.IDMethods import assignTagsToBBoxes, identifyTags
        from src.Backend.StateMethods import SiftStateDetector, colorDetection

        cap = cv.VideoCapture(data.args.streamPath)
        tagClassID = data.args.tagClassID
        frameID = 1

        while data.dfsActive:
            _, frame = cap.read()

            if frame is None:
                data.finishedFlag = True
                break

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
                for bbd in bboxes:
                    bbd.valveState = SiftStateDetector.sift(frame, bbd)

                # Create a shared memory buffer for image for accessing across Processes
                shm = SharedMemory(create=True, size=frame.nbytes)
                copy = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
                copy[:] = frame[:]
                # ----------------------------------------------------------------------

                imgData = ImageData(data.args.streamID, frameID, SharedImage(shm.name, copy.dtype, copy.shape), bboxes,
                                    tagsData, uIdx)

                shm.close()
                self.resultQueue.put(imgData)
                frameID += 1

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



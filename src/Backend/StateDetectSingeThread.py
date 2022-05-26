import cv2 as cv
from src.Config import Config
from src.Backend.DataClasses import BBoxData, ImageData, SharedImage
import numpy as np
from src.Backend.Valve import ValveState
from src.Backend.StateMethods import SiftStateDetector, SIFTImageHandler
from src.Backend.IDMethods import createCNNModel, assignTagsToBBoxes, identifyTags
from src.Backend.StateMethods import ColorStateDetector
from typing import Iterable, Callable, Tuple
import PIL
from PIL import Image
import os
import time


class StateDetectSingleThread:
    TAG_BOX_COLOR = (155, 103, 60)

    VALVE_COLORS_BGRA = {
        ValveState.UNKNOWN: [2, 210, 238, 1],
        ValveState.CLOSED: [10, 34, 224, 1],
        ValveState.OPEN: [0, 230, 17, 1]
    }

    def __init__(self, tagClassID: int, frameSavePath: str, finalFrameSavePath: str, tagIDLength: int = 5):
        data = Config.getModelPaths()
        self.cfg = data.valveCfgSrc
        self.weights = data.valveWeightsSrc
        self.tagModelPath = data.tagIdCNNPath

        self.tagClassID = tagClassID
        self.tagIDLength = tagIDLength
        self.tagModel = createCNNModel(self.tagModelPath)
        self.font = cv.FONT_HERSHEY_PLAIN
        self.frameSavePath = frameSavePath
        self.finalFrameSavePath = finalFrameSavePath

        self.colorSDetect = ColorStateDetector()

        self.net = cv.dnn.readNet(self.weights, self.cfg)
        self.layerNames = self.net.getLayerNames()
        self.outputLayers = [self.layerNames[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detectFromStream(self, streamPath: str, saveFrames: bool = True, method: str = "sift"):
        cap = cv.VideoCapture(streamPath)
        frameID = 0

        if method == "sift":
            SIFTImageHandler.loadAllSiftRefImages()
            stateMethod = SiftStateDetector.sift
        elif method == "colorSearch":
            stateMethod = self.colorSDetect.stateDetect
        else:
            print("invalid Method")
            return

        start = time.time()

        while True:
            _, frame = cap.read()

            if frame is None:
                break

            frameID += 1

            if saveFrames:
                cv.imwrite(os.path.join(self.frameSavePath, f"{frameID}.jpg"), frame)

            height, width, channels = frame.shape
            blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            self.net.setInput(blob)
            outs = self.net.forward(self.outputLayers)

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
                    if classes[i] == self.tagClassID:
                        tags.append(box)
                    else:
                        bboxes.append(BBoxData(classes[i], box))

            tagDatas = identifyTags(self.tagModel, frame, tags, self.tagIDLength)
            uIdx = assignTagsToBBoxes(tagDatas, bboxes)

            for bbd in bboxes:
                if bbd.tagBox is not None:
                    state = stateMethod(frame, bbd)
                    bbd.valveState = state

            # Drawing -----------------------------------------------------------------
            unassigned = (tagDatas[i] for i in uIdx)
            for td in unassigned:
                self.drawTag(frame, td.tagBox, td.tagID, color=(0, 0, 0))

            self.draw(frame, bboxes)

            for j, td in enumerate(tagDatas):
                if j not in uIdx:
                    cv.line(frame, td.tagLine[0].toTuple(), td.tagLine[1].toTuple(), (0, 255, 0), 3)

            resized = self.resizeImage(frame, width=600)
            cv.putText(resized, f"Frame {frameID - 1}", (3, 30), self.font, 2, color=(0, 255, 0), thickness=2)
            # -------------------------------------------------------------------------

            if saveFrames:
                cv.imwrite(os.path.join(self.finalFrameSavePath, f"{frameID}.jpg"), resized)

            cv.imshow("result", resized)
            if cv.waitKey(1) == ord('q'):
                break

        endTime = time.time() - start
        fps = frameID / endTime

        print(f"time={round(endTime, 3)}, FPS={round(fps, 3)}, TotalFrames={frameID}")

        cap.release()
        cv.destroyAllWindows()

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

        resized = pim.resize((width, hsize), PIL.Image.LANCZOS)
        return cv.cvtColor(np.asarray(resized), cv.COLOR_RGB2BGR)


if __name__ == "__main__":
    fn1, fn2 = "MOV_0368.mp4", "MOV_0375.mp4"
    fn = fn1
    baseName, ext = os.path.splitext(fn)

    method = "sift"  # "colorSearch"

    video = Config.createAppDataPath("video", "new-videos", fName=fn)
    finalSavePath = Config.createAppDataPath("images", "results", "SDST", method, baseName, "final")
    rawSavePath = Config.createAppDataPath("images", "results", "SDST", method, baseName, "raw")

    tester = StateDetectSingleThread(tagClassID=8, frameSavePath=rawSavePath, finalFrameSavePath=finalSavePath)
    tester.detectFromStream(video, saveFrames=False, method=method)

    # new colordetect single thread -> time=195.282, FPS=2.381, TotalFrames=465
    # new sift single thread -> time=218.978, FPS=2.124, TotalFrames=465
    # new sift split process (3 state detect procs) -> time=185.529, FPS=2.512, TotalFrames=465


import cv2 as cv
from src.Config import Config
from src.Backend.DataClasses import BBoxData, ImageData, SharedImage
import numpy as np
from src.Backend.Valve import ValveState
from src.Backend.StateMethods import SiftStateDetector
from src.Backend.IDMethods import assignTagsToBBoxes, detectTagID, createCNNModel
from typing import Iterable
import PIL
from PIL import Image
import os

class StateDetectSingleThread:
    TAG_BOX_COLOR = (155, 103, 60)

    VALVE_COLORS_BGRA = {
        ValveState.UNKNOWN: [2, 210, 238, 1],
        ValveState.CLOSED: [10, 34, 224, 1],
        ValveState.OPEN: [0, 230, 17, 1]
    }

    def __init__(self):
        data = Config.getModelPaths()

        self.cfg = data.valveCfgSrc
        self.weights = data.valveWeightsSrc

        self.net = cv.dnn_DetectionModel(self.cfg, self.weights)
        self.net.setInputSize(416, 416)
        self.net.setInputScale((1.0 / 255))
        self.net.setInputSwapRB(True)

    def detectFromStream(self, streamPath: str, confidThresh: float = 0.5):
        cap = cv.VideoCapture(streamPath)

        while True:
            _, frame = cap.read()

            if frame is None:
                break

            classes, confidences, boxes = self.net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
            bboxes = []
            tagIdx = []

            for i, (classID, confidence, box) in enumerate(zip(classes, confidences, boxes)):
                # tag class
                if (classID == 7) and (confidence > 0.5):
                    tagIdx.append(i)
                elif confidence > confidThresh:
                    bboxes.append(BBoxData(classID, box))

            if len(bboxes) > 0:

                tagBoxes = [boxes[i] for i in tagIdx]

                if len(tagBoxes) > 0:
                    imgData = ImageData("stream", None, bboxes)

                    #for boxData in imgData.bboxes:
                    #    retType, state = sift(frame, boxData)
                    #    boxData.valveState = state

                    self.draw(frame, imgData)
                    cv.imshow("result", frame)

                    if cv.waitKey(1) == ord('q'):
                        break
        cap.release()

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


class StateDetectSingleThread2:
    TAG_BOX_COLOR = (155, 103, 60)

    VALVE_COLORS_BGRA = {
        ValveState.UNKNOWN: [2, 210, 238, 1],
        ValveState.CLOSED: [10, 34, 224, 1],
        ValveState.OPEN: [0, 230, 17, 1]
    }

    def __init__(self):
        data = Config.getModelPaths()
        self.cfg = data.valveCfgSrc
        self.weights = data.valveWeightsSrc
        self.font = cv.FONT_HERSHEY_PLAIN

        self.net = cv.dnn.readNet(self.weights, self.cfg)
        self.layerNames = self.net.getLayerNames()
        self.outputLayers = [self.layerNames[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detectFromStream(self, streamPath: str, confidThresh: float = 0.5):
        cap = cv.VideoCapture(streamPath)

        frameID = 0

        while True:
            _, frame = cap.read()

            if frame is None:
                break

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
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        classes.append(class_id)

            indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

            for i, box in enumerate(boxes):
                if i in indexes:
                    x, y, w, h = box

                    if classes[i] == 7:
                        label = "Tag"
                        color = self.TAG_BOX_COLOR
                        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
                        cv.putText(frame, label, (x, y + 30), self.font, 2, (255, 255, 255), 2)
                    else:
                        boxData = BBoxData(classes[i], box)

                        retType, state = (frame, boxData)
                        boxData.valveState = state

                        color = self.VALVE_COLORS_BGRA.get(boxData.valveState)
                        label = f"id={boxData.valveID}, cls={classes[i]}"

                        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
                        cv.putText(frame, label, (x, y + 30), self.font, 2, (255, 255, 255), 3)

            cv.imshow("result", frame)
            if cv.waitKey(1) == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()


class StateDetectSingleThread3:
    TAG_BOX_COLOR = (155, 103, 60)

    VALVE_COLORS_BGRA = {
        ValveState.UNKNOWN: [2, 210, 238, 1],
        ValveState.CLOSED: [10, 34, 224, 1],
        ValveState.OPEN: [0, 230, 17, 1]
    }

    def __init__(self, tagClassID: int, frameSavePath: str, finalFrameSavePath: str):
        data = Config.getModelPaths()

        self.cfg = data.valveCfgSrc
        self.weights = data.valveWeightsSrc
        self.tagModelPath = data.tagIdCNNPath
        self.tagClassID = tagClassID
        self.font = cv.FONT_HERSHEY_PLAIN
        self.frameSavePath = frameSavePath
        self.finalFrameSavePath = finalFrameSavePath

        self.tagModel = createCNNModel(self.tagModelPath)
        self.net = cv.dnn.readNet(self.weights, self.cfg)
        self.layerNames = self.net.getLayerNames()
        self.outputLayers = [self.layerNames[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detectFromStream(self, streamPath: str):
        cap = cv.VideoCapture(streamPath)
        frameID = 1

        while True:
            _, frame = cap.read()

            if frame is None:
                break

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

                        boxes.append([x, y, w, h])
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

            assignTagsToBBoxes(tags, bboxes)

            for bbd in bboxes:
            #    if bbd.tagBox is not None:
            #        bbd.valveID = detectTagID(self.tagModel, frame, bbd.tagBox)
                _, state = SiftStateDetector.sift(frame, bbd)
                bbd.valveState = state


            cv.imwrite(os.path.join(self.frameSavePath, f"{frameID}.jpg"), frame)

            for tag in tags:
                tagId = detectTagID(self.tagModel, frame, tag)
                self.drawTag(frame, tag, tagId)

            self.draw(frame, bboxes)
            resized = self.resizeImage(frame, width=600)

            cv.imwrite(os.path.join(self.finalFrameSavePath, f"{frameID}.jpg"), resized)
            cv.imshow("result", resized)

            if cv.waitKey(1) == ord('q'):
                break
            frameID += 1

        cap.release()
        cv.destroyAllWindows()

    def draw(self, img: np.ndarray, bboxes: Iterable[BBoxData]):
        for bbox in bboxes:
            x, y, w, h = bbox.box

            color = self.VALVE_COLORS_BGRA.get(bbox.valveState)
            label = f"id={bbox.valveID}, cls={bbox.classID}"

            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.rectangle(img, (x, y), (x + w, y + 30), color, -1)
            cv.putText(img, label, (x, y + 30), self.font, 2, (255, 255, 255), 3)

            tagBox = bbox.tagBox

            if tagBox is not None:
                label = "Tag"
                color = self.TAG_BOX_COLOR
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv.rectangle(img, (x, y), (x + w, y + 30), color, -1)
                cv.putText(img, label, (x, y + 30), self.font, 2, (255, 255, 255), 2)

    def drawTag(self, img: np.ndarray, tagBox: tuple, tagId: str):
        x, y, w, h = tagBox
        label = f"TagID={tagId}"
        color = self.TAG_BOX_COLOR

        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.rectangle(img, (x, y), (x + w, y + 30), color, -1)
        cv.putText(img, label, (x, y + 30), self.font, 2, (255, 255, 255), 2)

    @classmethod
    def resizeImage(cls, img, width: int = 600) -> np.ndarray:
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        pim = Image.fromarray(rgb)

        wPct = (width / float(pim.size[0]))
        hsize = int((float(pim.size[1]) * float(wPct)))

        resized = pim.resize((width, hsize), PIL.Image.ANTIALIAS)
        return cv.cvtColor(np.asarray(resized), cv.COLOR_RGB2BGR)


if __name__ == "__main__":
    # MOV_0375.mp4, MOV_0368.mp4
    videoName = "MOV_0375.mp4"
    baseName, ext = os.path.splitext(videoName)

    video = Config.createAppDataPath("video", "new-videos", fName=videoName)
    finalSavePath = Config.createAppDataPath("images", "results", "SDST", baseName, "final")
    rawSavePath = Config.createAppDataPath("images", "results", "SDST", baseName, "raw")

    tester = StateDetectSingleThread3(tagClassID=8, frameSavePath=rawSavePath, finalFrameSavePath=finalSavePath)

    tester.detectFromStream(video)


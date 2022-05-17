import cv2 as cv
from src.Config import Config
from src.Backend.DataClasses import BBoxData, ImageData, SharedImage
import numpy as np
from src.Backend.Valve import ValveState
from src.Backend.StateMethods import SiftStateDetector
from src.Backend.IDMethods import assignTagsToBBoxes, detectTagID, createCNNModel
from typing import Iterable, Union
import PIL
from PIL import Image
import os


class AssignBBoxTest:
    TAG_BOX_COLOR = (155, 103, 60)

    VALVE_COLORS_BGRA = {
        ValveState.UNKNOWN: [2, 210, 238, 1],
        ValveState.CLOSED: [10, 34, 224, 1],
        ValveState.OPEN: [0, 230, 17, 1]
    }

    def __init__(self, tagClassID: int, savePath: str):
        data = Config.getModelPaths()

        self.cfg = data.valveCfgSrc
        self.weights = data.valveWeightsSrc
        self.tagModelPath = data.tagIdCNNPath
        self.tagClassID = tagClassID
        self.font = cv.FONT_HERSHEY_PLAIN
        self.savePath = savePath

        self.tagModel = createCNNModel(self.tagModelPath)
        self.net = cv.dnn.readNet(self.weights, self.cfg)
        self.layerNames = self.net.getLayerNames()
        self.outputLayers = [self.layerNames[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detectFromStream(self, p: str):
        if os.path.isfile(p):
            paths = p,
        elif os.path.isdir(p):
            paths = (os.path.join(p, fn) for fn in os.listdir(p))
            paths = tuple(filter(lambda path: os.path.isfile(path), paths))
        else:
            print("invalid input!")
            return

        for path in paths:
            frame = cv.imread(path)
            fn = os.path.basename(path)

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

            tagLines, unassigned = assignTagsToBBoxes(frame, tags, bboxes)

            for bbd in bboxes:
                if bbd.tagBox is not None:
                    bbd.valveID = detectTagID(self.tagModel, frame, bbd.tagBox)

            if len(unassigned):
                print("Unassigned tags: ", len(unassigned))
                for tag in unassigned:
                    tagId = detectTagID(self.tagModel, frame, tag)
                    self.drawTag(frame, tag, tagId)

            for line in tagLines:
                cv.line(frame, line[0].toTuple(), line[1].toTuple(), (0, 255, 0), 3)

            self.draw(frame, bboxes)

            resized = self.resizeImage(frame, width=600)
            cv.imwrite(os.path.join(self.savePath, fn), resized)
            cv.imshow(f"result", resized)
            cv.waitKey(0)

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

    def drawTag(self, img: np.ndarray, tagBox: tuple, tagId: str = None):
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


if __name__ == "__main__":
    fn1, fn2 = "MOV_0368.mp4", "MOV_0375.mp4"
    fn = fn1
    baseName, ext = os.path.splitext(fn)

    imgPath = Config.createAppDataPath("images", "results", "assignBBox", baseName, "raw")
    savePath = Config.createAppDataPath("images", "results", "assignBBox", baseName, "processed")

    tester = AssignBBoxTest(tagClassID=8, savePath=savePath)
    tester.detectFromStream(imgPath)


import cv2
import numpy as np
import numbers
import os
from src.Backend.StateMethods.Methods import TYDisplay


class Classification:
    def __init__(self, cfg: str, names: str, weights: str, inputSize: tuple = None,
                 inputScale: numbers.Number = 1.0/255):

        for path in (cfg, weights, names):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Specified file doesn't exist: {path}")

        self.yoloV4 = cv2.dnn_DetectionModel(cfg, weights)
        self.inputSize = (416, 416) if ((inputSize is None) or (len(inputSize) != 2)) else inputSize
        self.inputScale = inputScale

        self.yoloV4.setInputSize(self.inputSize[0], self.inputSize[1])
        self.yoloV4.setInputScale(inputScale)
        self.yoloV4.setInputSwapRB(True)

        with open(names, 'rt') as f:
            self.names = f.read().rstrip('\n').split('\n')

    def detectFromImg(self, img: np.ndarray, display: TYDisplay):
        height, width, channels = img.shape

        classes, confidences, boxes = self.yoloV4.detect(img, confThreshold=0.1, nmsThreshold=0.4)

        for classID, confidence, box in zip(classes, confidences, boxes):
            pass

    def detectFromStream(self, filePath):
        pass


if __name__ == "__main__":
    pass

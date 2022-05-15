import cv2
import numpy as np
import numbers
import os
from typing import Tuple, Union
import json
import os
from src.Backend.StateMethods.ValveDetection import ValveState


class ClassificationTest:
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

    def detectFromImg(self, img: np.ndarray):
        height, width, channels = img.shape

        classes, confidences, boxes = self.yoloV4.detect(img, confThreshold=0.6, nmsThreshold=0.4)
        return tuple(zip(classes, confidences, boxes))

    def createImageOverviewFile(self, rootDir: str, classes: Tuple[int, ...], states: Tuple[str, ...],
                                savePath: str = None, printData=False, returnAsJson=False) -> Union[dict, str]:

        data = dict(((key, {"OPEN": [], "CLOSED": []}) for key in classes))

        # Finding boxes
        for cls in classes:
            for state in states:

                dirPath = f"{rootDir}/{cls}/{state}/"

                for fn in os.scandir(dirPath):

                    if fn.is_file():

                        img = cv2.imread(fn.path)
                        res = self.detectFromImg(img)

                        for classID, confidence, box in res:
                            if classID == cls:
                                arr: list = data[cls][state]

                                arr.append(
                                    {
                                        "fileName": fn.name,
                                        "confidence": confidence.item(),
                                        "box": box.tolist()
                                    }
                                )
        s = json.dumps(data, indent=3)

        if printData:
            print(s)

        if savePath is not None:
            with open(savePath, mode='w') as f:
                json.dump(data, f, indent=3)

        return s if returnAsJson else data

    def createSiftImages(self, rootDir: str, classes: Tuple[int, ...], states: Tuple[str, ...],
                         saveDir: str = None) -> None:

        data: dict = self.createImageOverviewFile(rootDir, classes, states, returnAsJson=False)

        for classID, info in data.items():
            for state, items in info.items():

                path = os.path.abspath(f"{saveDir}/{classID}/{state}")

                if not os.path.exists(path):
                    os.makedirs(path)

                for item in items:

                    fn = item['fileName']
                    x, y, w, h = item['box']

                    img = cv2.imread(f"{rootDir}/{classID}/{state}/{fn}")
                    new_img = img[y:y+h, x:x+w]

                    path = os.path.abspath(f"{saveDir}/{classID}/{state}/{fn}_cropped")
                    status = cv2.imwrite(path, new_img)

                    if not status:
                        print("Failed to save Image")


if __name__ == "__main__":
    cfgP = f"resources/model/classif/yolov4-ventiler.cfg"
    namesP = f"resources/model/classif/ventiler.names"
    weightsP = f"resources/model/classif/yolov4-ventiler_best.weights"

    cir = ClassificationTest(cfgP, namesP, weightsP)

    #rootDir = "resources/testing/ventil-tilstand/tilstand-test"
    #cir.createImageOverviewFile(rootDir, classes=(0, 3, 6), states=("OPEN", "CLOSED"),
    #                            savePath=f"{rootDir}/ImageOverview.json", printData=True)

    rootDir = "resources/testing/ventil-tilstand/tilstand-test/sift"
    cir.createSiftImages(rootDir, classes=(0, 3, 6), states=("OPEN", "CLOSED"), saveDir=f"{rootDir}/cropped")

    img_p = os.path.join("resources", "testing", "ventil-tilstand", "tilstand-test","6", "CLOSED", "001220.jpg")
    img = cv2.imread(img_p)

    data = cir.detectFromImg(img)
    i = 0

    for cls, con, (x, y, w, h) in data:
        new_img = img[y:y+h, x:x+w]
        #cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=3)
        #vs.display(f"cropped{i}", new_img)
        i += 1

    #vs.display("Hello", img)
    #vs.show()



import src.Backend.StateMethods.Methods as dm
from src.Backend.Valve import ValveClass, ValveState
from src.Backend.Valve import Valve
from typing import Tuple, Dict, List
import numpy as np
import cv2
import json
import os
import time


class ImageData:
    def __init__(self, name: str, img: np.ndarray, bboxes: Tuple[Tuple[int, int, int, int]], confid: Tuple[float, ...],
                 valve: Valve):
        self.name = name
        self.img = img
        self.boxes = bboxes
        self.confid = confid
        self.valve = valve


class ValveStateTest:
    def __init__(self):
        self.window = None #PlotWindow()

    def display(self, title: str, img, cmap=None):
        #fig = plt.figure(figsize=(10, 8))
        #ax = fig.add_subplot(111)
        #ax.imshow(img, cmap=cmap)

        #self.window.addPlot(title, fig)
        if title == "Processed image":
            cv2.imshow("rmgr", img)
            cv2.waitKey(0)

    def show(self):
        self.window.show()

    def methodTest(self,
                   methodName: str,
                   classes: tuple,
                   states: Tuple[str, ...],
                   statesEnum: Tuple[ValveState, ...],
                   imgData: Dict[int, Dict[str, List[ImageData]]],
                   angleOpenThreshold=35) -> Tuple[float, float, Tuple[float, ...]]:

        method = getattr(dm, methodName, None)

        if method is None:
            raise AttributeError(f"Function '{method}' not found in {dm} module")
        if not callable(method):
            raise RuntimeError(f"Module attribute {method} is not callable")

        nums_correct = {cls: 0 for cls in classes}
        nums_wrong = {cls: 0 for cls in classes}

        start = time.time()

        for cls in classes:
            for i, state in enumerate(states):

                datas = imgData[cls][state]

                for imData in datas:
                    for j, bbox in enumerate(imData.boxes):

                        ret_type, v_state = method(imData.img.copy(), bbox, imData.valve, self.display)

                        if ret_type == dm.ReturnType.ANGLE:
                            check = (v_state < angleOpenThreshold) or (np.abs(v_state - 180) < angleOpenThreshold)
                            v_state = ValveState.OPEN if check else ValveState.CLOSED

                        if v_state == statesEnum[i]:
                            nums_correct[cls] += 1
                        else:
                            nums_wrong[cls] += 1

        end = time.time()
        totalTime = end - start

        total = float(sum(nums_correct.values()) + sum(nums_wrong.values()))
        v_total = {cls: float(nums_wrong[cls] + nums_correct[cls]) for cls in classes}

        tar = sum(nums_correct.values()) / max(total, 1)
        arv = tuple(nums_correct[cls] / max(v_total[cls], 1) for cls in classes)

        return totalTime, tar, arv

    def sortBBoxData(self, imgData: dict, savePath=None):
        for cls, info in imgData.items():
            for state, items in info.items():
                d = {}
                for item in items:
                    if item["fileName"] in d:
                        d[item["fileName"]]["confidences"].append(item["confidence"])
                        d[item["fileName"]]["boxes"].append(item["box"])
                    else:
                        d[item["fileName"]] = {
                           "confidences": [item["confidence"]],
                            "boxes": [item["box"]]
                        }
                imgData[cls][state] = d

        if savePath is not None:
            with open(savePath, mode='w') as fp:
                json.dump(imgData, fp, indent=3)

        return imgData

    def loadImages(self, rootDir: str, classes: Tuple[int, ...], states: Tuple[str, ...], valves: Dict[int, Valve],
                   imgData: dict) -> dict:

        res = {cls: {state: [] for state in states} for cls in classes}

        for cls in classes:
            v = valves[cls]
            for state in states:
                fs = imgData[str(cls)][state].keys()

                for f in fs:
                    path = os.path.join(rootDir, str(cls), state, f)
                    img = cv2.imread(path)

                    if img is not None:
                        res[cls][state].append(
                            ImageData(f, img, imgData[str(cls)][state][f]["boxes"],
                                      imgData[str(cls)][state][f]["confidences"], v)
                        )
        return res

    def showEachCroppedImage(self, imgData, classes: tuple,  states: Tuple[str, ...]):
        for cls in classes:
            for i, state in enumerate(states):

                datas = imgData[cls][state]

                for imData in datas:
                    for j, bbox in enumerate(imData.boxes):

                        x, y, w, h = bbox
                        #crop = imData.img[y:y + h, x:x + w]
                        img = imData.img.copy()

                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.imshow("img", img)
                        cv2.waitKey(0)


if __name__ == "__main__":
    cfgP = f"resources/model/classif/yolov4-ventiler.cfg"
    namesP = f"resources/model/classif/ventiler.names"
    weightsP = f"resources/model/classif/yolov4-ventiler_best.weights"

    rootDir = "resources/testing/ventil-tilstand/tilstand-test"
    classes = (0, 3, 6)
    states = ("OPEN", "CLOSED")
    statesEnum = (ValveState.OPEN, ValveState.CLOSED)

    # 0: l:(20, 100, 100), u: (30, 255, 255); l:(26, 106, 187), u:(35, 255, 255); l:(22, 46, 74), u:(83, 255, 216)
    # 3: (41,18,10), (21, 48, 23)
    valves = {0: Valve("", ValveClass(0, "", "", colorLower=(20, 100, 100), colorUpper=(30, 255, 255))),
              3: Valve("", ValveClass(3, "", "", colorLower=(0, 83, 25), colorUpper=(12, 255, 111))),
              6: Valve("", ValveClass(6, "", "", colorLower=(176, 123, 21), colorUpper=(180, 232, 221)))}

    vs = ValveStateTest()

    with open(os.path.join(rootDir, "ImageOverviewSorted.json"), mode='r') as f:
        data = json.load(f)

    imgData = vs.loadImages(rootDir, classes, states, valves, data)

    imgDataMarked = vs.loadImages(os.path.join(rootDir, "color-marking", "marked"),
                                  classes, states, valves, data)

    tests = (
        ("watershedVec", 26),
    )

    # ("sift", None) ("watershedVec", 26)

    if True:
        for methodName, angleTD in tests:
            time_sec, tar, (arv0, arv3, arv6) = vs.methodTest(methodName, classes, states, statesEnum, imgData,
                                                              angleOpenThreshold=angleTD)

            print(f"{methodName}: Time(sec)={round(time_sec, 3)},  TAR={round(tar, 3)}, "
                  f"arv0={round(arv0, 3)}, arv3={round(arv3, 3)}, arv6={round(arv6, 3)},"
                  f" angleOpenThreshold={angleTD}")

        #vs.show()
    elif False:
        vs.showEachCroppedImage(imgData, classes, states)
    elif False:
        methodName, angleTD = "pipeMarking", 26

        time_sec, tar, (arv0, arv3, arv6) = vs.methodTest(methodName, classes, states, statesEnum, imgDataMarked,
                                                          angleOpenThreshold=angleTD)

        print(f"{methodName}: Time(sec)={round(time_sec, 3)},  TAR={round(tar, 3)}, "
              f"arv0={round(arv0, 3)}, arv3={round(arv3, 3)}, arv6={round(arv6, 3)},"
              f" angleOpenThreshold={angleTD}")
    else:
        p = os.path.join("resources", "testing", "ventil-tilstand", "5.jpg")
        #p = os.path.join("resources", "testing", "ventil-tilstand", "tilstand-test", "color-marking", "marked", "0", "OPEN", "001408.jpg")
        img = cv2.imread(p)
        res = dm.watershedVec(img, (0,), valves[6], vs.display)
        print(res)

















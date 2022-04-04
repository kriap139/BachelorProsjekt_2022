import src.Backend.StateDetection.Methods as dm
from src.GUI.PlotWindow import PlotWindow
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import os
import glob


class ValveStateTest:
    def __init__(self):
        self.window = PlotWindow()

    def display(self, title: str, img, cmap=None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap=cmap)

        self.window.addPlot(title, fig)

    def show(self):
        self.window.show()

    def siftTest(self, imgFolder: str, imgData: dict):
        pass

    def markingsTest(self):
        pass

    def watershedTest(self):
        pass


if __name__ == "__main__":
    cfgP = f"resources/model/classif/yolov4-ventiler.cfg"
    namesP = f"resources/model/classif/ventiler.names"
    weightsP = f"resources/model/classif/yolov4-ventiler_best.weights"

    rootDir = "resources/testing/ventil-tilstand/tilstand-test"
    vs = ValveStateTest()

    with open(os.path.join(rootDir, "ImageOverview.json"), mode='r') as f:
        boxes = json.load(f)










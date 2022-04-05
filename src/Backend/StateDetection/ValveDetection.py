from src.GUI.PlotWindow import PlotWindow
import matplotlib.pyplot as plt
import src.Backend.StateDetection.Methods as dm

import numpy as np
import cv2


class ValveState:
    def __init__(self):
        self.window = PlotWindow()

    def display(self, title: str, img: np.ndarray, cmap=None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap=cmap)

        self.window.addPlot(title, fig)

    def show(self):
        self.window.show()


if __name__ == "__main__":

    imgPath = f"resources/testing/ventil-tilstand/2.jpg"
    vs = ValveState()

    img_original = cv2.imread(imgPath)
    ret_type, result = dm.watershedVec(img_original, vs.display)

    if ret_type == dm.ReturnType.ANGLE:
        print(f"Vinklen til ventil i forhold til pipen: {np.round(result)} deg")
    else:
        print("Failed to calculate angle")

    vs.show()


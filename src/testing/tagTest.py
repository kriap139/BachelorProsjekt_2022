from src.Backend.StateDetection.Methods.TypeDefs import TYDisplay
from src.GUI.PlotWindow import PlotWindow
from src.Backend.StateDetection.Methods.constants import ValveState, ReturnType, RAD_TO_DEG
from src.Backend.StateDetection.Methods.filter_and_sorting import filterContours
from src.Util.Logging import Logging
from src.Backend.Valve import Valve
from typing import Union, Tuple
import numpy as np
import os
import cv2
import numbers
import matplotlib.pyplot as plt


class ValveStateTest:
    def __init__(self):
        self.window = PlotWindow()

    def display(self, title: str, img, cmap=None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap=cmap)

        self.window.addPlot(title, fig)
        pass

    def show(self):
        self.window.show()


def findMaskContours(hsl: np.ndarray, colorLower: tuple, colorUpper: tuple):
    # Find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(hsl, colorLower, colorUpper)

    # Deleting noises which are in area of mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

# For Texst(svart): Lower=(0, 44, 0), Upper=(180, 184, 43)
# tag(hvit): Lower=(0, 227, 255), Upper=(180, 0, 255)

# tag(new): Lower=(0, 0, 0), Upper=(180, 218, 255)


vs = ValveStateTest()
p = os.path.join("resources", "testing", "ventil-tilstand", "9.png")

img = cv2.imread(p)
hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

counours_t, mask_t = findMaskContours(hsl, colorLower=(0, 0, 0), colorUpper=(180, 218, 255))

vs.display("og-img", img)
vs.display("hsl", hsl)
vs.display("mask-Tag", mask_t)

cv2.drawContours(img, counours_t, -1, (0, 255, 0))
vs.display("countours-Tag", img)

vs.show()

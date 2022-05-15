from src.Backend.StateMethods.Methods import sift
import cv2
from src.testing.PlotWindow import PlotWindow
import matplotlib.pyplot as plt
import os
from src.Backend.Valve import Valve, ValveClass


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


v = Valve("", ValveClass(3, "", "", colorLower=(0, 83, 25), colorUpper=(12, 255, 111)))
vs = ValveStateTest()
p = os.path.join("resources", "testing", "sift-test", "test-bilde.jpg")

img = cv2.imread(p)

sift(img, [0, 0], v, vs.display)

vs.show()

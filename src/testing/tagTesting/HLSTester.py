from __future__ import print_function
import cv2 as cv
import os
from typing import Union
import numpy as np

# Code modified from OpenCV: https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html


class HSLTester:
    MAX_VALUE = 255
    MAX_VALUE_H = MAX_VALUE // 2

    def __init__(self, win_w: int = 600, win_h: int = 600):

        self.low_H = 0
        self.low_S = 0
        self.low_L = 0
        self.high_H = self.MAX_VALUE_H
        self.high_S = self.MAX_VALUE
        self.high_L = self.MAX_VALUE

        self.window_name = "HSL Image"
        self.window_src_name = "src Image"
        self.low_H_name = "Low H"
        self.low_S_name = "Low S"
        self.low_L_name = "Low L"
        self.high_H_name = "High H"
        self.high_S_name = "High S"
        self.high_L_name = "High L"

        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.namedWindow(self.window_src_name, cv.WINDOW_NORMAL)

        cv.resizeWindow(self.window_name, win_w, win_h)
        cv.resizeWindow(self.window_src_name, win_w, win_h)

        cv.createTrackbar(self.low_H_name, self.window_name, self.low_H, self.MAX_VALUE_H, self.on_low_H_thresh_trackbar)
        cv.createTrackbar(self.high_H_name, self.window_name, self.high_H, self.MAX_VALUE_H, self.on_high_H_thresh_trackbar)
        cv.createTrackbar(self.low_L_name, self.window_name, self.low_L, self.MAX_VALUE, self.on_low_L_thresh_trackbar)
        cv.createTrackbar(self.high_L_name, self.window_name, self.high_L, self.MAX_VALUE, self.on_high_L_thresh_trackbar)
        cv.createTrackbar(self.low_S_name, self.window_name, self.low_S, self.MAX_VALUE, self.on_low_S_thresh_trackbar)
        cv.createTrackbar(self.high_S_name, self.window_name, self.high_S, self.MAX_VALUE, self.on_high_S_thresh_trackbar)

    def on_low_H_thresh_trackbar(self, val):
        self.low_H = val
        self.low_H = min(self.high_H - 1, self.low_H)
        cv.setTrackbarPos(self.low_H_name, self.window_name, self.low_H)

    def on_high_H_thresh_trackbar(self, val):
        self.high_H = val
        self.high_H = max(self.high_H, self.low_H + 1)
        cv.setTrackbarPos(self.high_H_name, self.window_name, self.high_H)

    def on_low_S_thresh_trackbar(self, val):
        self.low_S = val
        self.low_S = min(self.high_S - 1, self.low_S)
        cv.setTrackbarPos(self.low_S_name, self.window_name, self.low_S)

    def on_high_S_thresh_trackbar(self, val):
        self.high_S = val
        self.high_S = max(self.high_S, self.low_S + 1)
        cv.setTrackbarPos(self.high_S_name, self.window_name, self.high_S)

    def on_low_L_thresh_trackbar(self, val):
        self.low_L = val
        self.low_L = min(self.high_L - 1, self.low_L)
        cv.setTrackbarPos(self.low_L_name, self.window_name, self.low_L)

    def on_high_L_thresh_trackbar(self, val):
        self.high_L = val
        self.high_L = max(self.high_L, self.low_L + 1)
        cv.setTrackbarPos(self.high_L_name, self.window_name, self.high_L)

    def setCurrentRange(self, lower: tuple, upper: tuple) -> bool:
        if (type(lower) == tuple) and (type(upper) == tuple) and (len(lower) == 3) and (len(upper) == 3):
            self.on_low_H_thresh_trackbar(lower[0])
            self.on_high_H_thresh_trackbar(upper[0])
            self.on_low_L_thresh_trackbar(lower[1])
            self.on_high_L_thresh_trackbar(upper[1])
            self.on_low_S_thresh_trackbar(lower[2])
            self.on_high_S_thresh_trackbar(upper[2])
            return True
        else:
            return False

    def test(self, src: Union[str, np.ndarray]):

        if isinstance(src, np.ndarray):
            img = src
        elif isinstance(src, str):
            img = cv.imread(src)
        else:
            raise ValueError(f"Unsupported argument {type(src)}")

        while True:
            frame = img.copy()

            if frame is None:
                break

            hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
            mask = cv.inRange(hls, (self.low_H, self.low_L, self.low_S), (self.high_H, self.high_L, self.high_S))


            cv.imshow(self.window_src_name, frame)
            cv.imshow(self.window_name, mask)

            key = cv.waitKey(30)

            if key == ord('q') or key == 27:
                cv.destroyWindow(self.window_name)
                cv.destroyWindow(self.window_src_name)


if "__file__" == "__main__":
    p = os.path.join("resources", "testing", "ventil-tilstand", "9.png")
    tester = HSLTester()
    tester.test(p)




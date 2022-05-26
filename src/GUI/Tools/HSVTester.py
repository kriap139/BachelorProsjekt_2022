import cv2 as cv
from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QSlider, QMenu
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QFontMetrics, QAction
from PyQt6.QtCore import QPoint, Qt, pyqtSlot, pyqtSignal
from src.GUI.Ui.HSVTester import Ui_HSVTester
from src.Backend.Tools.DomColor import HLS
from src.GUI.Tools.constants import ToolEnums
from src.Config import Config
import sys
from typing import Union
import numpy as np
import os


class HSVTester(QWidget):
    MAX_VALUE = 255
    MAX_VALUE_H = MAX_VALUE

    # signals
    load_images_sl = pyqtSignal(QWidget, name="load_images_sl")
    next_img_sl = pyqtSignal(QWidget)
    prev_img_sl = pyqtSignal(QWidget)
    auto_calc_range_sl = pyqtSignal(QWidget)

    def __init__(self, parent=None):
        super(HSVTester, self).__init__(parent)

        self.ui = Ui_HSVTester()
        self.tool_enum = ToolEnums.HLSTester
        self.img = None
        self.img_name = ""
        self.low_H = 0
        self.low_S = 0
        self.low_V = 0
        self.high_H = self.MAX_VALUE_H
        self.high_S = self.MAX_VALUE
        self.high_V = self.MAX_VALUE
        self.is_setting_values = False

        self.ui.setupUi(self)

        self.ui.minH1.setText(str(0))
        self.ui.minH2.setText(str(0))
        self.ui.maxH1.setText(str(self.MAX_VALUE_H))
        self.ui.maxH2.setText(str(self.MAX_VALUE_H))

        self.ui.minS1.setText(str(0))
        self.ui.minS2.setText(str(0))
        self.ui.maxS1.setText(str(self.MAX_VALUE))
        self.ui.maxS2.setText(str(self.MAX_VALUE))

        self.ui.minV1.setText(str(0))
        self.ui.minV2.setText(str(0))
        self.ui.maxV1.setText(str(self.MAX_VALUE))
        self.ui.maxV2.setText(str(self.MAX_VALUE))

        self.ui.sliderHLow.setRange(self.low_H, self.high_H)
        self.ui.sliderHHigh.setRange(self.low_H, self.high_H)
        self.ui.sliderSLow.setRange(self.low_S, self.high_S)
        self.ui.sliderSHigh.setRange(self.low_S, self.high_S)
        self.ui.sliderVLow.setRange(self.low_V, self.high_V)
        self.ui.sliderVHigh.setRange(self.low_V, self.high_V)

        self.ui.sliderHLow.sliderMoved.connect(self.on_low_H_change)
        self.ui.sliderHHigh.sliderMoved.connect(self.on_high_H_change)
        self.ui.sliderSLow.sliderMoved.connect(self.on_low_S_change)
        self.ui.sliderSHigh.sliderMoved.connect(self.on_high_S_change)
        self.ui.sliderVLow.sliderMoved.connect(self.on_low_V_change)
        self.ui.sliderVHigh.sliderMoved.connect(self.on_high_V_change)

        self.ui.resetButton.clicked.connect(self.reset_values)

        self.ui.visualizer.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.__show_context_menu)

        self.ui.fileButton.clicked.connect(self.load_images)
        self.ui.nextIMG.clicked.connect(self.next_img)
        self.ui.prevIMG.clicked.connect(self.prev_img)
        self.ui.autoCalcRange.clicked.connect(self.auto_calc_range)

        self.set_slider_states()
        self.update_visualizer()

    @pyqtSlot(str, str)
    def update_top_bars(self, file_label: str, arrow_label: str):
        self.ui.fileLabel.setText(file_label)
        self.ui.arrowButtonsLabel.setText(arrow_label)

    @pyqtSlot()
    def load_images(self):
        self.load_images_sl.emit(self)

    def next_img(self, checked: bool):
        self.next_img_sl.emit(self)

    def prev_img(self, checked: bool):
        self.prev_img_sl.emit(self)

    def auto_calc_range(self):
        self.auto_calc_range_sl.emit(self)

    @pyqtSlot(QPoint)
    def __show_context_menu(self, pos: QPoint) -> None:
        txt = self.ui.visualizer.selectedText()

        if txt:
            menu = QMenu("Context menu", self)
            copy = QAction("copy", self)

            copy.triggered.connect(self.__copy_to_clipboard)
            menu.addAction(copy)
            menu.exec(self.mapToGlobal(pos))

    @pyqtSlot()
    def __copy_to_clipboard(self) -> None:
        text = f"{self.ui.visualizer.selectedText()}"
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    def update_visualizer(self):
        txt = f"lower=[{self.low_H}, {self.low_S}, {self.low_V}],   upper=[{self.high_H}, {self.high_S}, {self.high_V}]"
        self.ui.visualizer.setText(txt)

    def on_low_H_change(self, val):
        self.low_H = val
        self.low_H = min(self.high_H - 1, self.low_H)

        self.update_visualizer()
        self.ui.sliderHLow.setValue(self.low_H)
        self.update_img()

    def on_high_H_change(self, val):
        self.high_H = val
        self.high_H = max(self.high_H, self.low_H + 1)

        self.update_visualizer()
        self.ui.sliderHHigh.setValue(self.high_H)
        self.update_img()

    def on_low_S_change(self, val):
        self.low_S = val
        self.low_S = min(self.high_S - 1, self.low_S)

        self.update_visualizer()
        self.ui.sliderSLow.setValue(self.low_S)
        self.update_img()

    def on_high_S_change(self, val):
        self.high_S = val
        self.high_S = max(self.high_S, self.low_S + 1)

        self.update_visualizer()
        self.ui.sliderSHigh.setValue(self.high_S)
        self.update_img()

    def on_low_V_change(self, val):
        self.low_V = val
        self.low_V = min(self.high_V - 1, self.low_V)

        self.update_visualizer()
        self.ui.sliderVLow.setValue(self.low_V)
        self.update_img()

    def on_high_V_change(self, val):
        self.high_V = val
        self.high_V = max(self.high_V, self.low_V + 1)

        self.update_visualizer()
        self.ui.sliderVHigh.setValue(self.high_V)
        self.update_img()

    def set_current_values(self, lower: HLS, upper: HLS) -> bool:
        if (type(lower) == HLS) and (type(upper) == HLS):
            self.is_setting_values = True

            #print(lower, upper)

            self.on_low_H_change(lower.h)
            self.on_high_H_change(upper.h)
            self.on_low_S_change(lower.l)
            self.on_high_S_change(upper.l)
            self.on_low_V_change(lower.s)
            self.on_high_V_change(upper.s)

            self.is_setting_values = False
            self.update_img()
            return True
        else:
            return False

    def update_img(self):
        if self.img is not None and not self.is_setting_values:
            hsv = cv.cvtColor(self.img, cv.COLOR_RGB2HSV)
            mask = cv.inRange(hsv, (self.low_H, self.low_S, self.low_V), (self.high_H, self.high_S, self.high_V))

            result = cv.bitwise_and(hsv, hsv, mask=mask)
            result = cv.cvtColor(result, cv.COLOR_HSV2RGB)

            self.ui.hlsImg.setPixmap(self.to_pixmap(result))

    def set_img(self, src: Union[str, np.ndarray], name=""):
        """if src is an Image, it has to be RGB """

        if isinstance(src, np.ndarray):
            img = src
        elif isinstance(src, str):
            img = cv.imread(src)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported argument {type(src)}")

        self.img = img
        self.img_name = name
        self.ui.originalImg.setPixmap(self.to_pixmap(img))
        self.update_img()

    @classmethod
    def to_pixmap(cls, img: np.ndarray):
        """Convert Opencv RGB image to a QPixmap"""
        h, w, ch = img.shape
        bpl = ch * w
        converted = QImage(img.data, w, h, bpl, QImage.Format.Format_RGB888)

        return QPixmap.fromImage(converted)

    def set_slider_states(self):
        self.ui.sliderHLow.setValue(self.low_H)
        self.ui.sliderHHigh.setValue(self.high_H)
        self.ui.sliderSLow.setValue(self.low_S)
        self.ui.sliderSHigh.setValue(self.high_S)
        self.ui.sliderVLow.setValue(self.low_V)
        self.ui.sliderVHigh.setValue(self.high_V)

    def reset_values(self):
        self.low_H = 0
        self.low_S = 0
        self.low_V = 0
        self.high_H = self.MAX_VALUE_H
        self.high_S = self.MAX_VALUE
        self.high_V = self.MAX_VALUE
        self.set_slider_states()
        self.update_img()

    def reset(self):
        self.reset_values()
        self.set_slider_states()
        self.ui.hlsImg.clear()
        self.ui.originalImg.clear()
        self.update_visualizer()
        self.img = None


if __name__ == "__main__":
    app = QApplication(sys.argv)

    fn1, fn2 = "MOV_0368.mp4", "MOV_0375.mp4"
    fn = fn2
    baseName, ext = os.path.splitext(fn)

    p = Config.createAppDataPath("images", "results", "SDST", baseName, "raw", fName="140.jpg")

    main = QMainWindow()
    tester = HSVTester(main)

    main.resize(900, 600)
    main.setCentralWidget(tester)
    tester.set_img(p)

    main.show()
    sys.exit(app.exec())

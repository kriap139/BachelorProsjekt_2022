from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QSizePolicy, QCommonStyle, QStyle
from PyQt6.QtGui import QPainter, QPaintEvent, QImage, QPixmap, QTextOption, QColor
from PyQt6.QtCore import Qt, QRect, QRectF, pyqtSignal, pyqtSlot
from src.GUI.Ui.DomColorGUI import Ui_DomColorGUI
from src.GUI.Ui.ImageWindow import Ui_ImageWindow
from src.Backend.Logging import LOG
from src.Backend.Tools import *
from src.GUI.Tools.constants import *
from typing import Tuple, Union
import sys
import numpy as np
from typing_extensions import Protocol
import cv2 as cv
from enum import Enum

__all__ = ['DomColorGUI']


class TYDraw(Protocol):
    def __call__(self, area: QRect, painter: QPainter) -> None:
        pass


class DrawMethod(Enum):
    BOXES = 0
    BAR = 1


class DomColorDrawArea(QWidget):
    def __init__(self, parent=None, areaId: int = 0):
        super(DomColorDrawArea, self).__init__(parent)

        self.colorData: ColorData = None
        self.bkgColor = Qt.GlobalColor.white
        self.areaId = areaId

        self.boxSpacing = 10
        self.boxTextTopMargin = 6
        self.boxTextHeightRel = 0.09
        self.paletteXMarginRel = 0.01
        self.paletteLineWidth = 3
        self.textColor = Qt.GlobalColor.black

        self.draw = self.draw_boxes
        self.drawMethod = DrawMethod.BOXES

        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setObjectName(f"DrawArea {areaId}")

    def paintEvent(self, a0: QPaintEvent) -> None:
        super(DomColorDrawArea, self).paintEvent(a0)

        if self.colorData is None:
            return

        area = a0.rect()
        painter = QPainter()

        painter.begin(self)

        painter.fillRect(a0.rect(), self.bkgColor)
        self.draw(area, painter)

        painter.end()

    def draw_bar(self, area: QRect, painter: QPainter):

        xMargin = int(area.width() * self.paletteXMarginRel)
        w = area.width() - xMargin * 2

        h = area.height() // 3
        hBox = h
        hText = h - int(h * 0.05)

        yTextStep = int(h * 0.16)

        boxWidths = [int(w * pct) for pct, _ in self.colorData]
        delta = w - sum(boxWidths)

        x = area.x() + (delta // 2) + xMargin

        yBox = area.y() + h
        yText = yBox + hBox + yTextStep

        painter.setPen(self.textColor)

        for i, (pct, (r, g, b)) in enumerate(self.colorData):
            painter.fillRect(x, yBox, boxWidths[i], hBox, QColor.fromRgb(r, g, b))

            tRect = QRect(x, yText, boxWidths[i], hText)
            painter.drawText(QRectF(tRect), f"{round(pct * 100, 2)}%", QTextOption(Qt.AlignmentFlag.AlignHCenter))

            x += boxWidths[i]

    def draw_boxes(self, area: QRect, painter: QPainter):

        numBoxes = len(self.colorData)
        spacing = self.boxSpacing

        w = area.width()
        h = area.height() - int(area.height() * self.boxTextHeightRel) - self.boxTextTopMargin
        hText = area.height() - h

        boxSize = (w - spacing * (numBoxes - 1)) // numBoxes

        x = area.x()
        y = area.y()

        if boxSize > h:
            delta = boxSize - h
            boxSize -= delta
            x += (delta * numBoxes) // 2
        elif boxSize < h:
            delta = h - boxSize
            delta2 = delta // 2

            tmp = (w - spacing * (numBoxes - 1) - delta) // numBoxes

            if not (tmp < 0):
                boxSize = tmp
                x += delta2

            y += delta2

        yText = y + boxSize + self.boxTextTopMargin
        tRect = QRect(x, yText, boxSize, hText)

        painter.setPen(self.textColor)

        for (pct, (r, g, b)) in self.colorData:
            painter.fillRect(x, y, boxSize, boxSize, QColor.fromRgb(r, g, b))

            painter.drawText(QRectF(tRect), f"{round(pct * 100, 2)}%", QTextOption(Qt.AlignmentFlag.AlignHCenter))

            x += spacing + boxSize
            tRect = QRect(x, yText, boxSize, hText)

    def set_color_data(self, colorData: ColorData):
        self.colorData = colorData
        self.repaint()

    def set_draw_method(self, method: DrawMethod):
        if method == DrawMethod.BOXES:
            if not (method == self.drawMethod):
                self.draw = self.draw_boxes
                self.drawMethod = DrawMethod.BOXES
                self.repaint()
        elif method == DrawMethod.BAR:
            if not (method == self.drawMethod):
                self.draw = self.draw_bar
                self.drawMethod = DrawMethod.BAR
                self.repaint()
        else:
            LOG(f"Unknown Draw Method {DrawMethod}")

    def get_draw_method(self):
        return self.drawMethod

    def set_attr(self, name: str, val):
        attr = getattr(self, name, None)

        if attr is None:
            LOG(f"Unable to find Attribute {name}")
        else:
            setattr(self, name, val)
            self.repaint()


class ImageWindow(QWidget):
    def __init__(self, parent=None):
        super(ImageWindow, self).__init__(parent, Qt.WindowType.Window)

        self.ui = Ui_ImageWindow()
        self.ui.setupUi(self)

    def set_img(self, img: QPixmap, title: str = None, winTitle: str = None):
        self.ui.image.setPixmap(img)

        if title is not None:
            self.ui.title.setText(title)

        if winTitle is not None:
            self.setWindowTitle(winTitle)


class DomColorGUI(QWidget):

    # signals
    load_images_sl = pyqtSignal(QWidget, name="load_images_sl")
    next_img_sl = pyqtSignal(QWidget)
    prev_img_sl = pyqtSignal(QWidget)

    def __init__(self, parent=None, clusters: int = 4):
        super(DomColorGUI, self).__init__(parent)

        self.ui = Ui_DomColorGUI()
        self.imgWin = ImageWindow(self)
        self.tool_enum = ToolEnums.DomColorGUI

        self.img = None
        self.img_name = ""
        self.imgFormat = None

        self.clusters = {
            1: max(1, clusters),
            2: max(1, clusters)
        }

        self.dataArr = {
            1: None,
            2: None
        }

        self.formatArr = {
            1: ColorFormat.RGB,
            2: ColorFormat.RGB
        }

        self.drawArea1 = DomColorDrawArea(parent=self, areaId=1)
        self.drawArea2 = DomColorDrawArea(parent=self, areaId=2)
        self.currentArea = self.drawArea1
        self.currentAreaChanged = False

        self.drawArea1.set_draw_method(DrawMethod.BAR)
        self.drawArea2.set_draw_method(DrawMethod.BOXES)

        self.ui.setupUi(self)

        style = QCommonStyle()
        self.ui.nextIMG.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
        self.ui.prevIMG.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_ArrowLeft))

        self.ui.areaCFormat.clear()

        for cf in ColorFormat:
            self.ui.areaCFormat.addItem(cf.name)

        self.ui.mainLayout.replaceWidget(self.ui.drawArea2, self.drawArea2)
        self.ui.drawArea2.deleteLater()

        self.ui.mainLayout.replaceWidget(self.ui.drawArea1, self.drawArea1)
        self.ui.drawArea1.deleteLater()

        self.ui.area1Picker.setChecked(True)
        self.set_area_bar_info()

        self.ui.showButton.clicked.connect(self.show_image)
        self.ui.area1Picker.toggled.connect(self.__area_picker_changed)
        self.ui.areaDrawBoxes.toggled.connect(self.change_draw_method)
        self.ui.areaCFormat.currentTextChanged.connect(self.color_format_changed)
        self.ui.areaClassPicker.valueChanged.connect(self.clusters_changed)
        self.ui.fileButton.clicked.connect(self.load_images)
        self.ui.nextIMG.clicked.connect(self.next_img)
        self.ui.prevIMG.clicked.connect(self.prev_img)

    def load_images(self):
        self.load_images_sl.emit(self)

    @pyqtSlot()
    def load_images(self):
        self.load_images_sl.emit(self)

    def next_img(self, checked: bool):
        self.next_img_sl.emit(self)

    def prev_img(self, checked: bool):
        self.prev_img_sl.emit(self)

    @pyqtSlot(str, str)
    def update_top_bars(self, file_label: str, arrow_label: str):
        self.ui.fileLabel.setText(file_label)
        self.ui.arrowButtonsLabel.setText(arrow_label)

    @pyqtSlot(bool)
    def __area_picker_changed(self, checked: bool):
        if self.ui.area1Picker.isChecked():
            self.currentArea = self.drawArea1
            self.set_area_bar_info()
        elif self.ui.area2Picker.isChecked():
            self.currentArea = self.drawArea2
            self.set_area_bar_info()

    def set_area_bar_info(self):
        self.currentAreaChanged = True

        area = self.currentArea
        m = area.drawMethod

        if m == DrawMethod.BOXES:
            self.ui.areaDrawBoxes.setChecked(True)
        else:
            self.ui.areaDrawBar.setChecked(True)

        self.ui.areaClassPicker.setValue(self.clusters[area.areaId])
        self.ui.areaCFormat.setCurrentText(self.formatArr[area.areaId].name)

        self.currentAreaChanged = False

    def get_current_draw_method(self) -> DrawMethod:
        if self.ui.areaDrawBoxes.isChecked():
            return DrawMethod.BOXES
        elif self.ui.areaDrawBar.isChecked():
            return DrawMethod.BAR

    @pyqtSlot(str)
    def color_format_changed(self, name):
        if not self.currentAreaChanged:
            cf = ColorFormat[name]
            area = self.currentArea
            self.formatArr[area.areaId] = cf
            self.update_draw_area(area)

    @pyqtSlot(int)
    def clusters_changed(self, clusters: int):
        if not self.currentAreaChanged:
            area = self.currentArea
            self.clusters[area.areaId] = max(1, clusters)
            self.update_draw_area(area)

    @pyqtSlot(bool)
    def change_draw_method(self, checked: bool):
        if not self.currentAreaChanged:
            dm = self.get_current_draw_method()
            self.currentArea.set_draw_method(dm)

    @classmethod
    def calc_dom_colors(cls, img: np.ndarray, useFormat: ColorFormat = ColorFormat.RGB, clusters=4) \
            -> ColorData:
        """ img has to be in RGB format"""

        colorData = findDomColors(img, clusters, useFormat)

        if useFormat != ColorFormat.RGB:
            colorData = colorDataToRGB(colorData, useFormat)  # tuple((pct, cvPixelToRGB(color, formatOut)) for pct, color in colorData)

        return colorData

    def update_draw_area(self, areas: Union[DomColorDrawArea, Tuple[DomColorDrawArea, DomColorDrawArea]]):

        if self.img is None:
            return

        if type(areas) == tuple:
            a1, a2 = areas[0], areas[1]

            clu1, clu2 = self.clusters[a1.areaId], self.clusters[a2.areaId]
            cf1, cf2 = self.formatArr[a2.areaId], self.formatArr[a2.areaId]

            if (clu1 == clu2) and (cf1 == cf2):
                colorData = self.calc_dom_colors(self.img, cf1, clu1)
                self.drawArea1.set_color_data(colorData)
                self.drawArea2.set_color_data(colorData)
            else:
                for clu, cf, area in ((clu1, cf1, self.drawArea1), (clu2, cf2, self.drawArea2)):
                    colorData = self.calc_dom_colors(self.img, cf, clu)
                    area.set_color_data(colorData)

        elif type(areas) == DomColorDrawArea:
            colorData = self.calc_dom_colors(self.img, self.formatArr[areas.areaId], self.clusters[areas.areaId])
            self.drawArea1.set_color_data(colorData)

    def show_image(self):
        if self.img is not None:
            self.imgWin.set_img(self.to_pixmap(self.img))

            ww = int(self.width() * 0.5)
            wh = int(self.height() * 0.5)
            self.imgWin.resize(ww, wh)

            qScreen = QApplication.activeWindow()
            screen = qScreen.geometry()

            currDim = self.imgWin.frameGeometry()

            centerX = screen.center().x() - int(currDim.width() * 0.5)
            centerY = screen.center().y() - int(currDim.height() * 0.5)

            self.imgWin.move(centerX, centerY)

            self.imgWin.setFocus()
            self.imgWin.show()

    def set_img(self, src: Union[str, np.ndarray], name=""):
        """If src is an image, it has to be in RGB format"""

        if isinstance(src, np.ndarray):
            self.img = src
        elif isinstance(src, str):
            img = cv.imread(src)
            self.img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            if self.img is None:
                LOG(f"Unable to read image from path {src}")
                return tuple()
        else:
            raise ValueError(f"Unsupported argument {type(src)}")

        self.img_name = name
        self.update_draw_area((self.drawArea1, self.drawArea2))

    @classmethod
    def to_pixmap(cls, img: np.ndarray):
        """Convert Opencv RGB image to a QPixmap"""
        h, w, ch = img.shape
        bpl = ch * w
        converted = QImage(img.data, w, h, bpl, QImage.Format.Format_RGB888)

        return QPixmap.fromImage(converted)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    p = Config.createAppDataPath("testing", "tag", "old_tags", fName="2.jpg")
    dw = DomColorGUI(clusters=4)
    dw.set_img(p)

    main = QMainWindow()
    main.resize(900, 600)
    main.setCentralWidget(dw)

    main.show()
    sys.exit(app.exec())

    #colorData = findDomColors(p, 4, ColorFormat.HLS)
    #print(colorData)



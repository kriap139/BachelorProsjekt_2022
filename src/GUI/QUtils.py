import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def toQPixmap(img: np.ndarray) -> QPixmap:
    """Convert Opencv RGB image to a QPixmap"""
    h, w, ch = img.shape
    bpl = ch * w
    converted = QImage(img.data, w, h, bpl, QImage.Format.Format_RGB888)

    return QPixmap.fromImage(converted)


def toQImage(img: np.ndarray) -> QImage:
    """Convert Opencv RGB image to a QImage"""
    h, w, ch = img.shape
    bpl = ch * w
    return QImage(img.data, w, h, bpl, QImage.Format.Format_RGB888)

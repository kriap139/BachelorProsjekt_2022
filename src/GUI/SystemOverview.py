from src.GUI.Ui.SystemOverview import Ui_SystemOverview
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap


class SystemOverview(QWidget):

    # Signals

    # !Signals

    def __init__(self, parent=None):
        super(SystemOverview, self).__init__(parent)

        self.ui = Ui_SystemOverview()
        self.ui.setupUi(self)

    def setImage(self, img: QImage):
        self.ui.image.setPixmap(QPixmap.fromImage(img))


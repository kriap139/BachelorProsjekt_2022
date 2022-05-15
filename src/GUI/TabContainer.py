from PyQt6.QtWidgets import QTabWidget, QWidget
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtGui import QResizeEvent
from src.GUI.Ui.TabContainer import Ui_TabContainer
from src.GUI.ValveOverview import ValveOverview
from src.GUI.SystemOverview import SystemOverview
from src.GUI.BackInterface.QValveStateDetector import QValveStateDetector


class TabContainer(QTabWidget):
    def __init__(self, vsd: QValveStateDetector, parent=None, name: str = "TabContainer"):
        super(TabContainer, self).__init__(parent)

        self.ui = Ui_TabContainer()
        self.valveOverview = ValveOverview(self)
        self.systemOverview = SystemOverview(self)
        self.valveStateDetector = vsd

        self.ui.setupUi(self)
        self.setObjectName(name)

        self.replaceTab(0, self.valveOverview)
        self.replaceTab(1, self.systemOverview)
        self.setCurrentWidget(self.valveOverview)

        vsd.imageFeedUpdated.connect(self.systemOverview.setImage)

    def replaceTab(self, index: int, newPage: QWidget, title: str = "") -> None:
        nTabs = self.count()

        if (nTabs == 0) or (index > nTabs):
            return

        if not title:
            title = self.tabText(index)

        old = self.widget(index)
        self.removeTab(index)
        old.deleteLater()
        self.insertTab(index, newPage, title)

    def resizeEvent(self, e: QResizeEvent) -> None:
        super(TabContainer, self).resizeEvent(e)
        # self.valveOverview.resizeEvent(e)



import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QResizeEvent
from src.GUI.TabContainer import TabContainer
from src.GUI.Ui.MainWindow import Ui_MainWindow
from src.Config import Config, LOG
from src.GUI.ValveGUI import ValveGUI
from src.Backend import Valve, ValveState
from src.GUI.ToolsHandler import ToolsHandler, ToolEnums
from src.GUI.BackInterface.QValveStateDetector import QValveStateDetector
import os


class MainWindow(QMainWindow):

    def __init__(self, app: QApplication, vsd: QValveStateDetector, parent: QWidget = None, name: str = "MainWindow"):
        super(MainWindow, self).__init__(parent)

        self.ui = Ui_MainWindow()
        self.valveStateDetector = QValveStateDetector(self)
        self.tabContainer = TabContainer(vsd, self)
        self.toolHandler = ToolsHandler(self.tabContainer, self)

        self.ui.setupUi(self)
        self.setObjectName(name)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCentralWidget(self.tabContainer)

        self.ui.menuTools.triggered.connect(self.toolHandler.add_tool)

        for tool in ToolEnums:
            self.ui.menuTools.addAction(tool.name)

        theme, valveImages, windowGeom = Config.loadGUIData()

        self.setTheme(theme, app)
        ValveGUI.loadValveImages(valveImages)

        success = False
        if windowGeom is not None:
            success = self.setScreenSize(*windowGeom)

        if not success:
            self.setSizeRelToScreen(0.6, 0.6)

        app.aboutToQuit.connect(self.shutdownCleanup)

        v = Valve("ABCDEF", None, ValveState.CLOSED)
        v2 = Valve("ASDFGH", None, ValveState.OPEN)
        v3 = Valve("vbmjjjjjrløf", None, ValveState.UNKNOWN)
        self.tabContainer.valveOverview.renderValves((v, v, v, v, v,v ,v ,v ,v, v2, v, v, v, v3, v, v, v))

    def setSizeRelToScreen(self, relW: float, relH: float, center: bool = True):
        if (relW > 1 or relH > 1) or (relW < 0 or relH < 0):
            return

        qScreen = QApplication.primaryScreen()
        screen = qScreen.availableGeometry()

        newWidth = screen.width() * relW
        newHeight = screen.height() * relH
        self.resize(int(newWidth), int(newHeight))

        if center:
            currDim = self.frameGeometry()

            centerX = screen.center().x() - int(currDim.width() * 0.5)
            centerY = screen.center().y() - int(currDim.height() * 0.5)
            self.move(centerX, centerY)

    def setScreenSize(self, cx: int, cy: int, w: int, h: int) -> bool:
        qScreen = QApplication.primaryScreen()
        screen = qScreen.availableGeometry()

        if (screen.width() < (cx + w)) or (screen.height() < (cy + h)):
            return False

        self.resize(w, h)
        self.tabContainer.resize(w, h)
        self.move(cx, cy)
        return True

    def setTheme(self, filePath: str, app: QApplication = None) -> bool:
        if not os.path.exists(filePath):
            LOG(f"Failed to load theme: {filePath}")
            return False

        with open(filePath, mode='r') as f:
            styleSheet = f.read()

        if app is not None:
            app.setStyleSheet(styleSheet)
        else:
            self.setStyleSheet(styleSheet)
        return True

    @pyqtSlot()
    def shutdownCleanup(self):
        self.valveStateDetector.shutdown()
        Config.saveWindowGeometry(self.pos().x(), self.pos().y(), self.width(), self.height())

    def resizeEvent(self, a0: QResizeEvent) -> None:
        super(MainWindow, self).resizeEvent(a0)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    video = Config.createAppDataPath("video", fName="al.mp4")

    valveStateDetector = QValveStateDetector()
    #valveStateDetector.addStream(video)

    window = MainWindow(app, valveStateDetector)
    window.show()

    sys.exit(app.exec())

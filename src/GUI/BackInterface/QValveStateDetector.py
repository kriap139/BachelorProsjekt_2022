from src.GUI.BackInterface.QBackendHandlerINST import QBackendHandlerInstance
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage
from src.Config import Config
from src.Backend.Valve import Valve, ValveClass, ValveState
from typing import List, Tuple


class QValveStateDetector(QObject):

    # Signals
    valvesUpdated = pyqtSignal(list, name="valveUpdated")
    imageFeedUpdated = pyqtSignal(QImage, name="imageFeedUpdated")
    # !Signals

    def __init__(self, parent=None):
        super(QValveStateDetector, self).__init__(parent)

        data = Config.loadValveInfoData()

        self.valveClasses = data.valveClasses
        self.valves = data.valves
        self.backHandler = QBackendHandlerInstance
        self.backHandler.updateImageFeed.connect(self.imageFeedUpdated)

    def addStream(self, streamPath: str):
        self.backHandler.addStream(streamPath)

    def shutdown(self):
        self.backHandler.shutdown()

    def setCurrentImageFeed(self, streamID: str):
        self.backHandler.setCurrentImageFeed(streamID)

    def getValves(self):
        return self.valves.values()



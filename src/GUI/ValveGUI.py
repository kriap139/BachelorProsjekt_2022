from PyQt6.QtWidgets import QWidget
from src.Backend.Valve import Valve, ValveState
from src.GUI.Ui.ValveGUI import Ui_ValveGUI
from PyQt6.QtGui import QPixmap
from typing import Optional, Dict, Union
from src.Backend.Logging import LOG, LOGType


__all__ = ['ValveGUI']


class ValveGUI(QWidget):
    VALVE_IMAGES: Dict[ValveState, QPixmap] = {}

    VALVE_COLORS = {
        ValveState.UNKNOWN: [238, 210, 2, 1],
        ValveState.CLOSED: [224, 34, 10, 1],
        ValveState.OPEN: [17, 230, 0, 1]
    }

    VALVE_COLORS_BGRA = {
        ValveState.UNKNOWN: [2, 210, 238, 1],
        ValveState.CLOSED: [10, 34, 224, 1],
        ValveState.OPEN: [0, 230, 17, 1]
    }

    def __init__(self, parent=None, valve: Optional[Valve] = None):
        super(ValveGUI, self).__init__(parent)

        self.ui = Ui_ValveGUI()
        self.ui.setupUi(self)
        self.valve = valve
        self.imgState = ValveState.UNKNOWN

        self.setValve(valve)

    def setValve(self, v: Union[Valve, None]):
        if v is None:
            self.valve = None
            self.setImage(ValveState.UNKNOWN)
        else:
            self.valve = v
            self.setObjectName(f"{self.objectName()}: {v.id}")
            self.ui.valveID.setText(v.id)

            if not (self.imgState == v.state):
                self.setImage(v.state)

    def setImage(self, state: ValveState):
        img = self.VALVE_IMAGES.get(state)

        if img is None:
            state = ValveState.UNKNOWN
            img = self.VALVE_IMAGES.get(state)
            LOG(f"Valve Image to UNKNOWN valveState is None")

        if img is None:
            LOG(f"Valve Image to {state} ValveState is None!")
            return

        self.imgState = state
        self.ui.valveImage.setPixmap(img)

        c = self.VALVE_COLORS.get(state)

        if c is None:
            LOG(f"Unable to set ValveID background color for valveState {state}")
        else:
            self.ui.valveID.setStyleSheet(
                f"background: rgba({c[0]}, {c[1]}, {c[2]}, {c[3]});"
                f"color: #000000;"
            )

    def reset(self):
        self.setValve(None)

    @classmethod
    def loadValveImages(cls, imgData: Dict[ValveState, str]):
        images: Dict[ValveState, QPixmap] = {state: QPixmap() for state in imgData.keys()}

        for state, path in imgData.items():
            success = images[state].load(path)

            if not success:
                LOG(f"Failed to load Valve Image for valve state {state}, from path: {path}", LOGType.WARNING)

        cls.VALVE_IMAGES = images








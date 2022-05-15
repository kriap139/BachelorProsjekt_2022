from PyQt6.QtWidgets import QWidget, QMenu
from PyQt6.QtCore import pyqtSlot, Qt, QSize, QPoint, QTimer
from src.Backend import Valve
from src.GUI.ValveGUI import ValveGUI
from typing import List, Union, Tuple, Dict
from PyQt6.QtGui import QResizeEvent, QCursor, QAction
from src.GUI.Ui.ValveOverview import Ui_ValveOverview
from PyQt6.QtWidgets import QListWidgetItem
from src.GUI.Ui.InfoBubble import Ui_InfoBubble


class InfoBubble(QWidget):
    def __init__(self, parent=None, windowType=Qt.WindowType.ToolTip):
        super(InfoBubble, self).__init__(parent, windowType)

        self.ui = Ui_InfoBubble()

        # self.bubbleMargins = QMargins(15, 5, 15, 5)
        # self.textMargin = QMargins(25, 15, 25, 15)
        # self.bubbleColor = QColor(56, 79, 98)
        # self.textColor = Qt.GlobalColor.black
        self.ui.setupUi(self)

    def setValve(self, v: Valve):
        self.ui.textLabel.setText(v.infoLabelStr())
        self.repaint()


class ValveOverview(QWidget):
    def __init__(self, parent=None, infoBubbleTimeoutMS: int = 3600, name: str = "ValveOverview"):
        super(QWidget, self).__init__(parent)

        self.ui = Ui_ValveOverview()
        self.infoBubble = InfoBubble(self, Qt.WindowType.ToolTip | Qt.WindowType.WindowStaysOnTopHint)
        self.infoBubbleWindow = InfoBubble(self, Qt.WindowType.Tool)
        self.infoBubbleWindowValve = None
        self.infoBubbleTimeoutMS = infoBubbleTimeoutMS
        self.infoTimer = QTimer(self)
        self.valveMappings: Dict[str, ValveGUI] = {}

        self.ui.setupUi(self)
        self.setObjectName(name)

        self.ui.valveList.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.ui.valveList.customContextMenuRequested.connect(self.__show_context_menu)
        self.infoTimer.timeout.connect(self.hideValveInfo)

        # Have significant impact on presentation!, Unsure how the grid is calculated
        self.ui.valveList.setIconSize(QSize(100,  199))

        self.ui.valveList.itemClicked.connect(self.showValveInfo)

    def showValveInfo(self, item: QListWidgetItem):
        if self.infoTimer.isActive():
            self.infoTimer.stop()

        self.infoBubbleWindow.hide()

        vg: ValveGUI = self.ui.valveList.itemWidget(item)
        self.infoBubble.setValve(vg.valve)

        pos = QCursor.pos()

        self.infoBubble.show()
        ihh = self.infoBubble.rect().height()
        pos.setY(pos.y() - ihh)
        self.infoBubble.move(pos)

        self.infoTimer.start(self.infoBubbleTimeoutMS)

    def hideValveInfo(self):
        self.infoBubble.hide()

    @pyqtSlot(QPoint)
    def __show_context_menu(self, pos: QPoint) -> None:
        item = self.ui.valveList.itemAt(pos)

        if item is None:
            return

        vg: ValveGUI = self.ui.valveList.itemWidget(item)
        v: Valve = vg.valve if type(vg.valve) == Valve else None

        if type(v) == Valve:
            self.infoBubbleWindowValve = v
            menu = QMenu("Context menu", self)
            show = QAction("show valveInfo", self)
            show.triggered.connect(self.showValveInfoWindow)
            menu.addAction(show)
            menu.exec(QCursor.pos())

    def showValveInfoWindow(self):
        v: Valve = self.infoBubbleWindowValve

        if v is not None:
            self.infoBubbleWindow.hide()
            self.infoBubble.hide()

            self.infoBubbleWindow.setValve(v)
            self.infoBubbleWindowValve = None

            pos = QCursor.pos()
            listP = self.ui.valveList.mapFromGlobal(pos)
            item = self.ui.valveList.itemAt(listP)

            if item is not None:
                vg: ValveGUI = self.ui.valveList.itemWidget(item)
                if type(vg) == ValveGUI:
                    self.infoBubbleWindow.show()
                    pos.setY(pos.y() - int(self.infoBubbleWindow.rect().height()))
                    pos.setX(pos.x() - int(self.infoBubbleWindow.rect().width() * 0.5))
                    self.infoBubbleWindow.move(pos)
            else:
                self.infoBubbleWindow.move(pos)
                self.infoBubbleWindow.show()
        else:
            self.infoBubbleWindow.hide()

    def setValves(self, valves: Union[List[Valve], Tuple[Valve]]):
        self.valveMappings.clear()

        for valve in valves:
            self.valveMappings[valve.id] = None

        self.renderValves(valves)

    def renderValves(self, valves: Union[List[Valve], Tuple[Valve]]):
        if not any(valves):
            for i in range(self.ui.valveList.count()):
                self.ui.valveList.setHidden(True)
                return

        count = len(valves)
        glen = self.ui.valveList.count()

        if count > glen:
            self.reserveGuiItems((count - glen))
        else:
            for i in range(count, glen):
                self.ui.valveList.item(i).setHidden(True)

        for i, v in enumerate(valves):
            listItem = self.ui.valveList.item(i)
            vg: ValveGUI = self.ui.valveList.itemWidget(listItem)
            vg.setValve(v)
            listItem.setSizeHint(vg.size())
            listItem.setHidden(False)

            self.valveMappings[v.id] = vg

    def setValveStates(self, valves: Union[List[Valve], Tuple[Valve]]):
        for v in valves:
            vg: ValveGUI = self.valveMappings.get(v.id, default=None)

            if vg is not None:
                vg.setValve(v)

    def resizeEvent(self, e: QResizeEvent) -> None:
        super(ValveOverview, self).resizeEvent(e)

    def reserveGuiItems(self, count: int) -> None:
        iconSize = self.ui.valveList.iconSize()

        for i in range(count):
            vg = ValveGUI(self)
            vg.resize(iconSize)

            listItem = QListWidgetItem(self.ui.valveList)
            listItem.setHidden(True)
            listItem.setSizeHint(vg.size())
            self.ui.valveList.setItemWidget(listItem, vg)


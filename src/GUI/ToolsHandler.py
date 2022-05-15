import src.GUI.Tools as tls
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QObject, QDir, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QWidget
from src.GUI.TabContainer import TabContainer
from src.GUI.Tools.constants import *
from src.Backend.Tools import calcHLSRange
import filetype
from typing import Dict, Union, List
from src.Backend.Logging import LOG
import cv2 as cv
import numpy as np
import os


DEF_TOOLS = Union[tls.HLSTester, tls.DomColorGUI]


class ToolsHandler(QObject):

    # signals
    update_top_bars_sl = pyqtSignal(str, str)

    def __init__(self, tabWidget: TabContainer, parent=None):
        super(ToolsHandler, self).__init__(parent)

        self.tabWidget = tabWidget
        self.tools: Dict[ToolEnums, DEF_TOOLS] = {}
        self.filter = ""  # "PNG file (*.png *.PNG);;JPG file (*.jpg *.JPG *.jpeg)"

        self.files = []
        self.images: List[Union[None, np.ndarray]] = []
        self.curr_directory = ""
        self.curr_file_name = ""
        self.curr_img_idx = 0

        self.tabWidget.currentChanged.connect(self.on_tab_change)
        self.tabWidget.tabCloseRequested.connect(self.on_tab_close)

    def load_images(self, tool: DEF_TOOLS):
        directory = QDir.homePath() if not self.curr_directory else self.curr_directory
        paths = QFileDialog.getOpenFileNames(self.tabWidget, caption="Select folder", directory=directory, filter=self.filter)[0]

        if any(paths):
            self.reset()

            if len(paths) > 1:
                for path in paths:
                    if os.path.isfile(path) and filetype.is_image(path):
                        self.files.append(path)

                self.curr_directory = os.path.dirname(paths[0])
            else:
                if not (os.path.isfile(paths[0]) and filetype.is_image(paths[0])):
                    LOG(f"File is not an image {paths[0]}")
                    return

                self.curr_directory = os.path.dirname(paths[0])
                self.files.append(paths[0])

                for fn in os.listdir(self.curr_directory):
                    f = os.path.join(self.curr_directory, fn)

                    if os.path.isfile(f) and filetype.is_image(f):
                        self.files.append(f)

            self.images = [None for _ in self.files]
            self.curr_img(tool)

    def set_tool_img(self, index: int, tool: DEF_TOOLS):
        ln = len(self.images)

        if ln and (ln > index) and (index >= 0):
            img = self.images[index]

            if img is None:
                img = cv.imread(self.files[index])
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                self.images[index] = img

            self.curr_img_idx = index
            self.update_top_bars()
            tool.set_img(img)
            tool.img_name = self.curr_file_name

    def next_img(self, tool: DEF_TOOLS):
        index = self.curr_img_idx + 1
        self.set_tool_img(index, tool)

    def curr_img(self, tool: DEF_TOOLS):
        self.set_tool_img(self.curr_img_idx, tool)

    def prev_img(self, tool: DEF_TOOLS):
        index = self.curr_img_idx - 1
        self.set_tool_img(index, tool)

    def update_top_bars(self):
        arrow_label = f"{self.curr_img_idx} of {len(self.images) - 1}"
        file = self.files[self.curr_img_idx]

        self.curr_file_name = os.path.basename(file)
        file_label = f"...{os.path.sep}{self.curr_file_name}"

        self.update_top_bars_sl.emit(file_label, arrow_label)

    def add_tool(self, action: QAction):
        name = action.text()
        tool_enum = ToolEnums[name]
        tool = self.tools.get(tool_enum)

        if tool is None:
            cls = getattr(tls, name, None)

            if cls is None:
                LOG(f"Unable to load tool {name}")
                return
            else:
                tool: tls.HLSTester = cls()
                self.tools[tool_enum] = tool

                tool.load_images_sl.connect(self.load_images)
                tool.next_img_sl.connect(self.next_img)
                tool.prev_img_sl.connect(self.prev_img)
                self.update_top_bars_sl.connect(tool.update_top_bars)

                if isinstance(tool, tls.HLSTester):
                    tool.auto_calc_range_sl.connect(self.auto_calc_HLS_range)

                index = self.tabWidget.addTab(tool, name)
                self.tabWidget.setCurrentIndex(index)
        else:
            index = self.tabWidget.indexOf(tool)
            self.tabWidget.setCurrentIndex(index)

    def get_curr_img(self):
        ln = len(self.images)
        index = self.curr_img_idx

        if ln and (ln > index) and (index >= 0):
            return self.images[index]

    def auto_calc_HLS_range(self, tool: tls.HLSTester):
        img = self.get_curr_img()

        if img is None:
            return

        cr = calcHLSRange(img)
        tool.set_current_values(cr[0], cr[0])

    @classmethod
    def is_tool_object(cls, widget: QWidget) -> bool:
        classes = (getattr(tls, t.name, None) for t in ToolEnums)
        classes = tuple(filter(lambda o: o is not None, classes))

        return isinstance(widget, classes)

    def on_tab_change(self, index: int):
        widget: DEF_TOOLS = self.tabWidget.currentWidget()

        if self.is_tool_object(widget):
            if widget.img_name != self.curr_file_name:
                self.curr_img(widget)

    def on_tab_close(self, index: int):
        widget: DEF_TOOLS = self.tabWidget.widget(index)

        if self.is_tool_object(widget):
            self.tools.pop(widget.tool_enum)
            self.tabWidget.removeTab(index)
            widget.deleteLater()

            if not bool(self.tools):
                self.reset()

    def reset(self):
        self.files.clear()
        self.images.clear()
        self.curr_img_idx = 0
        self.curr_file_name = ""
        self.curr_directory = ""

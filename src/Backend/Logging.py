import inspect
import os
from enum import Enum, unique
from typing import Union, Optional


@unique
class LOGType(Enum):
    WARNING = 0
    ERROR = 1
    INFO = 2


@unique
class LOGColor(Enum):
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'


def LOG(message: str, logType: LOGType = LOGType.INFO, out=print):
    caller = inspect.stack()[1]

    if logType != LOGType.INFO:
        txt = f"{LOGColor[logType.name].value}" \
              f"<file={os.path.basename(caller[1])}, line={caller[2]}, function={caller[3]}>: {message}" \
              f"{LOGColor.ENDC.value}"
    else:
        txt = f"<file={os.path.basename(caller[1])}, line={caller[2]}, function={caller[3]}>: {message}"

    out(txt)


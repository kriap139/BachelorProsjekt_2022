import numpy as np
from enum import Enum


PI_OVER_TWO = np.pi / 2
RAD_TO_DEG = 180 / np.pi


class ValveState(Enum):
    UNKNOWN = 1,
    OPEN = 2
    CLOSED = 3


class ReturnType(Enum):
    ANGLE = 1
    STATE = 2


class PipeDirection(Enum):
    TOP_BOTTOM = 0
    LEFT_RIGHT = 1

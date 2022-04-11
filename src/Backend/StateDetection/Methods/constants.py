import numpy as np
from enum import Enum, unique


PI_OVER_TWO = np.pi / 2
RAD_TO_DEG = 180 / np.pi
DEG_TO_RAD = np.pi / 180


@unique
class ValveState(Enum):
    UNKNOWN = 1,
    OPEN = 2
    CLOSED = 3


@unique
class ReturnType(Enum):
    ANGLE = 1
    STATE = 2


@unique
class PipeDirection(Enum):
    TOP_BOTTOM = 0
    LEFT_RIGHT = 1






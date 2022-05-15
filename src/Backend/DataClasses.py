import numpy as np
from typing import Iterable, Union, Tuple, List
from dataclasses import dataclass, field
from src.Backend.Valve import ValveState
import multiprocessing as mp


# Image data classes ---------------------------------------------------------------------
@dataclass
class ValveStateData:
    """Minimal class containing all necessary info for setting  ValveState"""
    valveID: str
    valveState: ValveState
    angle: float = None


@dataclass
class BBoxData:
    """Class containing info for a single bbox (A single valve)"""
    classID: int
    box: Tuple[int, int, int, int]
    tagBox: Tuple[int, int, int, int] = None
    valveID: str = None
    valveState: ValveState = ValveState.UNKNOWN
    angle: float = None


@dataclass
class SharedImage:
    """Class containing necessary info for recreating image from a shared buffer"""
    memName: str
    dType: np.dtype
    shape: tuple


@dataclass
class ImageData:
    """Class containing info for a single frame/image"""
    streamID: str
    sharedImg: SharedImage
    bboxes: List[BBoxData] = field(default_factory=list)
# -----------------------------------------------------------------------------------------


# Process Data Classes ----------------------------------------------------------------------
@dataclass
class PreStateDetectArgs:
    """Class containing necessary data to activate a PreStateDetectProcess"""
    streamPath: str = None
    streamID: str = None
    confidValveThresh: float = 0.5
    confidTagThresh: float = 0.5


@dataclass
class PreStateDetectData:
    """class containing a collection of all variables/data used by the PreStateDetectProcess"""
    mainActive: bool = True
    dfsActive: bool = False
    activateFlag: bool = False
    finishedFlag: bool = False
    shutdownFlag: bool = False
    valveModel = None
    args: PreStateDetectArgs = field(default_factory=PreStateDetectArgs)


@dataclass
class StateDetectData:
    """class containing a collection of all variables/data used by the StateDetectProcess"""
    mainActive: bool = True
    stActive: bool = False
    activateFlag: bool = False
    finishedFlag: bool = False
# ---------------------------------------------------------------------------------------------

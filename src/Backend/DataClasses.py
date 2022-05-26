import numpy as np
from typing import Iterable, Union, Tuple, List, Optional
from dataclasses import dataclass, field
from src.Backend.Valve import ValveState
import multiprocessing as mp
# from src.Backend.IDMethods import Point2D


# Image data classes ---------------------------------------------------------------------
@dataclass(frozen=True, eq=True)
class Point2D:
    """Class defining a point in 2d space"""
    x: Union[float, int]
    y: Union[float, int]

    def toTuple(self):
        return self.x, self.y

    @staticmethod
    def distance(p1: 'Point2D', p2: 'Point2D') -> float:
        """Calculates distance between two points"""
        return np.sqrt(
            (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
        )


@dataclass
class ValveStateData:
    """Minimal class containing all necessary info for setting  ValveState"""
    valveID: str
    valveState: ValveState
    angle: float = None


@dataclass(frozen=True, eq=True)
class TagData:
    """Class containing info for a tag BBox"""
    tagID: str
    tagBox: Tuple[int, int, int, int]
    tagLine: Optional[Tuple[Point2D, Point2D]] = None


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
    frameID: int
    sharedImg: SharedImage
    bboxes: List[BBoxData] = field(default_factory=list)
    tagsData: List[TagData] = field(default_factory=list)
    unassignedIndexes: Tuple[int] = field(default_factory=tuple)
# -----------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=True)
class SiftRefImg:
    img: np.ndarray
    kp: list
    des: list


# Process Data Classes ----------------------------------------------------------------------
@dataclass
class PreStateDetectArgs:
    """Class containing necessary data to activate a PreStateDetectProcess"""
    streamPath: str = None
    streamID: str = None
    tagClassID: int = 8
    confidValveThresh: float = 0.5
    confidTagThresh: float = 0.5


@dataclass
class PreStateDetectData:
    """class containing a collection of all variables/data used by the PreStateDetectProcess"""
    mainActive: bool = True
    dfsActive: bool = False
    sdActive: bool = False
    activateFlag: bool = False
    finishedFlag: bool = False
    shutdownFlag: bool = False
    valveModel = None
    vmOutputLayers: List[str] = None
    tagIdModel = None
    finishedSDFlags: List[bool] = field(default_factory=list)
    args: PreStateDetectArgs = field(default_factory=PreStateDetectArgs)


@dataclass
class StateDetectData:
    """class containing a collection of all variables/data used by the StateDetectProcess"""
    mainActive: bool = True
    stActive: bool = False
    activateFlag: bool = False
    finishedFlag: bool = False
    finishIfEmpty: bool = False
# ---------------------------------------------------------------------------------------------

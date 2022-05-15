from enum import Enum, unique
import numpy as np
from typing_extensions import Protocol
from typing import Tuple, Union
from src.Backend import ValveState
from src.Backend.Valve import Valve


@unique
class ReturnType(Enum):
    ANGLE = 1
    STATE = 2


# TypeDef classes
class TYDisplay(Protocol):
    def __call__(self, title: str, img: np.array, cmap: str = None) -> None:
        pass


class TYStateMethod(Protocol):
    def __call__(self, img: np.ndarray, bbox: Tuple[int, int, int, int], v: Valve, display: TYDisplay) \
            -> Tuple[ReturnType, Union[ValveState, float]]:
            pass


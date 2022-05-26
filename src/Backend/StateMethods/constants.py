from enum import Enum, unique
import numpy as np
from typing_extensions import Protocol
from typing import Tuple, Union
from src.Backend import ValveState
from src.Backend.Valve import Valve

Number = Union[float, int]

# TypeDef classes
class TYDisplay(Protocol):
    def __call__(self, title: str, img: np.array, cmap: str = None) -> None:
        pass


class TYStateMethod(Protocol):
    def __call__(self, img: np.ndarray, bbox: Tuple[int, int, int, int], v: Valve) -> ValveState:
            pass


import numpy as np
from typing_extensions import Protocol


# Type classes
class TYDisplay(Protocol):
    def __call__(self, title: str, img: np.array, cmap: str = None) -> None:
        pass

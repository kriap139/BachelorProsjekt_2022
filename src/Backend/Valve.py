from typing import Tuple

# "pipeDirection": "topBottom/leftRight/unset"


class Valve:

    def __init__(self, valveID: str, classID: int, className: str,  stateMethod: str, state: int,
                 colorUpper: tuple, colorLower: tuple):

        self.valveID = valveID
        self.classID = classID
        self.className = className
        self.state = state
        self.colorUpper = colorUpper
        self.colorLower = colorLower
        self.stateMethod = stateMethod



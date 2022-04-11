from typing import Tuple
from src.Backend.StateDetection.Methods import ValveState

# "pipeDirection": "topBottom/leftRight/unset"


class ValveClass:
    pass


class Valve:

    def __init__(self, valveID: str, classID: int, className: str,  stateMethod: str, colorLower: tuple,
                 colorUpper: tuple, state: ValveState = ValveState.UNKNOWN):

        self.valveID = valveID
        self.classID = classID
        self.className = className
        self.state = state
        self.colorLower = colorLower
        self.colorUpper = colorUpper
        self.stateMethod = stateMethod
        self.pipeDetectMethod = ""



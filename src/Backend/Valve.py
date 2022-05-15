from enum import Enum, unique
import datetime


@unique
class ValveState(Enum):
    UNKNOWN = 1
    OPEN = 2
    CLOSED = 3


class ValveClass:
    def __init__(self, classID: int, className: str, colorLower: tuple, colorUpper: tuple, stateMethod: str = "sift"):
        self.classID = classID
        self.className = className
        self.colorLower = colorLower
        self.colorUpper = colorUpper
        self.stateMethod = stateMethod

    def __str__(self) -> str:
        return f"ValveClass({self.className}(id={self.classID}))"

    def infoLabelString(self) -> str:
        return f"{self.className}(id={self.classID})"


class Valve:
    def __init__(self, valveID: str, valveClass: ValveClass, state: ValveState = ValveState.UNKNOWN, angle: float = None):
        self.id = valveID
        self.cls = valveClass
        self.state = state
        self.angle = angle
        self.lastUpdated = datetime.datetime.today()

    def setState(self, state: ValveState):
        self.state = state
        self.lastUpdated = datetime.datetime.today()

    def infoLabelStr(self) -> str:
        return (f"ID:  {self.id}\n"
                f"ValveClass:  {self.cls.infoLabelString() if self.cls else None}\n"
                f"Last updated:  {self.lastUpdated.strftime('%c')}\n"
                f"State:  {self.state.name}\n"
                f"Angle: {' -' if self.angle is None else self.angle}")


from enum import Enum

class ValveType(Enum):
    ArrowValve = 0


class Valve:

    def __init__(self, v_type: ValveType, v_id: str = None):
        pass

    def detect_state(self):
        pass

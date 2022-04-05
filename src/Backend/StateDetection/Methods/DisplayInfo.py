from src.Backend.Valve import Valve


class DisplayInfo:
    def __init__(self, method: str, valve: Valve):
        self.method = method
        self.valve = valve

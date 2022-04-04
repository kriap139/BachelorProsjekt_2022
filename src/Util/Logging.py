import inspect
import os


class Logging:

    @classmethod
    def print(cls, message: str):
        caller = inspect.stack()[1]
        print(f"[file={os.path.basename(caller[1])}, line={caller[2]}, function={caller[3]}]: {message}")

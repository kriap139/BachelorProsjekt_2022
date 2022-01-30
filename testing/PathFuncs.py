import os
import platform
from pathlib import Path


def dirUp(path: str, n: int) -> str:
    sep = '\\' if path.__contains__('\\') else '/'

    for _ in range(n):
        path = dirUp(path.rpartition(sep)[0], 0)
    return path


def pathAppend(path: str, add: str) -> str:

    sep, rep = ('\\', '/') if path.__contains__('\\') else ('/', '\\')
    add = add.replace(rep, sep)

    if path.endswith(sep):
        if add.startswith(sep):
            add = add[1:]
    else:
        if not add.startswith(sep):
            add = "".join((sep, add))

    return path.__add__(add)


def fileDir(file) -> str:
    return os.path.dirname(os.path.abspath(file))


def dataDir(add: str = None) -> str:
    system = platform.system()

    if system == "Linux":
        relPath = "/School/OneDrive-Østfold-University-College/Bachelor/Bachelor_2022_shared"
    elif system == "Windows":
        relPath = "/Bachelor_2022_shared"
    else:
        raise NotImplementedError(f"OS: {system} is not supported")

    dDir = pathAppend(str(Path.home()), add=relPath)
    dDir = pathAppend(dDir, add)

    if not os.path.exists(dDir):
        raise NotADirectoryError(f"Data Directory: [{dDir}], doesn't exist")
    else:
        print(f"Data Directory: {dDir}")

    return dDir

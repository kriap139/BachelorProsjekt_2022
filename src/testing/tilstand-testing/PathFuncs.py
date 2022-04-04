import os
import platform
from pathlib import Path


def dirUp(path: str, n: int) -> str:
    sep = '\\' if path.__contains__('\\') else '/'

    for _ in range(n):
        path = dirUp(path.rpartition(sep)[0], 0)
    return path


def fileDir(file) -> str:
    return os.path.dirname(os.path.abspath(file))


def createDirPath(path: str) -> str:
    """Creates the folder structure specified by the path argument"""

    path = os.path.abspath(path)

    if os.path.exists(path):
        return path

    dirs = []
    sep = os.path.sep

    while not os.path.exists(path):
        part = path.rpartition(sep)
        path = part[0]
        dirs.append(part[2])

        if not part[1].strip():
            break

    if not os.path.exists(path):
        raise Exception(f"Path doesn't exist")

    dirs.reverse()
    for dm in dirs:
        os.makedirs()


    return path

def dataDir(add: str = None) -> str:
    system = platform.system()

    sysRels = {
        "Windows": [
            "\\Østfold University College\\Kristoffer Pinås - Bachelor_2022_shared\\"
        ],
        "Linux": [
            "/School/OneDrive-Østfold-fylkeskommune/Bachelor/Bachelor_2022_shared"
        ]
    }

    if system in sysRels.keys():
        home  = str(Path.home())
        aDir = ''
        
        for relPath in sysRels.get(system):
            dDir = pathAppend(home, add=relPathm, sep='/')
            
            if os.path.exists(dDir):
                break
        
        if not os.path.exists(aDir):
           raise NotADirectoryError(f"Data Directory: [{dDir}], doesn't exist")
    else:
        raise NotImplementedError(f"OS: {system} is not supported")

    dDir = pathAppend(dDir, add, sep='/')

    if not os.path.exists(dDir):
        raise NotADirectoryError(f"Data Directory Path: [{dDir}], doesn't exist")
    else:
        print(f"Data Directory: {dDir}")

    return dDir

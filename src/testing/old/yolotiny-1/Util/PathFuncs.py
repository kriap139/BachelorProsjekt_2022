import os
import platform
from pathlib import Path


def dirUp(path: str, n: int) -> str:
    sep = '\\' if path.__contains__('\\') else '/'

    for _ in range(n):
        path = dirUp(path.rpartition(sep)[0], 0)
    return path


def pathAppend(path: str, add: str, sep = None) -> str:

    if sep is None:
        sep, rep = ('\\', '/') if path.__contains__('\\') else ('/', '\\')
    else:
        rep = ''
        if sep == '/':
            rep = '\\'
        elif sep == '\\':
            rep = '/'
        path.replace(rep, sep)
    
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
            dDir = pathAppend(home, add=relPath, sep='\\')
            
            if os.path.exists(dDir):
                break
        
#        if not os.path.exists(aDir):
#           raise NotADirectoryError(f"Data Directory: [{dDir}], doesn't exist")
    else:
        raise NotImplementedError(f"OS: {system} is not supported")

    dDir = pathAppend(dDir, add, sep='\\')

    if not os.path.exists(dDir):
        raise NotADirectoryError(f"Data Directory Path: [{dDir}], doesn't exist")
    else:
        print(f"Data Directory: {dDir}")

    return dDir

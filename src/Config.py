import json
import os
from src.Backend.Logging import LOG, LOGType
from src.Backend.Valve import ValveState, ValveClass, Valve
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import schema as sh


REQUIRED_PACKAGES = [
    "PyQt6",
    "Pillow",
    "schema",
    "numpy",
    "opencv-python",
    "tensorflow",
    "imutils",
    "scikit-learn"
]


@dataclass
class ModelPaths:
    valveWeightsSrc: str
    valveCfgSrc: str
    tagIdCNNPath: str


@dataclass
class ValveData:
    valveClasses: Dict[int, ValveClass] = field(default_factory=dict)
    valves: Dict[str, Valve] = field(default_factory=dict)


class Config:
    APP_DATA_PATH = os.path.abspath("resources/")
    SEP_REP = '\\' if (os.path.sep == '/') else '/'

    appConfigSchema = sh.Schema(
        {
            "GUI": {
                sh.Optional("theme"): sh.Or(str, None),
                sh.Optional("windowGeometry"): [int, int, int, int],
                "valveImages": {
                    "UNKNOWN": sh.And(str, len),
                    "OPEN": sh.And(str, len),
                    "CLOSED": sh.And(str, len)
                }
            }
        }
    )

    valveInfoSchema = {
        "valves": [
            sh.Optional(
                {
                    "id": sh.And(str, len),
                    "cls": int
                }
            )
        ],
        "valveClasses": [
            {
                "classID": int,
                "className": sh.And(str, len),
                "colorLower": [int, int, int],
                "colorUpper": [int, int, int],
                "stateMethod": sh.Or(sh.And(str, len), None)
            }
        ]
    }

    @classmethod
    def createAppDataPath(cls, *dirPath, fName=None) -> str:
        if not os.path.exists(cls.APP_DATA_PATH):
            raise NotADirectoryError(f"appdata folder not found: {cls.APP_DATA_PATH}")

        path = os.path.join(cls.APP_DATA_PATH, *dirPath)
        os.makedirs(path, exist_ok=True)

        return str(path) if (fName is None) else str(os.path.join(path, fName))

    @classmethod
    def getAppConfigPath(cls) -> str:
        cp = cls.createAppDataPath(fName="app_settings.json")

        if not os.path.exists(cp):
            raise FileNotFoundError(f"Config file doesn't exist: {cp}")

        return cp

    @classmethod
    def loadAppConfFile(cls) -> dict:
        path = cls.getAppConfigPath()

        with open(path, mode='r') as f:
            data = json.load(f)

        return cls.appConfigSchema.validate(data)

    @classmethod
    def saveAppConfFile(cls, data):
        path = cls.getAppConfigPath()

        with open(path, mode='w') as f:
            json.dump(data, f, indent=3)

    @classmethod
    def getSiftRefsDir(cls) -> str:
        siftImagesPath = cls.createAppDataPath("images", "sift", "cropped")

        if not os.path.exists(siftImagesPath):
            LOG(f"siftImagesPath doesn't exist: {siftImagesPath}")
            exit(-1)

        return siftImagesPath

    @classmethod
    def loadValveInfoData(cls) -> ValveData:
        valveInfoPath = cls.createAppDataPath(fName="valve_info.json")

        if not os.path.exists(valveInfoPath):
            LOG(f"Valve info file doesn't exist: {valveInfoPath}", LOGType.ERROR)
            exit(-1)

        with open(valveInfoPath, mode='r') as f:
            data = json.load(f)

        data = sh.Schema(cls.valveInfoSchema).validate(data)

        valves = {d["id"]: Valve(**d) for d in data["valves"]}
        valveClasses = {d["classID"]: ValveClass(**d) for d in data["valveClasses"]}

        return ValveData(valveClasses, valves)

    @classmethod
    def getModelPaths(cls) -> ModelPaths:
        valveWeightsPath = cls.createAppDataPath("model", "classif", fName="yolo_Tiny_ventiler_best9.weights")
        valveCfgPath = cls.createAppDataPath("model", "classif", fName="yolo_Tiny_ventiler.cfg")
        tagIdCNNPath = cls.createAppDataPath("model", "tagid", fName="CNN_Characters_Classification.h5")

        pathData = (
            ("valveWeightsPath", valveWeightsPath),
            ("valveCfgPath", valveCfgPath),
            ("tagIdCNNPath", tagIdCNNPath)
        )

        terminate = False

        for var, path in pathData:
            if not os.path.exists(path):
                LOG(f"{var} doesn't exist: {path}", LOGType.ERROR)
                terminate = True

        if terminate:
            exit(-1)

        return ModelPaths(valveWeightsPath, valveCfgPath, tagIdCNNPath)

    @classmethod
    def loadGUIData(cls, appdata=None):
        if appdata is None:
            appdata = cls.loadAppConfFile()

        appData: dict = cls.loadAppConfFile()["GUI"]

        theme = appData.get("theme")
        themePath = ""

        if theme is None:
            LOG("No theme specified")
        else:
            themePath = cls.createAppDataPath("themes", theme, fName=f"{theme}.qss")

        if not os.path.exists(themePath):
            LOG(f"Theme path doesn't exist: {themePath}", LOGType.WARNING)

        valveImgPaths = {}

        for state, p in appData["valveImages"].items():

            if not p.__contains__(os.path.sep):
                p = p.replace(cls.SEP_REP, os.path.sep)

            parts = p.split(os.path.sep)
            fn = parts.pop()

            path = cls.createAppDataPath(*parts, fName=fn)
            valveImgPaths[ValveState[state]] = path

            if not os.path.exists(path):
                LOG(f"Valve image path doesn't exist: {path}", LOGType.WARNING)

        windowGeom = appData.get("windowGeometry")

        return themePath, valveImgPaths, windowGeom

    @classmethod
    def saveWindowGeometry(cls, x: int, y: int, w: int, h: int):
        data = cls.loadAppConfFile()
        data["GUI"]["windowGeometry"] = [x, y, w, h]
        cls.saveAppConfFile(data)


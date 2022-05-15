import imutils
from typing import Union
import numpy as np
import cv2 as cv
from src import Config
from typing import Tuple, NamedTuple, List
from dataclasses import dataclass
from enum import Enum
from sklearn.cluster import KMeans


class RGB(NamedTuple):
    r: int
    g: int
    b: int


class BGR(NamedTuple):
    b: int
    g: int
    r: int


class HLS(NamedTuple):
    h: int
    l: int
    s: int


class HSV(NamedTuple):
    h: int
    s: int
    v: int


@dataclass
class DomColor:
    pct: float
    color: Union[RGB, BGR, HLS, HSV]


@dataclass
class DomColorRGB:
    pct: float
    color: RGB


@dataclass
class DomColorBGR:
    pct: float
    color: BGR


@dataclass
class DomColorHLS:
    pct: float
    color: HLS


@dataclass
class DomColorHSV:
    pct: float
    color: HSV


@dataclass
class RangeHSV:
    lower: HSV
    upper: HSV


class ColorFormat(Enum):
    RGB = 0
    BGR = 1
    HLS = 2
    HSV = 3


ColorData = tuple[Tuple[float, tuple[int, int, int]], ...]

FORMATS_RGB2 = {
        ColorFormat.BGR: cv.COLOR_RGB2BGR,
        ColorFormat.HLS: cv.COLOR_RGB2HLS,
        ColorFormat.HSV: cv.COLOR_RGB2HSV
}

FORMATS_2RGB = {
    ColorFormat.BGR: cv.COLOR_BGR2RGB,
    ColorFormat.HLS: cv.COLOR_HLS2RGB,
    ColorFormat.HSV: cv.COLOR_HSV2RGB
}


def cvColorFromRGB(img: np.ndarray, colorFormat: ColorFormat) -> np.ndarray:
    format_code = FORMATS_RGB2.get(colorFormat)

    if format_code is None:
        raise ValueError(f"Unsupported Color format {colorFormat}")

    return cv.cvtColor(img, format_code)


def cvColorToRGB(img, colorFormat: ColorFormat) -> np.ndarray:
    format_code = FORMATS_2RGB.get(colorFormat)

    if format_code is None:
        raise ValueError(f"Unsupported Color format {colorFormat}")

    return cv.cvtColor(img, format_code)


def cvPixelFromRGB(pixel: Union[np.ndarray, List, Tuple], colorFormat: ColorFormat) -> np.ndarray:
    return cvColorFromRGB(np.uint8([[pixel]]), colorFormat)[0][0]


def cvPixelToRGB(pixel: Union[np.ndarray, List, Tuple], colorFormat: ColorFormat) -> np.ndarray:
    return cvColorToRGB(np.uint8([[pixel]]), colorFormat)[0][0]


def colorDataToRGB(colorData: Tuple[DomColor, ...], fromFormat: ColorFormat) -> ColorData:
    format_code = FORMATS_2RGB.get(fromFormat)

    if format_code is None:
        raise ValueError(f"Unsupported Color format {fromFormat}")

    return tuple((pct, cv.cvtColor(np.uint8([[color]]), format_code)[0][0]) for pct, color in colorData)


def colorDataFromRGB(colorData: Tuple[DomColorRGB, ...], toFormat: ColorFormat) -> ColorData:
    format_code = FORMATS_2RGB.get(toFormat)

    if format_code is None:
        raise ValueError(f"Unsupported Color format {toFormat}")

    return tuple((pct, cv.cvtColor(np.uint8([[color]]), format_code)[0][0]) for pct, color in colorData)


def findDomColors(src: Union[str, np.ndarray],
                  clusters: int = 4,
                  useFormat: ColorFormat = ColorFormat.RGB) -> ColorData:
    """Find dominant color in src Image, Image format has to be in RGB,"""

    clusters = max(1, clusters)

    if isinstance(src, np.ndarray):
        og_img = src
    elif isinstance(src, str):
        og_img = cv.imread(src)
        og_img = cv.cvtColor(og_img, cv.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported argument {type(src)}")

    if useFormat != ColorFormat.RGB:
        og_img = cvColorFromRGB(og_img, useFormat)

    img = imutils.resize(og_img, height=200)
    flat = np.reshape(img, (-1, 3))

    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(flat)

    dom_colors = np.array(kmeans.cluster_centers_, dtype='uint')
    pct = np.unique(kmeans.labels_, return_counts=True)[1] / flat.shape[0]
    result = sorted(zip(pct, dom_colors), reverse=True)

    result = ((pct, tuple(color)) for pct, color in result)
    return tuple(result)


def calcHLSRange(img: np.ndarray, clusters=4, useFormat=ColorFormat.HLS) -> Tuple[HLS, HLS]:

    colorData = findDomColors(img, clusters, useFormat=useFormat)
    ln = len(colorData)

    if ln > 1:
        i = 1 if (ln > 2) else 2

        pct1, c1 = colorData[0][0], colorData[1][1]
        pct2, c2 = colorData[i][0], colorData[i][1]

        lower = int(min(c1[0], c2[0])), int(min(c1[1], c2[1])), int(min(c1[2], c2[2]))
        upper = int(max(c1[0], c2[0])), 255, 255

        return HLS(*lower), HLS(*upper)
    elif ln == 1:
        c = colorData[0][1]

        lr = int(c[0]), int(c[1]), int(c[2])
        ur = int(c[0]), 255, 255

        return HLS(*lr), HLS(*ur)
    else:
        return HLS(0, 0, 0), HLS(255, 255, 255)


if __name__ == '__main__':
    p = Config.createAppDataPath("testing", "tag", "old_tags", fName="2.jpg")
    res = findDomColors(p, 4, ColorFormat.RGB)
    print(res[0])


import cv2 as cv2
import numpy as np

__all__ = ['filterContours', 'getExtremaOfContours', 'sortContours']


def filterContours(conts, requiredLength=19, filterThreshold=0.16):
    """fiilters contours based on their arcLength. Contours Smaller then the requiredLength(in pixels)
    parameter gets removed. Additionally any Contour that are to small commpared to the biggest countour in the list,
    will also be removed. The cantors minimum length (compared to the longest in the list),
    is given by the filterThreshold param in precent[0-1]"""

    if not conts.__len__():
        return tuple()

    lengths = tuple(cv2.arcLength(cont, True) for cont in conts)

    longest = np.max(lengths)
    filtered = []

    for i in range(conts.__len__()):
        if lengths[i] > requiredLength and (lengths[i]/longest > filterThreshold):
            filtered.append(conts[i])

    return filtered


def getExtremaOfContours(conts):
    """Creates a list of tuples that contains the extrema cordinated of each contour(conts :param).

    Example of a contour extrema tuple ---> tuple(top: (x, y), left: (x, y), bottom: (x, y), right: (x, y)), where
    top has the smallest Y value, left has the smallest X value, bottom has the biggest Y value and right has the
    biggest X value.

    :returns tuple((top,left, bottom, right), (top,left, bottom, right), ...) , where tuple[0] is extrema conts[0], ...
    """

    extrema = []
    for c in conts:
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBottom = tuple(c[c[:, :, 1].argmax()][0])

        extrema.append((extTop, extLeft, extBottom, extRight))
    return extrema


def sortContours(conts, extrema=None, order='RTL', then=None, keepAll=False, unzip=True):
    """ Sorts a list of Contours(conts param) based on the order parameter.

    Valid order Values: [RTL: (Right To Left), LTR, TTB: (Top To Bottom), BTT).

     The then param,  is as a tiebreaker for situations where the position of
     multiple Contours are considered equal in the specified order.

    Valid 'then' Values: Same as order Param """

    if order == 'RTL':
        i = 3           # (top: 0, left: 1, bottom: 2, right: 3)
        j = 0           # (x: 0, y: 1)
        rev = True      # True if RightToLeft (Biggest x first) or TopToBottom (biggest y first)
    elif order == 'LTR':
        i = 1
        j = 0
        rev = False
    elif order == 'TTB':
        i = 0
        j = 1
        rev = False
    elif order == 'BTT':
        i = 2
        j = 1
        rev = True
    else:
        raise TypeError('\'order\' Parameter has to be one of these string values (RTL, LTR, TTB, BTT)')

    if extrema is None:
        extrema = getExtremaOfContours(conts)

    sortedConts = list(zip(conts, extrema))                     # [(cont 0, extrema 0), (cont 1 extrema 1), etc.]
    sortedConts.sort(key=lambda tup: tup[1][i][j], reverse=rev)

    if then is not None:
        compVal = sortedConts[0][1][i][j]
        newList = [o for o in sortedConts if o[1][i][j] == compVal]

        if then == 'RTL':
            newList.sort(key=lambda tup: tup[1][i][0], reverse=True)
        elif then == 'LTR':
            newList.sort(key=lambda tup: tup[1][i][0], reverse=False)
        elif then == 'TTB':
            newList.sort(key=lambda tup: tup[1][i][1], reverse=False)
        elif then == 'BTT':
            newList.sort(key=lambda tup: tup[1][i][1], reverse=True)

        if keepAll:
            for i in range(newList.__len__(), sortedConts.__len__()):
                newList.append(sortedConts[i])

        sortedConts = newList

    return tuple(zip(*sortedConts))

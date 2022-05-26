from src.Backend.DataClasses import Point2D
import numpy as np
from typing import Tuple, List, Union

# Code modified from: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/


def onSegment(p: Point2D, q: Point2D, r: Point2D) -> bool:
    """Given three collinear points p, q, r, the function checks if
    point q lies on line segment 'pr'"""

    if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False


def orientation(p: Point2D, q: Point2D, r: Point2D) -> int:
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:

    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))

    if val > 0:
        # Clockwise orientation
        return 1

    elif val < 0:
        # Counterclockwise orientation
        return 2

    else:
        # Collinear orientation
        return 0


def intersect(p1: Point2D, q1: Point2D, p2: Point2D, q2: Point2D) -> bool:
    """Checks if the line segment 'p1q1' and 'p2q2' intersect."""

    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and onSegment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and onSegment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and onSegment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0) and onSegment(p2, q1, q2):
        return True

    # If none of the cases
    return False


BOX_DEF = Tuple[int, int, int, int]


def checkForEdgeIntersections(img: np.ndarray, boxes: Tuple[BOX_DEF]) -> List[bool]:
    """checks if bboxes intersects  image edge"""
    fh, fw, _ = img.shape

    frameTL = Point2D(0, 0)
    frameTR = Point2D(fw, fh)
    frameBL = Point2D(fw, fh)
    frameBR = Point2D(fw, fh)

    frameSegments = (frameTL, frameBL), (frameBL, frameBR), (frameBR, frameTR), (frameTR, frameTL)

    result = []

    for box in boxes:
        x, y, w, h = box

        tl = Point2D(x, y)
        tr = Point2D(x + w, y)
        bl = Point2D(x, y + h)
        br = Point2D(x + w, y + h)

        segments = (tl, bl), (bl, br), (br, tr), (tr, tl)
        intersecting = False

        for fSeg in frameSegments:
            res = tuple(intersect(fSeg[0], fSeg[1], bp[0], bp[1]) for bp in segments)

            if any(res):
                intersecting = True
                break

        result.append(intersecting)
    return result

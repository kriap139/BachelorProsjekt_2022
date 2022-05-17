import cv2
import numpy as np
from typing import Iterable, Tuple, List, Union, Optional
from src.Backend.DataClasses import BBoxData
from dataclasses import dataclass


@dataclass
class Point2D:
    x: Union[float, int]
    y: Union[float, int]

    def toTuple(self):
        return self.x, self.y

    @staticmethod
    def distance(p1: 'Point2D', p2: 'Point2D') -> float:
        """Calculates distance between two points"""
        return np.sqrt(
            (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
        )


def calcTagLine(frame: np.ndarray, tagBox: tuple) -> Tuple[Point2D, Point2D]:
    """Create a symmetry line from center of tag to edge of image (along the shortest part of the tag)"""

    x, y, w, h = tagBox
    wh, hh = int(w * 0.5), int(h * 0.5)
    cx, cy = x + wh, y + hh

    # find the smallest Symmetry line to tagBox
    if w > h:
        p1 = (x + wh, y)
        p2 = (x + wh, y + h)
    else:
        p1 = (x, y + hh)
        p2 = (x + w, y + hh)

    # find normalized heading vector
    vec = np.array((p1[0] - p2[0], p1[1] - p2[1]))
    vec = vec / np.linalg.norm(vec)

    # calculate angle and max distance (frame diagonal)
    angle = np.arctan2(vec[1], vec[0])

    # length= w + h, du to Triangle inequality the line has to be shorter than this value
    length = frame.shape[0] + frame.shape[1]

    x1 = int(cx + length * np.cos(angle))
    y1 = int(cy + length * np.sin(angle))

    x2 = int(cx - length * np.cos(angle))
    y2 = int(cy - length * np.sin(angle))

    # clip line to fit image
    success, p1, p2 = cv2.clipLine((0, 0, frame.shape[1], frame.shape[0]), (x1, y1), (x2, y2))

    # print(f"frame=({frame.shape[0]}, {frame.shape[1]}), angle={np.degrees(angle)}, "
    #      f"(x1={x1}, y1={y1}), (x2={x2}, y2={y2}),   p1={p1}, p2={p2}")

    return Point2D(p1[0], p1[1]), Point2D(p2[0], p2[1])


# lineIntersection ----------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------------------------------


def assignTagsToBBoxes(frame: np.ndarray, tagBoxes: Tuple[Tuple[int, int, int, int], ...],
                       bboxes: List[BBoxData]) -> Tuple[Tuple[Tuple[Point2D, Point2D]], Tuple[tuple]]:

    tagLines = tuple(calcTagLine(frame, tag) for tag in tagBoxes)
    bbSegments = []

    for bbd in bboxes:
        x, y, w, h = bbd.box
        tl = Point2D(x, y)
        tr = Point2D(x + w, y)
        bl = Point2D(x, y + h)
        br = Point2D(x + w, y + h)
        # line segments for rectangle: (tl, bl), (bl, br), (br, tr), (tr, tl)
        segments = (tl, bl), (bl, br), (br, tr), (tr, tl)
        bbSegments.append(segments)

    for i, tagBox in enumerate(tagBoxes):
        tp1, tp2 = tagLines[i]

        # [(index, bbox), ...]
        candidates = []

        for j, bbd in enumerate(bboxes):
            segments = bbSegments[j]
            res = tuple(intersect(tp1, tp2, bp[0], bp[1]) for bp in segments)

            if any(res):
                candidates.append((j, bbd))

        tx, ty, tw, th = tagBox

        if tw > th:
            tagCord = ty
            cordIdx = 1
        else:
            tagCord = tx
            cordIdx = 0

        if len(candidates) == 1:
            bbd = candidates[0][1]
            bbd.tagBox = tagBox
        elif len(candidates) > 0:
            # Handling of multiple BBoxes that intersects tagLine
            bCords = [tup[1].box[cordIdx] for tup in candidates]
            index = bCords.index(min(bCords))

            if bCords[index] < tagCord:
                # BBox lies over tagBox, so find the closest BBox that still lies underneath tagBoz
                bCords = np.array(bCords)
                sortIdx = np.argsort(bCords)

                for k in sortIdx:
                    if bCords[k] > tagCord:
                        candidates[k][1].tagBox = tagBox
            else:
                bbd = candidates[index][1]
                bbd.tagBox = tagBox

    assigned = tuple(bbd.tagBox for bbd in bboxes)
    unassigned = tuple(set(tagBoxes) - set(assigned))

    return tagLines, unassigned

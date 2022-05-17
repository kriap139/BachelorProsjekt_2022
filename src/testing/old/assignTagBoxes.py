import cv2
import numpy as np
from typing import Iterable, Tuple, List, Union, Optional
from src.Backend.DataClasses import BBoxData
from dataclasses import dataclass
from enum import Enum


@dataclass
class Point2D:
    x: Union[float, int]
    y: Union[float, int]

    def toTuple(self):
        return self.x, self.y


def calcTagLine(frame: np.ndarray, tagBox: tuple) -> Tuple[Point2D, Point2D]:
    """
    old args: xmin, ymin, xmax, ymax, x1, y1, x2, y2

    Extend a line so that it reaches the walls of the bbox.

    Args:
        xmin(int): The very left coordinate of the bbox.
        ymin(int): The very top coordinate of the bbox.
        xmax(int): The very right coordinate of the bbox.
        ymax(int): The very bottom coordinate of the bbox.
        x1(int): The start x coordinate of the line.
        y1(int): The start y coordinate of the line.
        x2(int): The end x coordinate of the line.
        y2(int): The end y coordinate of the line.

    Returns:
        - (list): The start and end (x, y) coordinates of the extended line.
    """


    # If we imagine extending the line until it crosses the top wall of the
    # bbox at point `(xmin, y_for_xmin)` and then imagine drawing
    # perpendicular lines from each point `(x1, y1)`, `(x2, y2)` to the wall
    # of the bbox, we end up with 2 perpendicular trianlges with the same
    # angles - similar triangles. The rule of the similar triangles is that
    # the side lengths of two similar triangles are proportional.
    # That's how we get the equal ratios:
    # `| y_for_xmin - y1 | / | xmin - x1 | == | y2 - y1 | / | x2 - x1 |`
    # After we move some numbers from one to the other side of this equation,
    # we get the value for `y_for_xmin`. That's where the line should cross
    # the top wall of the bbox. We do the same for all other coordinates.
    # NOTE: These calculations are valid if one starts to draw a line from top
    # to botton and from left to right. In case the direction is reverted, we
    # need to switch the min and max for each point (x, y). We do that below.

    x, y, w, h = tagBox
    wh, hh = int(w * 0.5), int(h * 0.5)

    # find Symmetry line
    if w > h:
        p1 = (x + wh, y)
        p2 = (x + wh, y + h)
    else:
        p1 = (x, y + hh)
        p2 = (x + w, y + hh)

    xmin, ymin = 0, 0
    xmax, ymax = frame.shape[0], frame.shape[1]
    x1, y1 = p1
    x2, y2 = p2

    # ----------------------------------------------------------------

    d1 = (x2 - x1)
    d2 = (y2 - y1)

    if d1 == 0:
        d1 = 1
    if d2 == 0:
        d2 = 1

    y_for_xmin = y1 + (y2 - y1) * (xmin - x1) / d1
    y_for_xmax = y1 + (y2 - y1) * (xmax - x1) / d1
    x_for_ymin = x1 + (x2 - x1) * (ymin - y1) / d2
    x_for_ymax = x1 + (x2 - x1) * (ymax - y1) / d2

    # The line is vertical
    if (x2 - x1) < (y2 - y1):
        # The line is drawn from right to left
        if x1 > x2:
            # Switch the min and max x coordinates for y,
            # because the direction is from right (min) to left (max)
            y_for_xmin, y_for_xmax = y_for_xmax, y_for_xmin
    # The line is horizontal
    else:
        # The line is drawn from bottom to top
        if y1 > y2:
            # Switch the min and max y coordinates for x,
            # because the direction is from bottom (min) to top (max)
            x_for_ymin, x_for_ymax = x_for_ymax, x_for_ymin

    # The line is drawn from right to left
    if x1 > x2:
        # Get the maximal value for x1.
        # When `x_for_ymin < xmin`(line goes out of the
        # bbox from the top), we clamp to xmin.
        x1 = max(max(int(x_for_ymin), xmin), x1)
    # The line is drawn from left to right
    else:
        # Get the minimal value for x1.
        # When `x_for_ymin < xmin`(line goes out of the
        # bbox from the top), we clamp to xmin.
        x1 = min(max(int(x_for_ymin), xmin), x1)

    # Get the maximal value for x2.
    # When `x_for_ymax > xmax` (line goes out of the
    # bbox from the bottom), we clamp to xmax.
    x2 = max(min(int(x_for_ymax), xmax), x2)

    # Get the minimal value for y1
    # When `y_for_xmin < ymin`(line goes out of the
    # bbox from the left), we clamp to ymin.
    y1 = min(max(int(y_for_xmin), ymin), ymax)

    # Get the minimal value for y2
    y2 = min(int(y_for_xmax), ymax)

    return Point2D(x1, y1), Point2D(x2, y2)


# lineIntersection -------------------------------------------------------------
class Point2D:
    def __init__(self, x: Union[float, int], y: Union[float, int]):
        self.x = x
        self.y = y


def distance(a: Point2D, b: Point2D) -> float:
    return np.sqrt((b.x - a.x)**2 + (b.y - a.y)**2)


def subPoints(a: Point2D, b: Point2D) -> Point2D:
    return Point2D(a.x - b.x, a.y - b.y)


def addPoints(a: Point2D, b: Point2D) -> Point2D:
    return Point2D(a.x + b.x, a.y + b.y)


def dot(a: Point2D, b: Point2D):
    return a.x * b.x + a.y * b.y


def cross(a: Point2D, b: Point2D):
    return a.x * b.y - a.y * b.x


epsilon = 1 / 1000000


def eq0(num: Union[float, int]):
    return np.abs(num) < epsilon


class LineSegment:
    def __init__(self, start: Point2D, end: Point2D):
        self.start = start
        self.end = end

        self.len = distance(start, end)

        if self.len == 0:
            raise ArithmeticError("Invalid length")

        self.direction = Point2D(end.x - start.x, end.y - start.y)


class InterSectionTypes(Enum):
    COLLINEAR_DISJOINT = 0
    COLLINEAR_OVERLAPPING = 1
    PARALLEL_NON_INTERSECTING = 2
    INTERSECTION = 3
    NO_INTERSECTION = 4


def intersection(ls1: LineSegment, ls2: LineSegment) -> Optional[Point2D]:
    p, r = ls1.start, ls1.direction
    q, s = ls2.start, ls2.direction

    # r x s
    rs = cross(r, s)

    # (q - p) x r
    qpr = cross(subPoints(q, p), r)

    if eq0(rs) and eq0(qpr):
        # t0 = (q − p) · r / (r · r)
        t1 = dot(addPoints(q, subPoints(s, p)), r) / dot(r, r)
        t0 = t1 - dot(s, r) / dot(r, r)

        if ((t0 >= 0) and (t0 <= 1)) or ((t1 >= 0) and (t1 <= 1)):
            return Point2D(t0, t1)
        else:
            return

    elif eq0(rs) and not eq0(qpr):
        return

    else:
        # t = (q − p) × s / (r × s)
        t = cross(subPoints(q, p), s) / cross(r, s)

        # u = (q − p) × r / (r × s)
        u = cross(subPoints(q, p), r) / cross(r, s)

        if (not eq0(rs)) and (t >= 0) and (t <= 1) and (u >= 0) and (u <= 1):
            return Point2D(t, u)

# ------------------------------------------------------------------------------


def assignTagsToBBoxes(frame: np.ndarray, tagBoxes: Tuple[Tuple[int, int, int, int], ...],
                       bboxes: List[BBoxData], draw: bool = False) -> None:
    if not any(tagBoxes):
        return

    for tagBox in tagBoxes:
        tx, ty, tw, th = tagBox
        tcx, tcy = tx + int(tw * 0.5), ty + int(th * 0.5)

        tp1, tp2 = calcTagLine(frame, tagBox)
        ls1 = LineSegment(Point2D(tp1[0], tp2[0]), Point2D(tp2[0], tp2[1]))

        if draw:
            cv2.line(frame, tp1, tp2, (0, 255, 0), 3)

        # [(bbox, pointOfIntersect), ...]
        candidates = []

        for bbox in bboxes:
            x, y, w, h = bbox.box

            tl = Point2D(x, y)
            tr = Point2D(x + w, y)
            bl = Point2D(x, y + h)
            br = Point2D(x + w, y + h)

            segments = (
                LineSegment(tl, bl),
                LineSegment(bl, br),
                LineSegment(br, tr),
                LineSegment(tr, tl)
            )

            res = tuple(intersection(ls1, ls2) for ls2 in segments)

            print(res)

            if any(res):
                intersects = tuple(filter(lambda obj: obj is not None, res))

                if len(intersects) == 1:
                    candidates.append((bbox, intersects[0]))
                else:
                    dists = [np.sqrt((tcx - p.x)**2, (tcy - p.y)**2) for p in intersects]
                    intersect = intersects[res.index(min(dists))]
                    candidates.append((bbox, intersect))

        if len(candidates) == 1:
            bbd: BBoxData = candidates[0][0]
            bbd.tagBox = tagBox
        elif len(candidates) > 1:
            dists = []
            for bbd, intersect in candidates:
                dists.append(np.sqrt((tcx - intersect.x) ** 2, (tcy - intersect.y) ** 2))

            index = dists.index(min(dists))
            bbd: BBoxData = candidates[index]
            bbd.tagBox = tagBox




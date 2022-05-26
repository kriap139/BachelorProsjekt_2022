import cv2
import numpy as np
from typing import Iterable, Tuple, List, Union, Optional
from src.Backend.DataClasses import BBoxData, Point2D, TagData
from dataclasses import dataclass, field
from src.Backend.IDMethods.Tag.TagID import detectTagID, createCNNModel
from src.Backend.Tools import intersect


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


def assignTagsToBBoxes(tagDatas: List[TagData], bboxes: Iterable[BBoxData]) -> Tuple[int]:
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

    assigned: List[int] = []

    for i, td in enumerate(tagDatas):
        tp1, tp2 = td.tagLine
        tx, ty, tw, th = td.tagBox

        # [(index, bbox), ...]
        candidates = []

        for j, bbd in enumerate(bboxes):
            segments = bbSegments[j]
            res = tuple(intersect(tp1, tp2, bp[0], bp[1]) for bp in segments)

            if any(res):
                candidates.append((j, bbd))

        if len(candidates) == 0:
            continue

        if tw > th:
            tagCord = ty
            tagCordIdx = 1
        else:
            tagCord = tx
            tagCordIdx = 0

        index = 0

        if len(candidates) > 0:
            # Handling of multiple BBoxes that intersects tagLine
            bCords = [tup[1].box[tagCordIdx] for tup in candidates]
            index = bCords.index(min(bCords))

            if bCords[index] < tagCord:
                # BBox lies over tagBox, so find the closest BBox that still lies underneath tagBox
                bCords = np.array(bCords)
                sortIdx = np.argsort(bCords)
                for k in sortIdx:
                    if bCords[k] > tagCord:
                        index = k
                        break

        bbd = candidates[index][1]

        if bbd.tagBox is not None:
            # bbox has tag already assigned, so pick the tag with the shortest
            # distance (from center of bbox to center of tag)

            bx, by, bw, bh = bbd.box
            bcx, bcy = int(bx + bw * 0.5), int(by + bh * 0.5)

            tox, toy, tow, toh = bbd.tagBox
            tocX, tocY = int(tox + tow * 0.5), int(toy + toh * 0.5)

            tcx, tcy = int(tx + tw * 0.5), int(ty + th * 0.5)

            oldDist = np.sqrt((tocX - bcx)**2 + (tocY - bcy)**2)
            newDist = np.sqrt((tcx - bcx)**2 + (tcy - bcy)**2)

            if oldDist < newDist:
                continue

        bbd.tagBox = td.tagBox
        bbd.valveID = td.tagID
        assigned.append(i)

    unassignedIdx = tuple(set(range(len(tagDatas))) - set(assigned))
    return unassignedIdx


def identifyTags(tagModel, frame: np.ndarray, tagBoxes: Tuple[Tuple[int, int, int, int]], tagIDLength: int = 5) -> List[TagData]:
    tagDatas = []

    for i, tag in enumerate(tagBoxes):
        tagID, charsFound = detectTagID(tagModel, frame, tag, tagIDLength)
        tagLine = calcTagLine(frame, tag)

        tagDatas.append(
            TagData(tagID, tag, tagLine)
        )
    return tagDatas





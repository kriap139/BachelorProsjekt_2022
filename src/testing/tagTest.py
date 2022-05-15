from src.testing.PlotWindow import PlotWindow
from src.Backend.StateMethods.Methods import filterContours
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


class ValveStateTest:
    def __init__(self):
        self.window = PlotWindow()

    def display(self, title: str, img, cmap=None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap=cmap)

        self.window.addPlot(title, fig)
        pass

    def show(self):
        self.window.show()


def findMaskContours(hsl: np.ndarray, colorLower: tuple, colorUpper: tuple):
    # Find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(hsl, colorLower, colorUpper)

    # Deleting noises which are in area of mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask


def findTagIndex(conts):
    b_area = cv2.contourArea(conts[0])
    b_idx = 0

    nb_area = 0
    nb_idx = 0

    for i in range(1, len(conts)):
        area = cv2.contourArea(conts[i])

        if area > b_area:
            nb_area = b_area
            nb_idx = b_idx

            b_area = area
            b_idx = i

    return nb_idx



def sortBiggestFirst(conts: np.ndarray):
    sc = sorted(conts, key=cv2.contourArea, reverse=True)
    return np.array(sc, dtype=object)


# https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))
    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot


# https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# For Texst(svart): Lower=(0, 44, 0), Upper=(180, 184, 43)
# tag(hvit): Lower=(0, 227, 255), Upper=(180, 0, 255)
# tag(new): Lower=(0, 0, 0), Upper=(180, 218, 255)


# vs = ValveStateTest()
p = os.path.join("resources", "testing", "ventil-tilstand", "9.png")

img = cv2.imread(p)
hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

mask = findMaskContours(hsl, colorLower=(0, 0, 0), colorUpper=(180, 218, 255))
conts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)



# Grab only the innermost child components
# https://stackoverflow.com/questions/49958640/sort-associated-hierarchy-with-already-sorted-contours-in-opencv-in-python

# hier = hier[0]  # get the actual inner list of hierarchy descriptions
# inner_contours = [i for i, c in enumerate(zip(conts, hier)) if c[1][3] > 0]

#cv2.imshow("mask", mask)
#cv2.waitKey(0)

sc = sortBiggestFirst(conts)
sc = sc[2:]
sc = filterContours(sc, filterThreshold=0.30)
# mar_tag = cv2.minAreaRect(sc[0])
# box = np.int64(cv2.boxPoints(mar_tag))

# cropped, rot = crop_rect(img, mar_tag)

# mask = findMaskContours(cropped, colorLower=(0, 0, 0), colorUpper=(180, 218, 255))
# conts, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in sc:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # mar = cv2.minAreaRect(c)
    # box = np.int64(cv2.boxPoints(mar))
    # cv2.drawContours(img, [box], -1, (0, 255, 0), thickness=3)

    cv2.imshow("Test", img)
    cv2.waitKey(0)



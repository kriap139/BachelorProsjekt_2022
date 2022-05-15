import cv2
import numpy as np
from typing import Iterable, Tuple, List
from src.Backend.DataClasses import BBoxData
import os
from PIL import Image
from tensorflow import keras


def assignTagsToBBoxes(tagBoxes: Tuple[Tuple[int, int, int, int], ...], bboxes: List[BBoxData]) -> None:
    if not any(tagBoxes):
        return

    bCenters = tuple((b.box[0] + b.box[2] * 0.5, b.box[1] + b.box[3] * 0.5) for b in bboxes)
    tCenters = tuple((box[0] + box[2] * 0.5, box[1] + box[3] * 0.5) for box in tagBoxes)

    for bc, bbd in zip(bCenters, bboxes):

        currTc = tCenters[0]

        sq = (currTc[0] - bc[0])**2 + (currTc[1] - bc[1])**2
        currDist = np.sqrt(sq)

        for i in range(1, len(tagBoxes)):
            tc = tCenters[i]

            sq = (tc[0] - bc[0])**2 + (tc[1] - bc[1])**2
            dist = np.sqrt(sq)

            if (dist < currDist) and (tc[1] < bc[1]):
                bbd.tagBox = tagBoxes[i]
                currTc = tCenters[i]
                currDist = dist


def Filter_Contours_Based_On_ArcLength(contours, requiredLength=40, filterThreshold=0.16):
    """fiilters contours based on their arcLength. Contours Smaller then the requiredLength(in pixels)
    parameter gets removed. Additionally any Contour that are to small commpared to the biggest countour in the list,
    will also be removed. The cantors minimum length (compared to the longest in the list),
    is given by the filterThreshold param in precent[0-1]"""
    filtered_Contours_Based_On_ArcLength = []
    if not contours.__len__():
        return tuple()
    lengths = tuple(cv2.arcLength(contour, True) for contour in contours)
    longest = np.max(lengths)
    for i in range(contours.__len__()):
        if lengths[i] > requiredLength and (lengths[i] / longest > filterThreshold):
            filtered_Contours_Based_On_ArcLength.append(contours[i])
    return filtered_Contours_Based_On_ArcLength


def Filter_Contours_Based_On_Dimensions_Of_Bounding_Boxes(image, conturs):
    filtered_Contours_Based_On_Bounding_Boxes = []
    imageArea = image.shape[0] * image.shape[1]
    height, width, _ = image.shape
    for i in range(0, len(conturs)):
        contour = conturs[i]
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) <= (0.3 * imageArea) and cv2.contourArea(contour) >= (0.00009 * imageArea) and (
                w <= 0.2 * width) and (w > 0.017 * width) and (h >= 0.3 * height) and (h <= 0.8 * height):
            filtered_Contours_Based_On_Bounding_Boxes.append(contour)
    return filtered_Contours_Based_On_Bounding_Boxes


def Get_x_Coordinates_For_The_Contour_Center(conturs):
    x_Coordinates_For_Contours_Centers = []

    # Getting the x center coordinate for the filtered contours
    for i in range(0, len(conturs)):
        contour = conturs[i]
        # moment
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            x_Coordinates_For_Contours_Centers.append(cx)

    # print("x_centers",x_Coordinates_For_Contours_Centers)
    return x_Coordinates_For_Contours_Centers


def Filter_Contours_That_Have_Same_x_Coordinate_Of_The_Center(conturs):
    filtered_Contours_Based_On_Same_x_Coordinate = []
    unique_x_Centers_Coordinate = []
    x_Centers_Coordinate = Get_x_Coordinates_For_The_Contour_Center(conturs)
    # Filtering  the contours that have the same x center coordinate
    for unique_cx, unique_contour in zip(x_Centers_Coordinate, conturs):
        # Check if exists in unique_x_Centers_Coordinate list or not
        if unique_cx not in unique_x_Centers_Coordinate:
            unique_x_Centers_Coordinate.append(unique_cx)
            filtered_Contours_Based_On_Same_x_Coordinate.append(unique_contour)
    return filtered_Contours_Based_On_Same_x_Coordinate, unique_x_Centers_Coordinate


def Filter_Inner_Contours_Insid_Parent_Contours(contours, unique_x_Centers):
    filtered_Contours_Final = []
    x_Centers_Final = []
    # Filtering child contours that insid parent contours
    for i in range(1, len(unique_x_Centers)):

        if i == 1:
            x_Centers_Final.append(unique_x_Centers[0])
            filtered_Contours_Final.append(contours[0])

        if (i != (len(unique_x_Centers) - 1)) and (unique_x_Centers[i - 1] + 9 < unique_x_Centers[i]):
            x_Centers_Final.append(unique_x_Centers[i])
            filtered_Contours_Final.append(contours[i])

        if i == (len(unique_x_Centers) - 1):
            x_Centers_Final.append(unique_x_Centers[len(unique_x_Centers) - 1])
            filtered_Contours_Final.append(contours[len(contours) - 1])

    for i in range(0, len(x_Centers_Final)):
        # filter last 2 items
        if (x_Centers_Final[len(x_Centers_Final) - 2] + 7) > (x_Centers_Final[len(x_Centers_Final) - 1]):
            x_Centers_Final = x_Centers_Final[:-1]
            filtered_Contours_Final = filtered_Contours_Final[:-1]

    if len(filtered_Contours_Final) == 6:
        filtered_Contours_Final.pop()

    return filtered_Contours_Final, x_Centers_Final


def Filter_Contours(image, contours, filterThreshold=0.05):
    filtered_Contours_Based_On_ArcLength = Filter_Contours_Based_On_ArcLength(contours, filterThreshold=0.05)

    filtered_Contours_Based_On_Bounding_Boxes = Filter_Contours_Based_On_Dimensions_Of_Bounding_Boxes(image,
                                                                                                      filtered_Contours_Based_On_ArcLength)

    filtered_Contours_Based_On_Same_x_Coordinate, unique_x_Centers_Coordinate = Filter_Contours_That_Have_Same_x_Coordinate_Of_The_Center(
        filtered_Contours_Based_On_Bounding_Boxes)

    filtered_Contours_Final, x_Centers_Final = Filter_Inner_Contours_Insid_Parent_Contours(
        filtered_Contours_Based_On_Same_x_Coordinate, unique_x_Centers_Coordinate)

    return filtered_Contours_Final


def Sort_Contours(contours: np.ndarray):
    # Sort contours left-to-right based on boundingRect for each contour
    sorted_contours_based_boundingRect = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    return np.array(sorted_contours_based_boundingRect, dtype=object)


def Fix_Dimension(image):
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = image
    return new_img


def CNN_Model(model, digits):
    dic = {}
    result = []
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    #model = keras.models.load_model(CNN_Model_Path)

    for i, char in enumerate(characters):
        dic[i] = char
    for j in range(0, len(digits)):
        digit = digits[j]
        digit = Fix_Dimension(digit)
        digit = digit.reshape(1, 28, 28, 3)
        predictions = model.predict(digit)[0]
        # Cutting_Digits(digits[j], predictions, j ,counterValue)
        # print (predictions)
        character = dic[np.argmax(predictions)]
        result.append(character)

    Tag_Characters = ''.join(result)

    return Tag_Characters


def createCNNModel(modelPath: str):
    return keras.models.load_model(modelPath)


def detectTagID(cnnModel, frame: np.ndarray, tagBox: Tuple[int, int, int, int]):

    x, y, w, h = tagBox
    tagImage = frame[y:y + h, x:x + w]

    digits = []

    # Convert the input image to grayscale
    gray = cv2.cvtColor(tagImage, cv2.COLOR_RGB2GRAY)

    # Appling gaussian blur to smoothen image for ignoring much of the detail and instead focus on the actual structure.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Appling the Canny edge detector
    canny_output = cv2.Canny(blur, 170, 230)

    # Find contours of regions of interest within the license plate image
    try:
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours left-to-right based on boundingRect for each contour
    sorted_contours = Sort_Contours(contours)

    # Filter unwanted contours
    filtered_contours = Filter_Contours(tagImage, sorted_contours, filterThreshold=0.05)

    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)

        # grab digit region of the tag Image
        try:
            digit = tagImage[y - 3:y + h + 3, x - 3:x + w + 3]
        except:
            digit = tagImage[y:y + h, x:x + w]

        # Convert the digit image to Grayscale
        gray = cv2.cvtColor(digit, cv2.COLOR_RGB2GRAY)

        # Perform GaussianBlur on the digit image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # resize the digit image to the dimension (28,28)
        digit = cv2.resize(blur, (28, 28), interpolation=cv2.INTER_AREA)
        # print(digit.shape)

        # Threshold the digit image using Otsus method
        ret, digit = cv2.threshold(digit, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        digits.append(digit)

    digits = np.array(digits, dtype="float")
    digits_CNN = CNN_Model(cnnModel, digits)

    return digits_CNN if len(digits_CNN) else None

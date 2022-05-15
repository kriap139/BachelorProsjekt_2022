#import numpy as np
import cv2 as cv
import glob
from src.testing.PlotWindow import PlotWindow
import matplotlib.pyplot as plt
import os


class ValveStateTest:

    def __init__(self):
        self.window = PlotWindow()

    def display(self, title: str, img, cmap=None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap=cmap)

        self.window.addPlot(title, fig)

    def show(self):
        self.window.show()

vs = ValveStateTest()
# img2 = cv.imread('b.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv.GaussianBlur(img2,(3,3),cv.BORDER_DEFAULT)
# unsharp_mask(img1)

p = os.path.join("resources", "testing", "sift-test", "test-bilde.jpg")
p2 = os.path.join("resources", "testing", "sift-test", "3", "ALL")

img1 = cv.imread(p, cv.IMREAD_GRAYSCALE)  # trainImage
img1 = cv.GaussianBlur(img1, (5, 5), cv.BORDER_DEFAULT)

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)

# Load all the images
all_images_to_compare = []
titles = []
for f in glob.iglob(f"{p2}{os.path.sep}*"):
    image = cv.imread(f, cv.IMREAD_GRAYSCALE)
    image = cv.GaussianBlur(image, (5, 5), cv.BORDER_DEFAULT)
    titles.append(f)
    all_images_to_compare.append(image)


for image_to_compare, title in zip(all_images_to_compare, titles):
    kp2, des2 = sift.detectAndCompute(image_to_compare, None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
        number_keypoints = 0
        if len(kp1) >= len(kp2):
            number_keypoints = len(kp1)
        else:
            number_keypoints = len(kp2)

    percentage_similarity = len(good) / number_keypoints * 100
    print(len(good))
    print(number_keypoints)
    print("Similarity: " + str(int(percentage_similarity)) + "\n")
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, image_to_compare, kp2, good, None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #plt.imshow(img3), plt.show()
    vs.display(title,img3)

vs.show()
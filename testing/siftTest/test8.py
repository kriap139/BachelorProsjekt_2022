import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob

#img2 = cv.imread('b.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
#img2 = cv.GaussianBlur(img2,(3,3),cv.BORDER_DEFAULT)
#unsharp_mask(img1)
img1 = cv.imread('apen.jpg',cv.IMREAD_GRAYSCALE) # trainImage
img1 = cv.GaussianBlur(img1,(9,9),cv.BORDER_DEFAULT)

# Initiate SIFT detector
sift = cv.SIFT_create()
#sift = cv.ORB_create()
#sift = cv.FREAK_create()
#sift = cv.SURF_create(400)

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
bf = cv.BFMatcher()
#index_params = dict(algorithm=0, trees=1)
#search_params = dict()
#bf = cv.FlannBasedMatcher(index_params, search_params)
# Load all the images
all_images_to_compare = []
titles = []
for f in glob.iglob("images\*"):
    image = cv.imread(f,cv.IMREAD_GRAYSCALE)
    image = cv.GaussianBlur(image,(3,3),cv.BORDER_DEFAULT)
    titles.append(f)
    all_images_to_compare.append(image)
    
for image_to_compare, title in zip(all_images_to_compare, titles):
    kp2, des2 = sift.detectAndCompute(image_to_compare,None)
    # BFMatcher with default params
    
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
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
    img3 = cv.drawMatchesKnn(img1,kp1,image_to_compare,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
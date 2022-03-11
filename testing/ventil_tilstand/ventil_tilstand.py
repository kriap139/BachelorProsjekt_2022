# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:35:27 2022

@author: almut
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from rembg.bg import remove
import io
from PIL import Image
from PIL import ImageFile
import os


#Defining function for plotting images
def display(img,cmap=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cmap)


# removing background of the input image 
def remover (input_image):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    f = np.fromfile(input_image)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    img=np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #output_image = 'output1.png'
    #img.save(output_image)
    return img
    

### Implemnting of the Watersheld Algorithma
def watersheld (img_Pipe):

    # To apply median Blur to imag for removing the unnecessary details from the image 
    img_Pipe = cv2.medianBlur(img_Pipe,35)#kernel size is 25 or 35
    
    # Convert the image to Grayscale
    gray_p = cv2.cvtColor(img_Pipe,cv2.COLOR_BGR2GRAY)
    
    # Apply Threshold (Inverse Binary with OTSU method as well)  in order to make it black and white
    ret, thresh = cv2.threshold(gray_p,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    display(thresh,cmap='gray')
    
    # noise removal
    # To define a kernel size 3X3 
    kernel = np.ones((3,3),np.uint8)
    # To remove unnecessary noise by applying morphologyEx
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    display(opening,cmap='gray')
    
    # To grab the sure Background area 
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    display(sure_bg,cmap='gray')
    
    # To finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    display(dist_transform,cmap='gray')
    
    #0.7*dist_transform.max()
    ret, sure_fg = cv2.threshold(dist_transform,0.998*dist_transform.max(),255,0)
    display(sure_fg,cmap='gray')
    
    # Finding unknown region in the image
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    display(unknown,cmap='gray')
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    
    # mark the region of unknown with zero
    markers[unknown==255] = 0
    display(markers,cmap='gray')
    
    # Applying Watershed Algorithm to find Markers
    markers = cv2.watershed(img_Pipe,markers)
    display(markers)
    return markers

# Mapping the angle from the rotated reqtangle

def angleCalc (w,h,angle):
    if angle < -45:
        angle = abs(90 + angle)
    if w < h and angle > 0:
        angle = abs((90 - angle) * -1)
    if w > h and angle < 0:
        angle = abs(90 + angle)
    return angle

# path to the original image with background
input_image = f'{os.getcwd()}/resources/ventil-tilstand/2.jpg'

#Read the orginal image
img_orginal=cv2.imread(input_image)
display(img_orginal)

#Remove the background from the original image    
rembg_img = remover(input_image)
display(rembg_img)


# getting the markers by using whatersheld from the rembg_img
markers= watersheld(rembg_img)

# Finding Contours on Markers
# cv2.RETR_EXTERNAL:Only extracts external contours
# cv2.RETR_CCOMP: Extracts both internal and external contours organized in a two-level hierarchy
# cv2.RETR_TREE: Extracts both internal and external contours organized in a  tree graph
# cv2.RETR_LIST: Extracts all contours without any internal/external relationship
contours_p, hierarchy_p = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# return number of contours
#print(len(contours_p))

# sort contours left-to-right
sorted_contours_p = sorted(contours_p, key=lambda ctr: cv2.boundingRect(ctr)[0])
#cv2.drawContours(sep_coins, sorted_contours, i, (255, 0, 0), 20)

# filtered Contours
filter_arr = sorted_contours_p[1]
for i in range(len(filter_arr)):
    # draw the rquaired contour
    cv2.drawContours(rembg_img, filter_arr, i, (0, 255, 0), 20)
display(rembg_img)

# draw boundingRect around the rquaired contour
x_p,y_p,w_p,h_p = cv2.boundingRect(filter_arr)


# Draw a rotated min area rectangle around the rquaired contour
minAreaPipe = cv2.minAreaRect(filter_arr)
box_Pipe = cv2.boxPoints(minAreaPipe)
box_Pipe = np.int0(box_Pipe)

output_Pipe = cv2.drawContours(rembg_img, [box_Pipe], -1, (0, 0, 255), 5)
display(rembg_img)



# To find the angle for the pipe according to the x-axis 
(x_p, y_p), (w_p, h_p), ang_P = minAreaPipe
#print(ang_P)

# calculte the angle for the pipe
angle_Pipe= angleCalc(w_p,h_p, ang_P)

# print("Vinklene til pipe i henhold til X-akse er: " + str(ang_Pipe) +" grader")
#info_P = f"x_p: {np.round(x_p)}, y_p: {np.round(y_p)}, width_p: {np.round(w_p)}, height_h: {np.round(h_p)}, Vinklen til pipen i henhold til X-akse er: {np.round(ang_Pipe)} grader"
info_P = f"Vinklen til pipen iht.x: {np.round(angle_Pipe)}"
print(info_P)

# print informtion for the pipe 
cv2.putText(img=rembg_img, text= info_P, org=(0, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX ,fontScale=3, color=(0, 0, 255), thickness=3)
display(rembg_img)




### To find the angle for the handventil according to the x-axis 


# Defining Color Ranges 

#lower bound and upper bound for yallow color_HSV
color_Lower = (20, 100, 100)
color_Upper = (30, 255, 255)

#Range for Brown color_HSV
#color_Lower = (2, 100, 65)
#color_Upper = (12, 170, 100)

#Range for Red color_HSV
#color_Lower = (161, 155, 84)
#color_Upper = (179, 255, 255)

#Range for Blue color_HSV
#color_Lower = (90, 50, 70)
#color_Upper = (128, 255, 255)
"""
color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
              'white': [[180, 18, 255], [0, 0, 231]],
              'red1': [[180, 255, 255], [159, 50, 70]],
              'red2': [[9, 255, 255], [0, 50, 70]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'blue': [[128, 255, 255], [90, 50, 70]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[24, 255, 255], [10, 50, 70]],
              'gray': [[180, 18, 230], [0, 0, 40]]}
"""
#blur the orginal image to remove the noise 
blurred = cv2.GaussianBlur(img_orginal, (11,11), 0)
display(blurred)

# Convert the image to HSV colorspace 
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
display(hsv)  

# Find the colors within the specified boundaries and apply the mask
mask = cv2.inRange(hsv, color_Lower, color_Upper)

# Deleting noises which are in area of mask
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
display(mask)

# Find contours from the mask
contours_h,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#center = None

if len(contours_h) > 0:
    # get max contour
    c = max(contours_h, key=cv2.contourArea)
    # Draw a rotated min area rectangle around the max contour
    rect = cv2.minAreaRect(c)
    ((x_h,y_h), (w_h, h_h), angle_h) = rect
    
    # Finding the angle for the handventil
    angle_handventil= angleCalc(w_h, h_h, angle_h)
    
    #print("Vinklene til handvantilen i henhold til X-akse er: " + str(ang_h) +" grader")
    #info_h = f"x_h: {np.round(x_h)}, y_h: {np.round(y_h)}, width_h: {np.round(w_h)}, height_h: {np.round(h_h)}, Vinklen til handVentil i henhold til X-akse er: {np.round(ang_h)} grader"
    info_h = f"Vinkel til vintel iht.x:{np.round(angle_handventil)}"
    print(info_h)

    # box
    box_h = cv2.boxPoints(rect)
    box_h = np.int64(box_h)

    # moment
    M = cv2.moments(c)
    center_h = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    
    # draw boundingRect around the detected contour for the pipe on the orginal image
    cv2.drawContours(img_orginal, [box_Pipe], 0, (0, 0, 255), 3)

    # draw boundingRect around the detected contour for the handventil  on the orginal image
    cv2.drawContours(img_orginal, [box_h], 0, (255, 0, 0), 3)

    # point in center
    cv2.circle(img_orginal, center_h, 5, (255, 0, 255), 1)

    
    # print informtion for the pipe on the orginal image
    cv2.putText(img=img_orginal, text= info_P, org=(0, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX ,fontScale=3, color=(0, 0, 255), thickness=3)
    
    # print informtion for the handventil on the orginal image
    cv2.putText(img=img_orginal, text= info_h, org=(0, 200), fontFace=cv2.FONT_HERSHEY_TRIPLEX ,fontScale=3, color=(255, 0, 0), thickness=3)

    # display the orginal image 
    display(img_orginal)
    
# make a decision
info_V=""

if ((angle_handventil>=74) and (angle_Pipe>=74)) or (angle_handventil == angle_Pipe):

    print("Ventilen er apent")
    info_V="Ventilen er apent"

else:

    print("Ventilen er stengt")
    info_V="Ventilen er stengt"
    
# print informtion about the valve state on the orginal image
cv2.putText(img=img_orginal, text= info_V, org=(0, 300), fontFace=cv2.FONT_HERSHEY_TRIPLEX ,fontScale=3, color=(0, 255, 0), thickness=3)
display(img_orginal)
plt.show()

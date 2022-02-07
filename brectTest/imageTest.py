# TechVidvan Object detection of similar color
import cv2
import numpy as np
# Reading the image
img = cv2.imread('image3.jpg')
cv2.imshow("Output", img)
#img = cv2.imread('image.jpg')
scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# Showing the output


# convert to hsv colorspace
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lower bound and upper bound for Green color
lower_bound = np.array([0, 140, 35])   
upper_bound = np.array([30, 255, 60])
# find the colors within the boundaries
mask = cv2.inRange(hsv, lower_bound, upper_bound)

#define kernel size  
kernel = np.ones((7,7),np.uint8)
# Remove unnecessary noise from mask
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
result = cv2.bitwise_and(img, img, mask = mask)

# Find contours from the mask
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key = cv2.contourArea)

x,y,w,h = cv2.boundingRect(c)

#minAreaTest
tst=cv2.minAreaRect(c)
box = cv2.boxPoints(tst)
box = np.int0(box)

output = cv2.drawContours(result, c, -1, (0, 0, 255), 3)
output = cv2.drawContours(img, [box], -1, (0, 0, 255), 3)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#cv2.imshow("Output", output)
# Showing the output
#cv2.imshow("Output", result)
cv2.imshow("Output", img)

if w < h:
    print("Ventil åpen")
else:
    print("Ventil stengt")


#cv2.imshow("Image", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
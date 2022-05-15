# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 22:23:03 2022

@author: almut
"""
# Import the necessary libraries
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow import keras

# Paths
#New tags
PathToNewtagsFolder=r"New_Tags_Dataset"

# Path to CNN Model h5 file
PathToCNNModel = r"CNN_Characters_Classification.h5"

# Path to save testing results
PathToSaveTestingResults = r"Testing_Results_Images"

#Defining function for plotting images
def display(img,cmap=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap=cmap)

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
        if lengths[i] > requiredLength and (lengths[i]/longest > filterThreshold):
            filtered_Contours_Based_On_ArcLength.append(contours[i])
    return filtered_Contours_Based_On_ArcLength


def Filter_Contours_Based_On_Dimensions_Of_Bounding_Boxes(image,conturs):
    filtered_Contours_Based_On_Bounding_Boxes = []
    imageArea = image.shape[0]*image.shape[1]
    height, width ,_ = image.shape
    for i in range(0,len(conturs)):
        contour = conturs[i]
        x,y,w,h = cv2.boundingRect(contour)
        if cv2.contourArea(contour)<= (0.3*imageArea) and cv2.contourArea(contour)>= (0.00009*imageArea) and (w <=0.2*width) and (w >0.017*width) and  (h >=0.3*height) and (h <=0.8*height):
            filtered_Contours_Based_On_Bounding_Boxes.append(contour)
    return filtered_Contours_Based_On_Bounding_Boxes

def Get_x_Coordinates_For_The_Contour_Center(conturs):
    x_Coordinates_For_Contours_Centers = []
    
    # Getting the x center coordinate for the filtered contours
    for i in range(0,len(conturs)):
        contour = conturs[i]
        # moment
        M = cv2.moments(contour)
        if M['m00'] !=0:
           cx = int(M['m10']/M['m00'])
           x_Coordinates_For_Contours_Centers.append(cx)
        
    #print("x_centers",x_Coordinates_For_Contours_Centers)
    return x_Coordinates_For_Contours_Centers

def Filter_Contours_That_Have_Same_x_Coordinate_Of_The_Center (conturs):
    filtered_Contours_Based_On_Same_x_Coordinate = []
    unique_x_Centers_Coordinate = []
    x_Centers_Coordinate = Get_x_Coordinates_For_The_Contour_Center(conturs)
    # Filtering  the contours that have the same x center coordinate 
    for unique_cx, unique_contour in zip(x_Centers_Coordinate ,conturs):
    # Check if exists in unique_x_Centers_Coordinate list or not    
      if unique_cx not in unique_x_Centers_Coordinate:
           unique_x_Centers_Coordinate.append(unique_cx)
           filtered_Contours_Based_On_Same_x_Coordinate.append(unique_contour)
    return filtered_Contours_Based_On_Same_x_Coordinate, unique_x_Centers_Coordinate


def Filter_Inner_Contours_Insid_Parent_Contours(contours,unique_x_Centers):

    filtered_Contours_Final = []
    x_Centers_Final = []
    # Filtering child contours that insid parent contours
    for i in range (1 ,len(unique_x_Centers)):
        
        if  i == 1:
            x_Centers_Final.append(unique_x_Centers[0])
            filtered_Contours_Final.append(contours[0])
         
        if (i != (len(unique_x_Centers)-1)) and (unique_x_Centers[i-1]+9 < unique_x_Centers[i]):
             x_Centers_Final.append(unique_x_Centers[i])
             filtered_Contours_Final.append(contours[i])
             
        if  i == (len(unique_x_Centers)-1):
            x_Centers_Final.append(unique_x_Centers[len(unique_x_Centers)-1])
            filtered_Contours_Final.append(contours[len(contours)-1])

    for i in range (0 ,len(x_Centers_Final)):
       # filter last 2 items
        if (x_Centers_Final[len(x_Centers_Final)-2]+7) > (x_Centers_Final[len(x_Centers_Final)-1]):
            x_Centers_Final = x_Centers_Final[:-1]
            filtered_Contours_Final = filtered_Contours_Final[:-1]
    
    if len(filtered_Contours_Final)==6:
        filtered_Contours_Final.pop()

    
    return filtered_Contours_Final , x_Centers_Final

def Filter_Contours(image, contours, filterThreshold=0.05):
    
    filtered_Contours_Based_On_ArcLength = Filter_Contours_Based_On_ArcLength(contours, filterThreshold=0.05)
    
    filtered_Contours_Based_On_Bounding_Boxes =  Filter_Contours_Based_On_Dimensions_Of_Bounding_Boxes(image,filtered_Contours_Based_On_ArcLength)
        
    filtered_Contours_Based_On_Same_x_Coordinate, unique_x_Centers_Coordinate = Filter_Contours_That_Have_Same_x_Coordinate_Of_The_Center (filtered_Contours_Based_On_Bounding_Boxes)
    
    filtered_Contours_Final , x_Centers_Final = Filter_Inner_Contours_Insid_Parent_Contours(filtered_Contours_Based_On_Same_x_Coordinate,unique_x_Centers_Coordinate)
    
    return filtered_Contours_Final


def Sort_Contours(contours: np.ndarray):
    # Sort contours left-to-right based on boundingRect for each contour
    sorted_contours_based_boundingRect = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    return np.array(sorted_contours_based_boundingRect, dtype=object)


def Saving_Digits (digit, i, counterValue, path_Folder):
    img_File = "{}{}.jpg".format(counterValue,i)
    joined_path = os.path.join(path_Folder, img_File)
    cv2.imwrite(joined_path, digit)

def Saving_Images (image,counterValue, path_Folder):
    img_File = "{}.jpg".format(counterValue)
    joined_path = os.path.join(path_Folder, img_File)
    cv2.imwrite(joined_path, image)
    
    
def Cutting_Digits(digit, predictions, i,counterValue ):
     # If the detected character is 0
     if predictions[0] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_0")
     # If the detected character is 1
     if predictions[1] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_1")
     # If the detected character is 2
     if predictions[2] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_2")
     # If the detected character is 3
     if predictions[3] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_3")
     # If the detected character is 4
     if predictions[4] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_4")
     # If the detected character is 5
     if predictions[5] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_5")
     # If the detected character is 6
     if predictions[6] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_6")
     # If the detected character is 7
     if predictions[7] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_7")
     # If the detected character is 8
     if predictions[8] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_8")
     # If the detected character is 9
     if predictions[9] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_9")
     # If the detected character is A
     if predictions[10] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_A")
     # If the detected character is B
     if predictions[11] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_B")
     # If the detected character is C
     if predictions[12] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_C")
     # If the detected character is D
     if predictions[13] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_D")
     # If the detected character is E
     if predictions[14] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_E")
     # If the detected character is F
     if predictions[15] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_F")
     # If the detected character is G
     if predictions[16] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_G")
     # If the detected character is H
     if predictions[17] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_H")
     # If the detected character is I
     if predictions[18] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_I")
     # If the detected character is J
     if predictions[19] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_J")
     # If the detected character is K
     if predictions[20] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_K")
     # If the detected character is L
     if predictions[21] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_L")
     # If the detected character is M
     if predictions[22] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_M")
     # If the detected character is N
     if predictions[23] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_N")
     # If the detected character is O
     if predictions[24] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_O")
     # If the detected character is P
     if predictions[25] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_P")
     # If the detected character is Q
     if predictions[26] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_Q")
     # If the detected character is R
     if predictions[27] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_R")
     # If the detected character is S
     if predictions[28] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_S")
     # If the detected character is T
     if predictions[29] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_T")
     # If the detected character is U
     if predictions[30] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_U")
     # If the detected character is V
     if predictions[31] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_V")
     # If the detected character is W
     if predictions[32] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_W")
     # If the detected character is X
     if predictions[33] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_X")
     # If the detected character is Y
     if predictions[34] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_Y")
     # If the detected character is Z
     if predictions[35] == 1:
        Saving_Digits (digit, i, counterValue, path_Folder=r"digits_dataset\train\class_Z")



def findMaskContours(hsl: np.ndarray, colorLower: tuple, colorUpper: tuple):
    # Find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(hsl, colorLower, colorUpper)
    # Deleting noises which are in area of mask
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    return mask

def Fix_Dimension(image):
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = image
    return new_img  

def CNN_Model(digits,CNN_Model_Path,counterValue):
    dic = {}
    result = []
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    model = keras.models.load_model(CNN_Model_Path)

    for i,char in enumerate(characters):
        dic[i] = char
    for j in range (0,len(digits)):
    #for digit in digits: 
        digit = digits[j]
        digit = Fix_Dimension(digit)
        digit = digit.reshape(1,28,28,3)
        predictions = model.predict(digit)[0]
        #Cutting_Digits(digits[j], predictions, j ,counterValue)
        #print (predictions)
        character = dic[np.argmax(predictions)]
        result.append(character) 
        
    Tag_Characters = ''.join(result)
    
    if len(Tag_Characters) != 0 and len(Tag_Characters) == 5 :
       print("CNN Tag Reader: ", Tag_Characters)
    else:
        print("CNN Tag Reader: ", "not detectable")
    return Tag_Characters 


def proccessing(tagImage,counterValue):
    
    digits = []
    
    # Display the License Plate image 
    #display(tagImage)

    #Make a copy of the orginal Image in order to do some tests
    copy1 = tagImage.copy()
    copy2 = tagImage.copy()
    copy3 = tagImage.copy()
    copy4 = tagImage.copy()
    copy5 = tagImage.copy()
    
    

    
    # First Method Canny method

    # Convert the input image to grayscale
    gray = cv2.cvtColor(copy1, cv2.COLOR_RGB2GRAY)
    #display(gray)
    
    # Appling gaussian blur to smoothen image for ignoring much of the detail and instead focus on the actual structure.
    blur =  cv2.GaussianBlur(gray, (5, 5), 0)
    #display(blur)
    


    # Appling the Canny edge detector
    canny_output = cv2.Canny(blur, 170, 230)
    #display(canny_output)
    

    # Second Method HLS method
    
    # Convert the image to HLS colorspace 
    #hls = cv2.cvtColor(copy1, cv2.COLOR_BGR2HLS)
    #display(hls) 
    
    # Find the colors within the specified boundaries and apply the mask
    #mask = findMaskContours(hls, colorLower=(0, 155, 0), colorUpper=(255, 255, 255))
    #display(mask)

    # Find contours of regions of interest within the license plate image
    try:
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    #print("The Number of contours before sortering and filtering = ", len(contours))

    # To draw all founded contours
    cv2.drawContours(copy2, contours, -1, (0,0,255), 1)
    #display(copy2)
    

    # To draw bounding Rectangels around contours before sortering and filtering
    for i in range(0,len(contours)):
        contourBeforFiltering = contours[i]
        x,y,w,h = cv2.boundingRect(contourBeforFiltering)
        color = list(np.random.random(size=3) * 256)
        cv2.rectangle(copy3, (x, y), (x + w, y + h), color, 1)
    display(copy3)
    


    # Sort contours left-to-right based on boundingRect for each contour
    sorted_contours = Sort_Contours(contours)
    
    # Filter unwanted contours
    filtered_contours = Filter_Contours(copy4, sorted_contours, filterThreshold=0.05)
    print("Number of Filtered Contours = ",len(filtered_contours))
    
    # To draw bounding Rectangels around contours after sortering and filtering
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        color = list(np.random.random(size=3) * 256)
        cv2.rectangle(copy5,(x,y),(x+w,y+h),color,1)

        # grab digit region of the tag Image
        try:
            digit = tagImage[y-3:y+h+3 , x-3:x+w+3]
        except:
            digit = tagImage[y:y+h , x:x+w]
       
        # Convert the digit image to Grayscale
        gray = cv2.cvtColor(digit, cv2.COLOR_RGB2GRAY)
        
        # Perform GaussianBlur on the digit image
        blur =  cv2.GaussianBlur(gray, (5, 5), 0)
        
        # resize the digit image to the dimension (28,28)
        digit = cv2.resize(blur, (28,28), interpolation=cv2.INTER_AREA)
        #print(digit.shape)

        # Threshold the digit image using Otsus method 
        ret, digit = cv2.threshold(digit, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        digits.append(digit)

    digits = np.array(digits, dtype="float")
    digits_CNN = CNN_Model(digits,PathToCNNModel,counterValue)
    #cv2.putText(img=copy5, text= digits_CNN, org=(10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX ,fontScale=1, color=(255, 0, 0), thickness=2)
    #display(copy5)
    #resized_up = cv2.resize(copy5, (600,200), interpolation= cv2.INTER_LINEAR)
    #Saving_Images (resized_up,counterValue, PathToSaveTestingResults)

    

#Testing from 0 to 100
#28,56,64,65,70
counterValue = 900
# Loop hrough the 50 test images 
for p in range(8,9):
    counterValue += 1
    print("Image Number: ",p)
    pathToImage = str(p)+'.jpg'
    joined_path = os.path.join(PathToNewtagsFolder, pathToImage)
    tagImage = cv2.imread(joined_path)
    proccessing(tagImage,counterValue)



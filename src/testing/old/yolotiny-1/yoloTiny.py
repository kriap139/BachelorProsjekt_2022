import cv2
print(cv2.__version__)
import numpy as np
import time
import imutils
from Util.PathFuncs import dataDir
#import pafy
# Load Yolo
#net = cv2.dnn.readNet("weights/yolov4-custom_last.weights", "cfg/yolov4-custom.cfg")
#weights = dataDir(add="/models/weights/v4tiny_model3.weights")
#cfg = dataDir(add="/models/cfg/v4tiny.cfg")
#names = dataDir(add="/models/classesnames/classes.names")
weights = "./models/weights/v4tiny_model3.weights"
cfg = "./models/cfg/v4tiny.cfg"
names = "./models/classesnames/classes.names"


net = cv2.dnn.readNet(weights, cfg)
classes = []
with open(names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# Loading camera
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("./testvid/al4.mp4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
#result = cv2.VideoWriter('filename.avi', 
#                         cv2.VideoWriter_fourcc(*'MJPG'),
#                         10, size)
   

#url= "https://www.youtube.com/watch?v=FbkkMF4LMnw"
#vPafy=pafy.new(url)
#play = vPafy.getbest(preftype = "mp4")
#cap = cv2.VideoCapture(play.url)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
    _, frame = cap.read()
 #   frame = imutils.resize(frame, width=320)
    frame_id += 1
    height, width, channels = frame.shape
    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            detectedObj = frame[y:y+h,x:x+w]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, (255,255,255), 3)
            
    print(boxes)        
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
    #Lagre video
    #result.write(frame)
   # cv2.imshow("Image", frame)
    if (detectedObj.size):
        cv2.imshow("Image", frame)
    
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
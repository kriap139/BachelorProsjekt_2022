import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

#Input files
weights = "weights/yolo-tiny-valve_final.weights"
cfg = "cfg/yolo-tiny-valve.cfg"
names = r'ventiler.names'
test_dir = r'dataset_for_testing_classifiction'

def runForSingleImage(model, image):
    # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
     swapRB=True, crop=False)
    #Detecting objects
    model.setInput(blob)
    outs = model.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
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
    return class_ids, confidences, boxes

def show(model, image):    
    #Non-maximum Suppression (ARGS: score threshold and nms threshold)
    class_ids,confidences, boxes  = runForSingleImage(model, image)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = color = [int(c) for c in COLORS[class_ids[i]]]
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label+ " " + str(round(confidence, 2)), (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,
			1/2, color, 2)
    imS = cv2. resize(img, (1600, 900)) # Resize image.
    cv2.imshow("img",imS)
    
    #cv2.imshow("finalImg",img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()
    return indexes
    
#Setup Yolo with weights and cfg
net = cv2.dnn.readNet(weights, cfg)
classes = []
with open(names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
COLORS=np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]

print("Starting object detection")
totalTattoos = 0
tic = time.perf_counter()
imagePaths = glob.iglob(os.path.join(test_dir, "*.jpg"))
#Performing object detection on all jpg in test_dir
for bilde in imagePaths:
    print(bilde)
    img = cv2.imread(bilde)
    height, width, channels = img.shape
    
    indexes=show(net,img)
    totalTattoos += len(indexes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
toc = time.perf_counter()

#Results
print(f"\n Total deteksjoner: {totalTattoos} \n Total Time: {toc - tic:0.2f} seconds")

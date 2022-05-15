import cv2 as cv2
import numpy as np


angle = 0
image = np.zeros((200, 400, 3), dtype=np.uint8)
originalRect = ()
vertices = ()
vertVect = []
calculatedRect = ()

while cv2.waitKey(5000) != 27:
    originalRect = ((100, 100), (100, 50), angle)

    # calc points and move rectangle
    vertices = cv2.boxPoints(originalRect)
    vertices = tuple((int(p[0]), int(p[1])) for p in vertices)

    for i in range(4):
        vertex = vertices[i]
        vertVect.append((vertex[0] + 200, vertex[1]))

    vertVect = np.array(vertVect, dtype=np.int32)
    calculatedRect = cv2.minAreaRect(vertVect)

    for i in range(4):
        cv2.line(image, vertices[i], vertices[(i + 1) % 4], (0, 255, 0))
        cv2.line(image, vertVect[i], vertVect[(i + 1) % 4], (255, 0, 0))

    cv2.imshow("rectangles", image)

    #print(f"Original: w={originalRect[1][0]}, h={originalRect[1][1]}, angle={angle}")
    #print(f"Calculated: w={calculatedRect[1][0]}, h={calculatedRect[1][1]}, angle={calculatedRect[2]}\n")

    ((cx, cy), (w, h), ma) = calculatedRect

    print(f"OG Angle={angle} -> MAR: w={w}, h={h}, angle={ma}")

    image = np.zeros((200, 400, 3), dtype=np.uint8)
    vertVect = []
    angle += 10

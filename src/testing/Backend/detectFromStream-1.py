import numpy as np
import cv2 as cv
import numbers
from typing import Tuple, Callable, Union, List
from src.Backend.DataClasses import ImageData, BBoxData, SharedImage
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from collections.abc import Iterable


def cleanupSharedMem(shmNames: dict, lock, shmName: str):
    with lock:
        isPresent = shmNames.pop(shmName, default=False)
        if isPresent:
            shm = SharedMemory(shmName)
            shm.close()
            shm.unlink()


def cleanupAllSharedMem(shmNames: dict, lock):
    with lock:
        for shmName in shmNames.keys():
            shm = SharedMemory(shmName)
            shm.close()
            shm.unlink()

        shmNames.clear()


def showTest(classIfQueue: mp.Queue, exitEvent: mp.Event):
    while not exitEvent.is_set():
        imgData: ImageData = classIfQueue.get()

        if imgData is None:
            return
        else:
            sharedImg = imgData.sharedImg
            shm = SharedMemory(sharedImg.memName)

            img = np.ndarray(shape=sharedImg.shape, dtype=sharedImg.dType, buffer=shm.buf)
            cv.imshow("img", img)
            cv.waitKey(3)

            print(imgData)
            cleanupSharedMem(shm.name)


def assignTagsToBBoxes(tagBoxes: Iterable[Tuple[int, int, int, int], ...], bboxes: List[BBoxData]) -> None:
    pass


def detectFromStream(cfg: str,
                     weights: str,
                     streamPath: str,
                     outQueue: mp.Queue,
                     exitEvent: mp.Event,
                     shmNames: dict,
                     shmNamesLock,
                     confidValveThresh=0.5,
                     confidTagThresh=0.5):

    net = cv.dnn_DetectionModel(cfg, weights)
    net.setInputSize(416, 416)
    net.setInputScale((1.0 / 255))
    net.setInputSwapRB(True)

    cap = cv.VideoCapture(streamPath)

    while not exitEvent.is_set():
        _, frame = cap.read()

        if frame is None:
            outQueue.put(None)
            break

        classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
        bboxes = []
        tagIdx = []

        for i, (classID, confidence, box) in enumerate(zip(classes, confidences, boxes)):
            # tag class
            if (classID == 7) and (confidence > confidTagThresh):
                tagIdx.append(i)
            elif confidence > confidValveThresh:
                bboxes.append(BBoxData(classID, box))

        if len(bboxes) > 0:
            tagBoxes = [boxes[i] for i in tagIdx]
            if len(tagBoxes) > 0:
                assignTagsToBBoxes(tagBoxes, bboxes)

                # Create a shared memory buffer for image for accessing on treads
                shm = SharedMemory(create=True, size=frame.nbytes)
                copy = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
                copy[:] = frame[:]

                print("hello")
                with shmNamesLock:
                    shmNames[shm.name] = True

                imageData = ImageData(SharedImage(shm.name, copy.dtype, copy.shape), bboxes)
                outQueue.put(imageData)

    cap.release()
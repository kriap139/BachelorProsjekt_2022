import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from src.Backend.DataClasses import ImageData


def freeAllShmInImageDataQueue(queue):
    while queue.empty() is False:
        imgData: ImageData = queue.get()

        if imgData is not None:
            shmName = imgData.sharedImg.memName

            shm = SharedMemory(name=shmName)
            shm.close()
            shm.unlink()


def freeShmFromImageData(imageData: ImageData):
    shmName = imageData.sharedImg.memName

    if shmName is not None:
        shm = SharedMemory(name=shmName)
        shm.close()
        shm.unlink()


def freeShm(shm: SharedMemory):
    if shm is not None:
        shm.close()
        shm.unlink()

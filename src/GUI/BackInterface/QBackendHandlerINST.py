from src.GUI.BackInterface.QBackendHandler import QBackendHandler, mainExitEvent
import signal

# Instance needs to be started before it is passed to QValveStateDetector
QBackendHandlerInstance = QBackendHandler()
QBackendHandlerInstance.start()
# ----------------------------------------------------


def mainSigintHandler(signum, frame):
    mainExitEvent.set()
    QBackendHandlerInstance.shutdown()


def mainSigTermHandler(signum, frame):
    mainSigintHandler(signum, frame)


signal.signal(signal.SIGINT, mainSigintHandler)
signal.signal(signal.SIGTERM, mainSigTermHandler)

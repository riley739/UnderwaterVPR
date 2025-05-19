from PyQt5.QtCore import pyqtSignal, QObject

# ──────────────────────────────────────────────
# 📡 Signal Communicator
# ──────────────────────────────────────────────
class Communicator(QObject):
    sonar_signal = pyqtSignal(object)
    pose_signal = pyqtSignal(float, float, float)
    image_signal = pyqtSignal(object)
    shutdown_signal = pyqtSignal()
    loopclosure_signal = pyqtSignal(object)




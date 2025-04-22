from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt 
import cv2
# ──────────────────────────────────────────────
# 🖼️ Image Display Widget (reusable)
# ──────────────────────────────────────────────
class ImageDisplayWidget(QWidget):
    def __init__(self, label_text=""):
        super().__init__()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(True)

        layout = QVBoxLayout()
        if label_text:
            label = QLabel(label_text)
            layout.addWidget(label)
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def update_image(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape

        target_height = self.image_label.height()
        scale_factor = target_height / h
        new_width = int(w * scale_factor)
        resized_img = cv2.resize(img_rgb, (new_width, target_height), interpolation=cv2.INTER_AREA)
        
        h, w, ch = resized_img.shape
        bytes_per_line = ch * w
        qimg = QImage(resized_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

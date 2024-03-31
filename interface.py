import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
from PyQt5.QtCore import Qt
import numpy as np
from extract import extract
from separate import separate
from PyQt5.QtGui import QPixmap, QImage
import os
import uuid
from skimage import measure

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_path = None
        self.image = None
        self.modified_image = None
        self.mode = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Development od document stamp and signature expertise system")
        self.setGeometry(100, 100, 1500, 800)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #e0e5ec; /* Lovely shade of gray-blue */
            }
            QLabel {
                color: #333333; /* Dark gray text */
                font-size: 20px;
            }
            QPushButton {
                background-color: #778899; /* Gray-blue button */
                color: white; /* White text */
                font-weight: bold;
                border: none;
                padding: 12px 24px;
                margin: 10px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #5f7c8a; /* Darker gray-blue on hover */
            }
        """)

        self.title_label = QLabel("Development of document stamp and signature expertise system",  self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 36px; font-weight: bold; margin-top: 40px; color: #4b6171;") # Darker gray-blue

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.loadImage)

        self.detect_button = QPushButton("Detect Automatically", self)
        self.detect_button.clicked.connect(self.detectAutomatically)
        self.detect_button.setEnabled(False)

        self.separate_button = QPushButton("Separate Stamps and Signature", self)
        self.separate_button.clicked.connect(self.separateStampsAndSignature)
        self.separate_button.setEnabled(False)

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.load_button)
        self.button_layout.addWidget(self.detect_button)
        self.button_layout.addWidget(self.separate_button)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.title_label)
        self.main_layout.addWidget(self.label)
        self.main_layout.addLayout(self.button_layout)

        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

    def loadImage(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)",
                                                   options=options)
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path)
            self.image = cv2.resize(self.image, (900, 1000))
            self.modified_image = self.image.copy()
            self.displayImage()
            self.detect_button.setEnabled(True)
            self.separate_button.setEnabled(True)            
            
    def displayImage(self):
        scaled_image = cv2.resize(self.modified_image, (0, 0), fx=0.5, fy=0.5)  # Adjust the scaling factor as needed
        if len(scaled_image.shape) == 2:  # Grayscale image
            q_img = QImage(scaled_image.data, scaled_image.shape[1], scaled_image.shape[0], QImage.Format_Grayscale8)
        else:  # Color image
            q_img = QImage(scaled_image.data, scaled_image.shape[1], scaled_image.shape[0], scaled_image.strides[0], QImage.Format_BGR888)
        self.label.setPixmap(QPixmap.fromImage(q_img))
        self.centerImage()

    def detectAutomatically(self):
        modified_image = extract(self.image)
        self.modified_image = modified_image
        original_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
        scale_factor = 0.5 
        original_image = cv2.resize(original_image, (0, 0), fx=scale_factor, fy=scale_factor)
        modified_image = cv2.resize(modified_image, (0, 0), fx=scale_factor, fy=scale_factor)
        combined_image = np.concatenate((original_image, modified_image), axis=1)
        height, width, channel = combined_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(combined_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)
        self.centerImage()

    def separateStampsAndSignature(self):
        if self.image is not None:
            sign_final_image, stamp_final_image = separate(self.image)
            self.displaySeparatedImages(sign_final_image, stamp_final_image)

    def displaySeparatedImages(self, sign_image, stamp_image):
        sign_image = cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB)
        stamp_image = cv2.cvtColor(stamp_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        scale_factor = 0.5 
        sign_image = cv2.resize(sign_image, (0, 0), fx=scale_factor, fy=scale_factor)
        stamp_image = cv2.resize(stamp_image, (0, 0), fx=scale_factor, fy=scale_factor)
        original_image = cv2.resize(original_image, (0, 0), fx=scale_factor, fy=scale_factor)
        combined_image = np.concatenate((original_image, sign_image, stamp_image), axis=1)
        height, width, channel = combined_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(combined_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)
        self.centerImage()

    def centerImage(self):
        label_height = self.label.height()
        pixmap_height = self.label.pixmap().height() if self.label.pixmap() else 0
        padding = max((label_height - pixmap_height) // 2, 0)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setContentsMargins(0, padding, 0, padding)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()


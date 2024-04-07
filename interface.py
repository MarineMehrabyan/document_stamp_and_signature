from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QRadioButton, QSlider, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QIcon
import sys
import cv2
import numpy as np
from skimage import measure
from extract import extract  # Assuming extract function is defined in 'extract.py'
from separate import separate 
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import joblib

SIZE = 224

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_path = None
        self.image = None
        self.modified_image = None
        self.initUI()
        
        self.hog_scaler = joblib.load('signature_models/hog_scaler.pkl')
        self.voting_classifier = joblib.load('signature_models/voting_classifier_model.pkl')

    def initUI(self):
        self.setWindowTitle("Image Processing App")
        self.setStyleSheet("""
            /* Main window background */
            QMainWindow {
                background-color: #777f91;
            }

            /* Styling for radio buttons */
            QRadioButton {
                color: #212020;
                background-color: #ced3de;
                border: 1px solid #555b69;
                border-radius: 5px;
                padding: 8px 12px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
                background-color: #e9edf8;
                border: 1px solid #babdc6;
            }
            QRadioButton::indicator:checked {
                background-color: #b1acc0;
                border: 1px solid #8c87a1;
            }

            /* Styling for push buttons */
            QPushButton {
                color: #212020;
                background-color: #ced3de;
                border: 1px solid #77a3a5;
                border-radius: 20px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #77a3a5;
            }

            /* Styling for labels */
            QLabel {
                color: #414145;
                background-color: #c7c1d8;
                border: 2px solid #babdc6;
                border-radius: 10px;
                padding: 20px;
            }
        """)

        self.setFixedSize(1500, 900)  # Fixed window size

        self.document_radio = QRadioButton("Detect Automatic Extraction", self)
        self.document_radio.setChecked(True)
        self.document_radio.toggled.connect(self.toggleAutomaticExtraction)

        self.overlapping_radio = QRadioButton("Separate Stamps and Signature", self)
        self.overlapping_radio.toggled.connect(self.toggleStampSeparation)

        self.signature_validation_radio = QRadioButton("Check Signature Validation", self)
        self.signature_validation_radio.toggled.connect(self.toggleSignatureValidation)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        self.load_button = QPushButton(QIcon("icons/open.png"), "Load Image", self)
        self.load_button.clicked.connect(self.loadImage)

        self.process_button = QPushButton(QIcon("icons/process.png"), "Process", self)
        self.process_button.clicked.connect(self.processImage)
        self.process_button.setEnabled(False)

        self.color_slider = QSlider(Qt.Horizontal)
        self.color_slider.setMinimum(0)
        self.color_slider.setMaximum(255)
        self.color_slider.setValue(100)  # Default value for color shade
        self.color_slider.valueChanged.connect(self.updateColorShade)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.process_button)

        radio_button_layout = QVBoxLayout()
        radio_button_layout.addWidget(self.document_radio)
        radio_button_layout.addWidget(self.overlapping_radio)
        radio_button_layout.addWidget(self.signature_validation_radio)  # Moved here

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.color_slider)
        control_layout.addStretch(1)  # Add stretch to push slider to top

        main_layout = QVBoxLayout()
        main_layout.addLayout(radio_button_layout)
        main_layout.addWidget(self.label)
        main_layout.addLayout(control_layout)  # Initially add control layout
        main_layout.addLayout(button_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Hide the color slider initially
        self.color_slider.setVisible(False)


        
    def toggleSignatureValidation(self):
        if self.signature_validation_radio.isChecked():
            self.color_slider.setVisible(False)  # Hide color slider for signature validation
        else:
            self.color_slider.setVisible(True)   # Show color slider for other processes

    def toggleAutomaticExtraction(self):
        self.color_slider.setVisible(False)
        self.process_button.setEnabled(True)

    def toggleStampSeparation(self):
        if self.overlapping_radio.isChecked():
            self.color_slider.setVisible(True)
        else:
            self.color_slider.setVisible(False)

        self.process_button.setEnabled(True)

    def loadImage(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path)
            self.image = cv2.resize(self.image, (800, 600))  # Adjusted image size
            self.modified_image = self.image.copy()
            self.displayImage()
            self.process_button.setEnabled(True)

    def displayImage(self):
        q_img = self.convertImageToQImage(self.modified_image)
        self.label.setPixmap(QPixmap.fromImage(q_img))

    def processImage(self):
        if self.document_radio.isChecked():
            self.detectAutomatically()
        elif self.overlapping_radio.isChecked():
            self.separateStampsAndSignature()
        elif self.signature_validation_radio.isChecked():
            self.validateSignature()

    def validateSignature(self):
        if self.image_path is not None:
            signature_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            signature_image = cv2.resize(signature_image, (SIZE, SIZE))
            signature_hog_features = hog(signature_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
            signature_hog_features_scaled = self.hog_scaler.transform(signature_hog_features.reshape(1, -1))
            prediction = self.voting_classifier.predict(signature_hog_features_scaled)
            if prediction == 0:
                QMessageBox.information(self, "Signature Validation", "The signature is classified as REAL.")
            else:
                QMessageBox.information(self, "Signature Validation", "The signature is classified as FORGED.")
        else:
            QMessageBox.warning(self, "Error", "Please load an image first.")
            
    def detectAutomatically(self):
        if self.image is not None:
            modified_image = extract(self.image)
            self.modified_image = modified_image
            self.displayImage()

    def separateStampsAndSignature(self):
        if self.image is not None:
            channel_value = self.color_slider.value()
            sign_final_image, stamp_final_image = separate(self.image, channel_value)
            combined_image = np.concatenate((sign_final_image, stamp_final_image), axis=1)
            self.modified_image = combined_image
            self.displayImage()

    def saveResult(self):
        if self.modified_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)")
            if file_path:
                cv2.imwrite(file_path, self.modified_image)
                QMessageBox.information(self, "Image Saved", "Result image saved successfully!")

    def updateColorShade(self):
        value = self.color_slider.value()
        if self.overlapping_radio.isChecked():
            self.separateStampsAndSignature()

    def convertImageToQImage(self, image):
        if image is None:
            return QImage()

        if len(image.shape) == 2:  # Grayscale image
            q_img = QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QImage.Format_Grayscale8)
        else:  # Color image
            q_img = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_BGR888)

        return q_img


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

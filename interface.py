import sys
import cv2
from PyQt5.QtWidgets import QApplication,QHBoxLayout, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
import os
import uuid
from extract import extract


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_path = None
        self.image = None
        self.modified_image = None
        self.mode = None
        self.start_point = None
        self.end_point = None
        self.signatures = []
        self.stamps = []

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Signature and Stamp Refinement")
        self.setGeometry(100, 100, 1500, 800)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        self.load_button = QPushButton("Load", self)
        self.load_button.setFixedWidth(250) 
        self.load_button.clicked.connect(self.loadImage)

        self.select_signatures_button = QPushButton("Signatures", self)
        self.select_signatures_button.setFixedWidth(250)  
        self.select_signatures_button.clicked.connect(self.selectSignatures)
        self.select_signatures_button.setEnabled(False)

        self.select_stamps_button = QPushButton("Stamps", self)
        self.select_stamps_button.setFixedWidth(250)  
        self.select_stamps_button.clicked.connect(self.selectStamps)
        self.select_stamps_button.setEnabled(False)

        self.save_button = QPushButton("Save", self)
        self.save_button.setFixedWidth(250)  
        self.save_button.clicked.connect(self.saveBoxes)
        self.save_button.setEnabled(False)

        self.detect_button = QPushButton("Detect Automatically", self)
        self.detect_button.setFixedWidth(250)  
        self.detect_button.clicked.connect(self.detectAutomatically)
        self.detect_button.setEnabled(False)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.load_button)
        buttons_layout.addWidget(self.select_signatures_button)
        buttons_layout.addWidget(self.select_stamps_button)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.detect_button)
        buttons_layout.addStretch()  
        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(buttons_widget) 
        main_layout.addWidget(self.label)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)


    def loadImage(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path)
            self.image = cv2.resize(self.image, (700, 800))
            self.modified_image = self.image.copy()
            self.displayImage()
            self.select_signatures_button.setEnabled(True)
            self.select_stamps_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.detect_button.setEnabled(True)

    def displayImage(self):
        
        height, width, channel = self.image.shape
        bytes_per_line = 3 * width
        q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(q_img)
        self.label.setPixmap(pixmap)

    def selectSignatures(self):
        self.mode = 'signatures'
        self.label.mousePressEvent = self.startDrawing
        self.label.mouseMoveEvent = self.continueDrawing
        self.label.mouseReleaseEvent = self.endDrawing

    def selectStamps(self):
        self.mode = 'stamps'
        self.label.mousePressEvent = self.startDrawing
        self.label.mouseMoveEvent = self.continueDrawing
        self.label.mouseReleaseEvent = self.endDrawing

    def startDrawing(self, event):
        if self.image is not None:
            if event.buttons() == Qt.LeftButton:
                self.start_point = event.pos()

    def continueDrawing(self, event):
        if self.image is not None and self.start_point is not None:
            self.end_point = event.pos()
            self.drawRectanglePreview()

    def endDrawing(self, event):
        if self.image is not None and self.start_point is not None and self.end_point is not None:
            if event.button() == Qt.LeftButton:
                self.drawRectanglePreview()
                if self.mode == 'signatures':
                    self.signatures.append((self.start_point, self.end_point))
                elif self.mode == 'stamps':
                    self.stamps.append((self.start_point, self.end_point))
                self.start_point = None
                self.end_point = None

    def drawRectanglePreview(self):
        if self.image is not None and self.start_point is not None and self.end_point is not None:
            temp_image = self.modified_image.copy()
            for start, end in self.signatures:
                cv2.rectangle(temp_image, (start.x(), start.y()), (end.x(), end.y()), (0, 0, 255), 2)
            for start, end in self.stamps:
                cv2.rectangle(temp_image, (start.x(), start.y()), (end.x(), end.y()), (255, 0, 0), 2)
            cv2.rectangle(temp_image, (self.start_point.x(), self.start_point.y()),
                          (self.end_point.x(), self.end_point.y()), (0, 255, 0), 2)

            height, width, channel = temp_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(temp_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            pixmap = QPixmap(q_img)
            self.label.setPixmap(pixmap)

    def saveBoxes(self):
        self.saveImages("manual_exstracted_signatures", self.signatures)
        self.saveImages("manual_exstracted_stamps", self.stamps)

    def saveImages(self, subdir, boxes):
        if self.image is not None:
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            for start, end in boxes:
                selected_region = self.image[start.y():end.y(), start.x():end.x()]
                filename = os.path.join(subdir, f"{subdir}_{uuid.uuid4().hex[:8]}.png")
                cv2.imwrite(filename, selected_region)

        self.start_point = None
        self.end_point = None


    def detectAutomatically(self):
        modified_image = extract(self.image)
        self.displayModifiedImage(modified_image)
        
    def displayModifiedImage(self, modified_image):
        modified_image_rgb = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
        height, width, channel = modified_image_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(modified_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.label.setPixmap(pixmap)
        self.label.adjustSize()  
        

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

               


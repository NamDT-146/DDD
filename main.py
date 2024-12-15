from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QDialog, QLineEdit, QMainWindow, QWidget, QStackedLayout, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox
from PyQt6.QtCore import QTimer , pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QIcon
import cv2
import sys
from datetime import datetime
import pandas as pd
import os
import numpy as np
import torch

class MainWindow(QMainWindow):
    stack_navigator = QStackedLayout()
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    threshold_rate = 0.5
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Detection System")
        self.setGeometry(400, 100, 1000, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: lightpink;
            }   
            QMenuBar {
                background-color: lightpink
            }
            QStatusBar {
                background-color: purple;
                color: lightpink;
            }
            QLabel {
                border: 2px solid purple;
            }
            QPushButton {
                background-color: lightpink;
                color: purple;
                border: 2px solid purple;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: purple;
                color: lightpink;
            }
            QPushButton:pressed {
                background-color: violet;
                color: mangenta;
            }
            """)
        self.setWindowIcon(QIcon("tải xuống.png"))


        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.stack_navigator)

        # Pages
        self.option_page = OptionPage()
        self.livetime_page = LiveTimePage()
        self.video_page = VideoPage()

        self.stack_navigator.addWidget(self.option_page)
        self.stack_navigator.addWidget(self.livetime_page)
        self.stack_navigator.addWidget(self.video_page)

        # Set first page
        self.stack_navigator.setCurrentIndex(0)

class OptionPage(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Create the two buttons
        self.livetime_button = QPushButton("LiveTimePage")
        self.video_button = QPushButton("VideoPage")

        # Add the buttons to the layout
        self.layout.addWidget(self.livetime_button)
        self.layout.addWidget(self.video_button)

        # Connect the buttons to their respective actions
        self.livetime_button.clicked.connect(self.show_livetime_page)
        self.video_button.clicked.connect(self.show_video_page)

    def show_livetime_page(self):
        MainWindow.stack_navigator.setCurrentIndex(1)  # Index of LiveTimePage

    def show_video_page(self):
        MainWindow.stack_navigator.setCurrentIndex(2)  # Index of VideoPage


class LiveTimePage(QWidget):
    def __init__(self):
        super().__init__()
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Layout settings
        self.hbox = QHBoxLayout()
        self.setLayout(self.hbox)    

        self.cam_label = QLabel()
        self.cam_label.setFixedSize(640, 480)
        self.hbox.addWidget(self.cam_label)

        # Timer for updating frame
        self.total = 0
        self.drowsy = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)   
    def update_frame(self):
        ret, ogriginal_frame = self.cam.read()
        if ret:
            frame = cv2.flip(ogriginal_frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qimg = QImage(frame.data, w, h, w * ch, QImage.Format.Format_RGB888)
            qpix = QPixmap.fromImage(qimg)
            self.cam_label.setPixmap(qpix)
            result = MainWindow.model(ogriginal_frame)
            prediction = result.pandas().xyxy[0]
            print(prediction)
            self.total += 1
            if prediction['confidence'][0] > MainWindow.threshold_rate:
                self.drowsy += 1
            if self.total % 100 == 0:
                if self.drowsy/self.total > MainWindow.threshold_rate:
                    QMessageBox.warning(self, "Warning", "Drowsiness detected!")
class VideoPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Upload Video")

        # Layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Label
        self.label = QLabel("Please select a video file:")
        layout.addWidget(self.label)

        # Button
        self.button = QPushButton("Browse")
        self.button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.button)

        # Video Path (to store the selected video path)
        self.video_path = None
        self.layout.addWidget(self.label)
    def open_file_dialog(self):
        self.video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi)")
        if self.video_path:
            self.label.setText(f"Selected video: {self.video_path}")
    
    def predict_video(self):
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                total = 0
                drowsy = 0
                if not ret:
                    break
                result = MainWindow.model(frame)
                prediction = result.pandas().xyxy[0]
                print(prediction)
                total += 1
                if prediction['confidence'][0] > MainWindow.threshold_rate:
                    drowsy += 1
            if drowsy/total > MainWindow.threshold_rate:
                QMessageBox.warning(self, "Warning", "Drowsiness detected!")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
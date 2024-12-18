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
import pathlib
import torchvision
from torchvision import transforms
import winsound
pathlib.PosixPath = pathlib.WindowsPath

class MainWindow(QMainWindow):
    stack_navigator = QStackedLayout()
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    classes = ['awake', 'drowsy']
    colors = {
        'awake': (0, 255, 0),   # Green for awake
        'drowsy': (0, 0, 255)    # Red for drowsy
    }
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

        # Create buttons
        self.livetime_button = QPushButton("LiveTimePage")
        self.video_button = QPushButton("VideoPage")

        # Add buttons to layout
        self.layout.addWidget(self.livetime_button)
        self.layout.addWidget(self.video_button)

        # Connect buttons to actions
        self.livetime_button.clicked.connect(self.show_livetime_page)
        self.video_button.clicked.connect(self.show_video_page)

    def show_livetime_page(self):
        MainWindow.stack_navigator.setCurrentIndex(1)  # Switch to LiveTimePage
        MainWindow.stack_navigator.widget(1).start_camera()  # Start the camera

    def show_video_page(self):
        MainWindow.stack_navigator.widget(1).stop_camera()  # Stop the camera before switching
        MainWindow.stack_navigator.setCurrentIndex(2)       # Switch to VideoPage



class LiveTimePage(QWidget):
    def __init__(self):
        super().__init__()
        self.cam = None  # Camera will be initialized when the page is opened
        self.timer = QTimer()

        # Layout settings
        self.hbox = QHBoxLayout()
        self.setLayout(self.hbox)

        self.cam_label = QLabel()
        self.cam_label.setFixedSize(640, 480)
        self.hbox.addWidget(self.cam_label)

        # Timer settings
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        """Initialize the camera and start the timer."""
        if self.cam is None:  # Open camera only if not already open
            self.cam = cv2.VideoCapture(0)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.timer.start(10)  # Start the timer for frame updates

    def stop_camera(self):
        """Release the camera and stop the timer."""
        if self.cam is not None:
            self.timer.stop()
            self.cam.release()
            self.cam = None
            self.cam_label.clear()  # Clear the label to remove the last frame

    def detect_drowsiness(self, frame):
        """Run detection on a single frame."""
        results = MainWindow.model(frame)  # YOLOv5 inference
        detections = results.pandas().xyxy[0]  # Results in pandas DataFrame

        max_confidence = 0
        detected_class = None

        for _, row in detections.iterrows():
            confidence = row['confidence']
            if confidence < MainWindow.threshold_rate:
                continue  # Skip low-confidence detections
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            class_name = row['name']

            if class_name in MainWindow.classes:
                color = MainWindow.colors[class_name]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Display label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                max_confidence = max(max_confidence, confidence)
                detected_class = class_name

        return detected_class, frame

    def update_frame(self):
        """Capture a frame from the camera and process it."""
        ret, original_frame = self.cam.read()
        if ret:
            frame = cv2.flip(original_frame, 1)
            detection_result, frame = self.detect_drowsiness(frame)

            # Convert frame to QImage and display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qimg = QImage(frame.data, w, h, w * ch, QImage.Format.Format_RGB888)
            qpix = QPixmap.fromImage(qimg)
            self.cam_label.setPixmap(qpix)

            # Alert if "drowsy" detected
            if detection_result == 'drowsy':
                winsound.Beep(1000, 1000)
                QMessageBox.warning(self, "Warning", "Drowsiness detected!")

class VideoPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Upload and Process Video")

        # Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Instructions label
        self.label = QLabel("Please select a video file:")
        self.layout.addWidget(self.label)

        # Browse button
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.browse_button)

        # QLabel for video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)  # Set video display size
        self.layout.addWidget(self.video_label)

        # Start button to process video
        self.start_button = QPushButton("Start Detection")
        self.start_button.setEnabled(False)  # Disabled until a video is loaded
        self.start_button.clicked.connect(self.process_video)
        self.layout.addWidget(self.start_button)

        # Video path variable
        self.video_path = None

    def open_file_dialog(self):
        """Open a file dialog to select a video file."""
        self.video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if self.video_path:
            self.label.setText(f"Selected video: {self.video_path}")
            self.start_button.setEnabled(True)  # Enable the Start button

    def process_video(self):
        """Process the video frame-by-frame to detect drowsiness and display it."""
        if not self.video_path:
            QMessageBox.critical(self, "Error", "No video file selected!")
            return

        # Open the video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open the video file.")
            return

        total_frames = 0
        drowsy_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:  # End of the video
                break

            total_frames += 1

            # Detect drowsiness and draw bounding boxes
            detected_class, frame = self.detect_drowsiness(frame)
            frame = cv2.resize(frame, (640, 480))  # Resize for display

            # Count drowsy frames
            if detected_class == "drowsy":
                drowsy_frames += 1

            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
            qpix = QPixmap.fromImage(qimg)

            # Display the frame in the QLabel
            self.video_label.setPixmap(qpix)
            QtWidgets.QApplication.processEvents()  # Update the GUI

        cap.release()  # Release the video file

        # Display the final result
        drowsy_percentage = (drowsy_frames / total_frames) * 100
        if drowsy_percentage > 50:  # Adjust the threshold as needed
            QMessageBox.warning(self, "Warning", f"Drowsiness detected in {drowsy_percentage:.2f}% of the video frames!")
        else:
            QMessageBox.information(self, "Result", f"The driver appears awake in {100 - drowsy_percentage:.2f}% of the video frames.")
        
    def detect_drowsiness(self, frame):
        """Run YOLOv5 detection on a single video frame and draw bounding boxes for awake and drowsy."""
        try:
            # Perform inference using the YOLOv5 model
            results = MainWindow.model(frame)
            detections = results.pandas().xyxy[0]  # Get detections as a pandas DataFrame

            detected_class = "awake"  # Default state is 'awake'

            # Loop through all detections and draw bounding boxes
            for _, row in detections.iterrows():
                confidence = row['confidence']
                class_name = row['name']

                # Only process classes we are interested in
                if confidence >= MainWindow.threshold_rate and class_name in MainWindow.classes:
                    # Draw bounding box for the current detection
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    color = MainWindow.colors[class_name]  # Use color based on class
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Display the label and confidence
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # If any detection is 'drowsy', mark the frame as drowsy
                    if class_name == "drowsy":
                        detected_class = "drowsy"

            return detected_class, frame
        except Exception as e:
            print(f"Error in detection: {e}")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
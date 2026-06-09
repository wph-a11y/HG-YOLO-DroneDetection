import sys
import cv2
import time
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QSlider, QGroupBox,
                             QTextEdit, QFileDialog, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QDateTime
from PyQt5.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO

STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}
QGroupBox {
    border: 2px solid #3c3c3c;
    border-radius: 8px;
    margin-top: 10px;
    color: #e0e0e0;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QLabel {
    color: #cccccc;
}
QPushButton {
    background-color: #007acc;
    color: white;
    border-radius: 5px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #0098ff;
}
QPushButton:pressed {
    background-color: #005c99;
}
QPushButton#stopBtn {
    background-color: #d32f2f;
}
QPushButton#fileBtn {
    background-color: #e6a23c;
    color: #000;
}
QComboBox {
    background-color: #2d2d2d;
    color: #ffffff;
    border: 1px solid #3c3c3c;
    border-radius: 5px;
    padding: 5px;
}
QComboBox::drop-down {
    border: 0px;
}
QTextEdit {
    background-color: #2d2d2d;
    color: #00ff00;
    border: 1px solid #3c3c3c;
    font-family: Consolas, monospace;
}
QSlider::groove:horizontal {
    border: 1px solid #3c3c3c;
    height: 8px;
    background: #2d2d2d;
    margin: 2px 0;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #007acc;
    border: 1px solid #007acc;
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 9px;
}
"""


class DetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray, dict, float)
    task_finished_signal = pyqtSignal()

    def __init__(self, model_path, source, mode):
        super().__init__()
        self.model_path = model_path
        self.source = source
        self.mode = mode
        self.running = True
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Cannot load custom model: {e}, switching to yolov8n.pt...")
            self.model = YOLO('yolov8n.pt')

    def run(self):
        if self.mode == 'image':
            start_time = time.time()
            frame = cv2.imread(self.source)
            if frame is None:
                return

            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
            annotated_frame = results[0].plot()

            stats = self.get_stats(results)
            inference_time = (time.time() - start_time) * 1000

            self.change_pixmap_signal.emit(frame, annotated_frame, stats, inference_time)
            self.task_finished_signal.emit()

        else:
            cap = cv2.VideoCapture(self.source)

            while self.running and cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
                annotated_frame = results[0].plot()

                stats = self.get_stats(results)
                inference_time = (time.time() - start_time) * 1000

                self.change_pixmap_signal.emit(frame, annotated_frame, stats, inference_time)

                if self.mode == 'video':
                    time.sleep(0.02)
                else:
                    time.sleep(0.005)

            cap.release()
            self.task_finished_signal.emit()

    def get_stats(self, results):
        stats = {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            stats[cls_name] = stats.get(cls_name, 0) + 1
        return stats

    def stop(self):
        self.running = False
        self.wait()

    def update_params(self, conf, iou):
        self.conf_threshold = conf
        self.iou_threshold = iou


class YOLOv8App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Detection System (Image/Video/Camera)")
        self.setGeometry(100, 100, 1400, 850)
        self.setStyleSheet(STYLESHEET)

        self.model_path = os.path.join('weights', 'best.pt')
        self.thread = None
        self.last_annotated_frame = None
        self.selected_file_path = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(320)
        sidebar_layout = QVBoxLayout()
        sidebar_widget.setLayout(sidebar_layout)

        title_label = QLabel("Detection Control Center")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title_label)

        source_group = QGroupBox("Input Source")
        source_layout = QVBoxLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Live Camera", "Image Detection", "Video File Detection"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_change)

        self.btn_file = QPushButton("Select File...")
        self.btn_file.setObjectName("fileBtn")
        self.btn_file.setVisible(False)
        self.btn_file.clicked.connect(self.select_file)

        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #888; font-size: 10px;")

        source_layout.addWidget(self.mode_combo)
        source_layout.addWidget(self.btn_file)
        source_layout.addWidget(self.file_label)
        source_group.setLayout(source_layout)
        sidebar_layout.addWidget(source_group)

        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout()

        self.conf_label = QLabel("Confidence: 0.25")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.update_thresholds)

        self.iou_label = QLabel("IOU: 0.45")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 99)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(self.update_thresholds)

        param_layout.addWidget(self.conf_label)
        param_layout.addWidget(self.conf_slider)
        param_layout.addWidget(self.iou_label)
        param_layout.addWidget(self.iou_slider)
        param_group.setLayout(param_layout)
        sidebar_layout.addWidget(param_group)

        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        sidebar_layout.addWidget(log_group)

        btn_layout = QVBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop = QPushButton("Stop / Reset")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_snap = QPushButton("Snapshot")
        self.btn_snap.clicked.connect(self.snapshot)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_snap)
        sidebar_layout.addLayout(btn_layout)

        video_layout = QVBoxLayout()

        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Original Input"))
        header_layout.addWidget(QLabel("YOLO Detection Result"))

        display_layout = QHBoxLayout()
        self.label_orig = QLabel()
        self.label_orig.setStyleSheet("background-color: #000; border: 1px solid #444;")
        self.label_orig.setFixedSize(600, 450)
        self.label_orig.setAlignment(Qt.AlignCenter)
        self.label_orig.setText("No Signal")

        self.label_detect = QLabel()
        self.label_detect.setStyleSheet("background-color: #000; border: 1px solid #444;")
        self.label_detect.setFixedSize(600, 450)
        self.label_detect.setAlignment(Qt.AlignCenter)
        self.label_detect.setText("Waiting for Detection...")

        display_layout.addWidget(self.label_orig)
        display_layout.addWidget(self.label_detect)

        self.status_label = QLabel("System Ready")
        self.status_label.setStyleSheet("color: #888;")

        video_layout.addLayout(header_layout)
        video_layout.addLayout(display_layout)
        video_layout.addWidget(self.status_label)

        main_layout.addWidget(sidebar_widget)
        main_layout.addLayout(video_layout)

    def on_mode_change(self):
        mode = self.mode_combo.currentText()
        if mode == "Live Camera":
            self.btn_file.setVisible(False)
            self.file_label.setText("Using default camera (0)")
            self.selected_file_path = None
        else:
            self.btn_file.setVisible(True)
            self.file_label.setText("Please select a file...")
            self.selected_file_path = None

    def select_file(self):
        mode = self.mode_combo.currentText()
        if mode == "Image Detection":
            fname, _ = QFileDialog.getOpenFileName(self, 'Select Image', '.',
                                                   'Image files (*.jpg *.png *.jpeg *.bmp)')
        elif mode == "Video File Detection":
            fname, _ = QFileDialog.getOpenFileName(self, 'Select Video', '.',
                                                   'Video files (*.mp4 *.avi *.mkv)')
        else:
            return

        if fname:
            self.selected_file_path = fname
            self.file_label.setText(os.path.basename(fname))
            self.log_text.append(f"Loaded: {os.path.basename(fname)}")
        else:
            self.file_label.setText("No file selected")

    def start_detection(self):
        mode_text = self.mode_combo.currentText()
        source = 0
        mode_key = 'camera'

        if mode_text == "Image Detection":
            mode_key = 'image'
            if not self.selected_file_path:
                QMessageBox.warning(self, "Error", "Please select an image first!")
                return
            source = self.selected_file_path

        elif mode_text == "Video File Detection":
            mode_key = 'video'
            if not self.selected_file_path:
                QMessageBox.warning(self, "Error", "Please select a video file first!")
                return
            source = self.selected_file_path

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.mode_combo.setEnabled(False)
        self.btn_file.setEnabled(False)

        self.log_text.append(
            f"[{QDateTime.currentDateTime().toString('HH:mm:ss')}] Processing: {mode_text}...")

        self.thread = DetectionThread(self.model_path, source, mode_key)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.task_finished_signal.connect(self.on_task_finished)
        self.thread.start()

    def stop_detection(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.reset_ui_state()
        self.log_text.append(f"[{QDateTime.currentDateTime().toString('HH:mm:ss')}] Task stopped")

    def on_task_finished(self):
        self.log_text.append("Processing completed.")
        self.reset_ui_state()

    def reset_ui_state(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.mode_combo.setEnabled(True)
        self.btn_file.setEnabled(True)

    def update_thresholds(self):
        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0
        self.conf_label.setText(f"Confidence: {conf:.2f}")
        self.iou_label.setText(f"IOU: {iou:.2f}")

        if self.thread:
            self.thread.update_params(conf, iou)

    def update_image(self, cv_orig, cv_detect, stats, time_cost):
        self.last_annotated_frame = cv_detect

        self.display_image(cv_orig, self.label_orig)
        self.display_image(cv_detect, self.label_detect)

        total_obj = sum(stats.values())
        self.status_label.setText(
            f"Time: {time_cost:.1f}ms | Targets: {total_obj} | {self.mode_combo.currentText()}")

    def display_image(self, cv_img, label_widget):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = qt_img.scaled(label_widget.width(), label_widget.height(), Qt.KeepAspectRatio,
                          Qt.SmoothTransformation)
        label_widget.setPixmap(QPixmap.fromImage(p))

    def snapshot(self):
        if self.last_annotated_frame is None:
            QMessageBox.warning(self, "Warning", "No frame to save!")
            return

        filename = f"Result_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.jpg"
        cv2.imwrite(filename, self.last_annotated_frame)
        self.log_text.append(f"Saved: {filename}")
        QMessageBox.information(self, "Success", f"File saved:\n{filename}")

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOv8App()
    window.show()
    sys.exit(app.exec_())

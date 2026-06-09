import sys
import cv2
import time
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QSlider, QGroupBox,
                             QTextEdit, QFileDialog, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QDateTime
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from ultralytics import YOLO

try:
    import winsound
except ImportError:
    winsound = None

STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}
QGroupBox {
    border: 2px solid #3c3c3c;
    border-radius: 8px;
    margin-top: 14px;
    color: #e0e0e0;
    font-weight: bold;
    font-size: 20px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    font-size: 20px;
}
QLabel {
    color: #cccccc;
    font-size: 20px;
}
QLabel#viewLabel {
    border: 2px solid #444;
    background-color: #000;
    font-size: 20px;
}
QLabel#viewLabel[alarm="true"] {
    border: 4px solid #ff0000;
}
QPushButton {
    background-color: #007acc;
    color: white;
    border-radius: 5px;
    padding: 12px;
    font-weight: bold;
    font-size: 20px;
    min-height: 35px;
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
QTextEdit {
    background-color: #2d2d2d;
    color: #00ff00;
    border: 1px solid #3c3c3c;
    font-family: Consolas, monospace;
    font-size: 20px;
}
QComboBox {
    font-size: 20px;
    padding: 8px;
}
QSlider::groove:horizontal {
    height: 12px;
}
QSlider::handle:horizontal {
    width: 24px;
    margin: -6px 0;
}
"""


class DetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray, dict, float, bool, str)
    task_finished_signal = pyqtSignal()

    def __init__(self, model_path, source, mode, alarm_classes):
        super().__init__()
        self.model_path = model_path
        self.source = source
        self.mode = mode
        self.alarm_classes = alarm_classes
        self.running = True
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = YOLO('yolov8n.pt')

    def run(self):
        if self.mode == 'image':
            self.process_image()
        else:
            self.process_video()

    def process_image(self):
        start_time = time.time()
        frame = cv2.imread(self.source)
        if frame is None:
            return

        annotated_frame, stats, is_alarm, alarm_msg = self.detect_frame(frame)
        inference_time = (time.time() - start_time) * 1000

        self.change_pixmap_signal.emit(frame, annotated_frame, stats, inference_time, is_alarm, alarm_msg)
        self.task_finished_signal.emit()

    def process_video(self):
        cap = cv2.VideoCapture(self.source)
        while self.running and cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, stats, is_alarm, alarm_msg = self.detect_frame(frame)
            inference_time = (time.time() - start_time) * 1000

            self.change_pixmap_signal.emit(frame, annotated_frame, stats, inference_time, is_alarm, alarm_msg)

            if self.mode == 'video':
                time.sleep(0.02)
            else:
                time.sleep(0.005)

        cap.release()
        self.task_finished_signal.emit()

    def detect_frame(self, frame):
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        annotated_frame = results[0].plot()

        stats = {}
        is_alarm = False
        alarm_msg = ""

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            stats[cls_name] = stats.get(cls_name, 0) + 1

            if cls_name in self.alarm_classes:
                is_alarm = True
                alarm_msg = cls_name

        return annotated_frame, stats, is_alarm, alarm_msg

    def stop(self):
        self.running = False
        self.wait()

    def update_params(self, conf, iou):
        self.conf_threshold = conf
        self.iou_threshold = iou


class YOLOv8App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Defense Detection System")
        self.setGeometry(100, 100, 1700, 1000)
        self.setStyleSheet(STYLESHEET)

        self.target_alarm_classes = ['drone', 'uav', 'airplane', 'bird',
                                     'drone_1', 'drone_2', 'drone_3', 'drone_others']

        self.model_path = os.path.join('weights', 'best.pt')
        self.thread = None
        self.last_annotated_frame = None
        self.selected_file_path = None
        self.last_beep_time = 0

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(420)
        sidebar_layout = QVBoxLayout()
        sidebar_widget.setLayout(sidebar_layout)

        title_label = QLabel("WARNING: Defense Monitoring Center")
        title_label.setFont(QFont("Arial", 26, QFont.Bold))
        title_label.setStyleSheet("color: #e74c3c;")
        title_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title_label)

        source_group = QGroupBox("Input Source")
        source_layout = QVBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.setFont(QFont("Arial", 20))
        self.mode_combo.addItems(["Live Camera", "Image Detection", "Video File Detection"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_change)
        self.btn_file = QPushButton("Select File...")
        self.btn_file.setFont(QFont("Arial", 20))
        self.btn_file.setObjectName("fileBtn")
        self.btn_file.setVisible(False)
        self.btn_file.clicked.connect(self.select_file)
        self.file_label = QLabel("No file selected")
        self.file_label.setFont(QFont("Arial", 20))
        self.file_label.setStyleSheet("color: #888;")
        source_layout.addWidget(self.mode_combo)
        source_layout.addWidget(self.btn_file)
        source_layout.addWidget(self.file_label)
        source_group.setLayout(source_layout)
        sidebar_layout.addWidget(source_group)

        param_group = QGroupBox("Sensitivity Settings")
        param_layout = QVBoxLayout()
        self.conf_label = QLabel("Confidence: 0.25")
        self.conf_label.setFont(QFont("Arial", 20))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.update_thresholds)
        self.iou_label = QLabel("IOU: 0.45")
        self.iou_label.setFont(QFont("Arial", 20))
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

        log_group = QGroupBox("Real-time Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 20))
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        sidebar_layout.addWidget(log_group)

        btn_layout = QVBoxLayout()
        self.btn_start = QPushButton("Start Defense")
        self.btn_start.setFont(QFont("Arial", 20, QFont.Bold))
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop = QPushButton("Stop Detection")
        self.btn_stop.setFont(QFont("Arial", 20, QFont.Bold))
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_snap = QPushButton("Snapshot")
        self.btn_snap.setFont(QFont("Arial", 20, QFont.Bold))
        self.btn_snap.clicked.connect(self.snapshot)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_snap)
        sidebar_layout.addLayout(btn_layout)

        video_layout = QVBoxLayout()
        header_layout = QHBoxLayout()
        orig_header = QLabel("Original Video")
        orig_header.setFont(QFont("Arial", 22, QFont.Bold))
        detect_header = QLabel("Detection Video")
        detect_header.setFont(QFont("Arial", 22, QFont.Bold))
        header_layout.addWidget(orig_header)
        header_layout.addWidget(detect_header)

        display_layout = QHBoxLayout()
        self.label_orig = QLabel("No Signal")
        self.label_orig.setObjectName("viewLabel")
        self.label_orig.setFixedSize(720, 540)
        self.label_orig.setAlignment(Qt.AlignCenter)
        self.label_orig.setFont(QFont("Arial", 22))

        self.label_detect = QLabel("Waiting for Detection...")
        self.label_detect.setObjectName("viewLabel")
        self.label_detect.setFixedSize(720, 540)
        self.label_detect.setAlignment(Qt.AlignCenter)
        self.label_detect.setFont(QFont("Arial", 22))

        display_layout.addWidget(self.label_orig)
        display_layout.addWidget(self.label_detect)

        self.status_label = QLabel("System Ready - Safe")
        self.status_label.setFont(QFont("Arial", 22))
        self.status_label.setStyleSheet("color: #00ff00;")

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

    def select_file(self):
        mode = self.mode_combo.currentText()
        ftype = 'Image files (*.jpg *.png)' if mode == "Image Detection" else 'Video files (*.mp4 *.avi)'
        fname, _ = QFileDialog.getOpenFileName(self, 'Select File', '.', ftype)
        if fname:
            self.selected_file_path = fname
            self.file_label.setText(os.path.basename(fname))

    def start_detection(self):
        mode_text = self.mode_combo.currentText()
        source = 0
        mode_key = 'camera'

        if mode_text != "Live Camera":
            if not self.selected_file_path:
                QMessageBox.warning(self, "Error", "Please select a file first!")
                return
            source = self.selected_file_path
            mode_key = 'image' if mode_text == "Image Detection" else 'video'

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.mode_combo.setEnabled(False)
        self.btn_file.setEnabled(False)
        self.log_text.append(f"[{self.get_time()}] System started...")

        self.thread = DetectionThread(self.model_path, source, mode_key, self.target_alarm_classes)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.task_finished_signal.connect(self.on_task_finished)
        self.thread.start()

    def stop_detection(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.reset_ui_state()
        self.log_text.append(f"[{self.get_time()}] Detection stopped")
        self.clear_alarm_state()

    def on_task_finished(self):
        self.log_text.append("Task completed.")
        self.reset_ui_state()
        self.clear_alarm_state()

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

    def update_image(self, cv_orig, cv_detect, stats, time_cost, is_alarm, alarm_msg):
        self.last_annotated_frame = cv_detect
        self.display_image(cv_orig, self.label_orig)
        self.display_image(cv_detect, self.label_detect)

        if is_alarm:
            self.trigger_alarm(alarm_msg)
            self.status_label.setText(f"WARNING! Target Detected: {alarm_msg} !!!")
            self.status_label.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 24px;")
        else:
            self.clear_alarm_state()
            total = sum(stats.values())
            self.status_label.setText(f"Status: Safe | Time: {time_cost:.1f}ms | Targets: {total}")
            self.status_label.setStyleSheet("color: #00ff00; font-size: 22px;")

    def trigger_alarm(self, msg):
        current_time = time.time()

        self.label_detect.setProperty("alarm", "true")
        self.label_detect.style().unpolish(self.label_detect)
        self.label_detect.style().polish(self.label_detect)

        if winsound and (current_time - self.last_beep_time > 1.0):
            winsound.Beep(1000, 200)
            self.last_beep_time = current_time

            log_html = f"<span style='color:#ff5555;'>[{self.get_time()}] WARNING: Drone detected: {msg}</span>"
            self.log_text.append(log_html)

    def clear_alarm_state(self):
        self.label_detect.setProperty("alarm", "false")
        self.label_detect.style().unpolish(self.label_detect)
        self.label_detect.style().polish(self.label_detect)

    def display_image(self, cv_img, label_widget):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        qt_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        p = qt_img.scaled(label_widget.width(), label_widget.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label_widget.setPixmap(QPixmap.fromImage(p))

    def snapshot(self):
        if self.last_annotated_frame is None:
            return
        fname = f"Evidence_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.jpg"
        cv2.imwrite(fname, self.last_annotated_frame)
        self.log_text.append(f"Evidence saved: {fname}")
        QMessageBox.information(self, "Snapshot Success", f"File saved to:\n{fname}")

    def get_time(self):
        return QDateTime.currentDateTime().toString('HH:mm:ss')

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOv8App()
    window.show()
    sys.exit(app.exec_())

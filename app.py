import sys
import cv2
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QLineEdit, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer
from ultralytics import YOLO
import numpy as np

# ------------------ LOGIN WINDOW ------------------
class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shrimp larvae counting - Login")
        self.showMaximized()  # Fullscreen

        layout = QVBoxLayout()
        title = QLabel("Shrimp larvae counting")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 40px; font-weight: bold;")
        layout.addWidget(title)

        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Username")
        layout.addWidget(self.username_edit)

        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Password")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password_edit)

        self.login_btn = QPushButton("Login")
        self.login_btn.setStyleSheet("font-size: 20px; padding: 10px;")
        self.login_btn.clicked.connect(self.check_login)
        layout.addWidget(self.login_btn)

        layout.addStretch()
        self.setLayout(layout)

    def check_login(self):
        username = self.username_edit.text().strip()
        password = self.password_edit.text().strip()
        if username == "UCL" and password == "123":
            self.accept_login()
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password!")

    def accept_login(self):
        self.hide()
        self.menu = MenuPage(self)
        self.menu.showMaximized()

# ------------------ MENU PAGE ------------------
class MenuPage(QWidget):
    def __init__(self, login_window):
        super().__init__()
        self.login_window = login_window
        self.setWindowTitle("Menu - Shrimp Counting")
        self.showMaximized()

        layout = QVBoxLayout()
        title = QLabel("Pilih Mode Input")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 35px; font-weight: bold;")
        layout.addWidget(title)

        self.btn_image = QPushButton("🖼️ Input Gambar")
        self.btn_video = QPushButton("🎞️ Input Video")
        self.btn_camera = QPushButton("📷 Real-time Kamera")
        self.back_btn = QPushButton("⬅️ Logout")

        for btn in [self.btn_image, self.btn_video, self.btn_camera, self.back_btn]:
            btn.setStyleSheet("font-size: 25px; padding: 15px;")
            layout.addWidget(btn)

        layout.addStretch()
        self.setLayout(layout)

        self.btn_image.clicked.connect(self.open_image_mode)
        self.btn_video.clicked.connect(self.open_video_mode)
        self.btn_camera.clicked.connect(self.open_camera_mode)
        self.back_btn.clicked.connect(self.back_to_login)

    def open_image_mode(self):
        self.hide()
        self.image_app = ContainerCountingApp(self, mode="image")
        self.image_app.showMaximized()

    def open_video_mode(self):
        self.hide()
        self.video_app = ContainerCountingApp(self, mode="video")
        self.video_app.showMaximized()

    def open_camera_mode(self):
        self.hide()
        self.cam_app = ContainerCountingApp(self, mode="camera")
        self.cam_app.showMaximized()

    def back_to_login(self):
        self.hide()
        self.login_window.showMaximized()

# ------------------ MAIN APP ------------------
class ContainerCountingApp(QWidget):
    def __init__(self, menu_page, mode="image"):
        super().__init__()
        self.menu_page = menu_page
        self.mode = mode
        self.setWindowTitle(f"Shrimp Counting - {mode.capitalize()} Mode")
        self.showMaximized()

        # ====== MAIN HORIZONTAL LAYOUT ======
        main_layout = QHBoxLayout()

        # -------- LEFT PANEL (Menu) --------
        left_layout = QVBoxLayout()

        self.title_label = QLabel(f"Mode: {mode.upper()}")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 28px; font-weight: bold;")
        left_layout.addWidget(self.title_label)

        if mode in ["image", "video"]:
            self.file_path_edit = QLineEdit()
            self.file_path_edit.setPlaceholderText("Pilih file")
            left_layout.addWidget(self.file_path_edit)

            self.select_btn = QPushButton("📂 Pilih File")
            self.select_btn.setStyleSheet("font-size: 20px; padding: 10px;")
            self.select_btn.clicked.connect(self.select_file)
            left_layout.addWidget(self.select_btn)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path YOLO model (.pt)")
        left_layout.addWidget(self.model_path_edit)

        self.select_model_btn = QPushButton("🧠 Pilih Model (.pt)")
        self.select_model_btn.setStyleSheet("font-size: 20px; padding: 10px;")
        self.select_model_btn.clicked.connect(self.select_model)
        left_layout.addWidget(self.select_model_btn)

        if mode in ["image", "video"]:
            self.count_btn = QPushButton("▶️ Jalankan Deteksi")
            self.count_btn.setStyleSheet("font-size: 20px; padding: 10px;")
            self.count_btn.clicked.connect(self.count_shrimp)
            left_layout.addWidget(self.count_btn)

        self.result_label = QLabel("Jumlah shrimp: 0")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px; font-weight: bold; color: blue;")
        left_layout.addWidget(self.result_label)

        # Tombol baru untuk pengelompokan ukuran
        self.group_btn = QPushButton("📊 Kelompokkan Ukuran")
        self.group_btn.setStyleSheet("font-size: 20px; padding: 10px;")
        self.group_btn.clicked.connect(self.group_by_size)
        left_layout.addWidget(self.group_btn)

        self.group_label = QLabel("Hasil pengelompokan: -")
        self.group_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.group_label.setStyleSheet("font-size: 20px; color: green;")
        left_layout.addWidget(self.group_label)

        self.back_btn = QPushButton("⬅️ Kembali ke Menu")
        self.back_btn.setStyleSheet("font-size: 20px; padding: 10px;")
        self.back_btn.clicked.connect(self.back_to_menu)
        left_layout.addWidget(self.back_btn)

        left_layout.addStretch()

        # -------- RIGHT PANEL (Preview) --------
        self.image_label = QLabel("Preview")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: black; color: white;")
        self.image_label.setFixedSize(1080, 1080)

        # -------- Add Panels to Main Layout --------
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.image_label)

        self.setLayout(main_layout)

        # -------- Variables --------
        self.file_path = None
        self.model = None
        self.cap = None
        self.timer = None
        self.max_count = 0

        if self.mode == "camera":
            self.start_camera()

    # ------------------ FILE / MODEL ------------------
    def select_file(self):
        if self.mode == "image":
            path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.png)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if path:
            self.file_path = path
            self.file_path_edit.setText(path)
            pixmap = QPixmap(path)
            self.image_label.setPixmap(pixmap)

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "Model Files (*.pt)")
        if path:
            self.model_path_edit.setText(path)
            self.model = YOLO(path)

    # ------------------ DETECTION ------------------
    def count_shrimp(self):
        file_path = self.file_path_edit.text().strip()
        if not self.model:
            self.result_label.setText("⚠️ Harap pilih model YOLO (.pt)")
            return

        if self.mode == "image":
            if not file_path:
                self.result_label.setText("⚠️ Harap pilih gambar")
                return
            results = self.model(file_path)
            img = results[0].plot(labels=False)   # ✅ tanpa label
            self.show_result(img, len(results[0].boxes))

        elif self.mode == "video":
            if not file_path:
                self.result_label.setText("⚠️ Harap pilih video")
                return
            self.cap = cv2.VideoCapture(file_path)
            self.max_count = 0
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_video_frame)
            self.timer.start(30)

    def update_video_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return
        results = self.model(frame)
        frame = results[0].plot(labels=False)   # ✅ tanpa label
        count = len(results[0].boxes)
        if count > self.max_count:
            self.max_count = count
        self.result_label.setText(f"🐟 Jumlah shrimp tertinggi: {self.max_count}")
        self.display_frame(frame)

    # ------------------ CAMERA ------------------
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.max_count = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_frame)
        self.timer.start(30)

    def update_camera_frame(self):
        if not self.cap or not self.cap.isOpened() or not self.model:
            self.result_label.setText("⚠️ Pilih model YOLO dulu")
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        results = self.model(frame)
        frame = results[0].plot(labels=False)   # ✅ tanpa label
        count = len(results[0].boxes)
        if count > self.max_count:
            self.max_count = count
        self.result_label.setText(f"🐟 Jumlah shrimp tertinggi: {self.max_count}")
        self.display_frame(frame)

    # ------------------ GROUPING ------------------
    def group_by_size(self):
        if not self.model:
            self.group_label.setText("⚠️ Harap pilih model YOLO dulu")
            return

        if self.mode == "image":
            if not self.file_path:
                self.group_label.setText("⚠️ Harap pilih gambar")
                return
            results = self.model(self.file_path)
        elif self.mode in ["video", "camera"]:
            if not self.cap or not self.cap.isOpened():
                self.group_label.setText("⚠️ Tidak ada frame tersedia")
                return
            ret, frame = self.cap.read()
            if not ret:
                self.group_label.setText("⚠️ Tidak ada frame tersedia")
                return
            results = self.model(frame)
        else:
            return

        # Ambil bounding box
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]

        # Threshold area larva (bisa disesuaikan dengan dataset nyata)
        small_thresh = 500
        large_thresh = 2000

        count_small = sum(1 for a in areas if a < small_thresh)
        count_medium = sum(1 for a in areas if small_thresh <= a <= large_thresh)
        count_large = sum(1 for a in areas if a > large_thresh)

        self.group_label.setText(
            f"Larva kecil: {count_small} | "
            f"Sedang: {count_medium} | "
            f"Besar: {count_large} | "
            f"Total: {len(areas)}"
        )

    # ------------------ UTILITY ------------------
    def display_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape

        # Batas ukuran preview
        max_w, max_h = 1080, 1080

        # Hitung rasio scaling
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        qimg = QImage(resized.data, new_w, new_h, ch * new_w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
        self.image_label.setFixedSize(new_w, new_h)  # ✅ Label ikut menyesuaikan

    def show_result(self, img, count):
        self.result_label.setText(f"🐟 Jumlah shrimp: {count}")
        self.display_frame(img)

    def back_to_menu(self):
        if self.timer:
            self.timer.stop()
        if self.cap:
            self.cap.release()
        self.hide()
        self.menu_page.showMaximized()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    login = LoginWindow()
    login.showMaximized()
    sys.exit(app.exec())

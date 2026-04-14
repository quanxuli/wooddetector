import sys
import socket
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import struct
import time
from PIL import ImageGrab # Cần cài đặt: pip install pillow

# Cấu hình mặc định
HEADER_FORMAT = 'BBBH'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
DEFAULT_PORT = 12345
DISCOVERY_PORT = 12346
MAX_PAYLOAD_SIZE = 1400 

class VideoClient(QWidget):
    def __init__(self, imgsz=640):
        super().__init__()
        self.imgsz = imgsz
        self.udp_streaming = False
        self.frame_id = 0
        self.last_frame_time = time.time()
        self.test_frame = None  # Dùng để lưu ảnh tải lên
        self.mode = "test"      # "cam", "file", "screen", "test"
        
        self.initUI()
        
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def initUI(self):
        self.setWindowTitle("Multi-Source UDP Client")
        self.setGeometry(100, 100, 800, 700)
        layout = QVBoxLayout()

        # 1. Điều khiển IP
        ip_layout = QHBoxLayout()
        self.ip_input = QLineEdit("127.0.0.1")
        self.btn_discover = QPushButton("Auto Discover")
        self.btn_discover.clicked.connect(self.auto_discover)
        ip_layout.addWidget(QLabel("Server IP:"))
        ip_layout.addWidget(self.ip_input)
        ip_layout.addWidget(self.btn_discover)
        layout.addLayout(ip_layout)

        # 2. Màn hình hiển thị
        self.result_label = QLabel("Source Closed")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("background-color: black; color: white; border: 2px solid gray;")
        self.result_label.setMinimumSize(640, 480)
        layout.addWidget(self.result_label)

        self.info_label = QLabel("FPS: -- | Latency: -- ms")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

        # 3. Các nút chọn nguồn dữ liệu
        btn_layout = QHBoxLayout()
        
        self.btn_cam = QPushButton("Use Webcam")
        self.btn_cam.clicked.connect(lambda: self.switch_mode("cam"))
        
        self.btn_file = QPushButton("Upload Image")
        self.btn_file.clicked.connect(self.upload_image)
        
        self.btn_screen = QPushButton("Capture Screen")
        self.btn_screen.clicked.connect(lambda: self.switch_mode("screen"))

        btn_layout.addWidget(self.btn_cam)
        btn_layout.addWidget(self.btn_file)
        btn_layout.addWidget(self.btn_screen)
        layout.addLayout(btn_layout)

        # 4. Điều khiển Stream
        self.btn_stream = QPushButton("Start Streaming UDP")
        self.btn_stream.setEnabled(False)
        self.btn_stream.clicked.connect(self.toggle_udp)
        layout.addWidget(self.btn_stream)

        self.setLayout(layout)

    def upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png *.jpeg)")
        if fname:
            self.test_frame = cv2.imread(fname)
            self.switch_mode("file")

    def switch_mode(self, mode):
        self.mode = mode
        if mode == "cam":
            if not self.cap: self.cap = cv2.VideoCapture(0)
        else:
            if self.cap: 
                self.cap.release()
                self.cap = None
        
        self.timer.start(30)
        self.btn_stream.setEnabled(True)
        print(f"Switched to {mode} mode")

    def auto_discover(self):
        self.btn_discover.setText("Searching...")
        QApplication.processEvents()
        dsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dsock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        dsock.settimeout(1.5)
        try:
            dsock.sendto(b"WHERE_IS_VIDEO_SERVER", ('<broadcast>', DISCOVERY_PORT))
            data, addr = dsock.recvfrom(1024)
            self.ip_input.setText(addr[0]); self.btn_discover.setText("Found!")
        except:
            self.btn_discover.setText("Not Found"); dsock.close()

    def toggle_udp(self):
        self.udp_streaming = not self.udp_streaming
        self.btn_stream.setText("Stop UDP" if self.udp_streaming else "Start UDP")

    def update_frame(self):
        frame = None
        
        if self.mode == "cam" and self.cap:
            ret, frame = self.cap.read()
        elif self.mode == "file":
            frame = self.test_frame.copy() if self.test_frame is not None else None
        elif self.mode == "screen":
            # Chụp màn hình (sử dụng Pillow)
            img = ImageGrab.grab()
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "NO SOURCE SELECTED", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        now = time.time()
        latency = int((now - self.last_frame_time) * 1000)
        self.last_frame_time = now
        self.info_label.setText(f"FPS: {int(1000/latency if latency>0 else 0)} | Latency: {latency} ms")
        
        if self.udp_streaming:
            self.send_frame_to_udp(frame)
        self.display_frame(frame)

    def send_frame_to_udp(self, frame):
        try:
            target_ip = self.ip_input.text()
            resized = cv2.resize(frame, (self.imgsz, self.imgsz))
            _, encoded = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            data = encoded.tobytes()
            num_chunks = (len(data) + MAX_PAYLOAD_SIZE - 1) // MAX_PAYLOAD_SIZE
            self.frame_id = (self.frame_id + 1) % 256
            for i in range(num_chunks):
                chunk = data[i*MAX_PAYLOAD_SIZE : (i+1)*MAX_PAYLOAD_SIZE]
                header = struct.pack(HEADER_FORMAT, self.frame_id, i, num_chunks, len(chunk))
                self.udp_socket.sendto(header + chunk, (target_ip, DEFAULT_PORT))
        except Exception as e: print(f"Send error: {e}")

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.result_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.result_label.size(), Qt.KeepAspectRatio))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    client = VideoClient(imgsz=640)
    client.show()
    sys.exit(app.exec_())
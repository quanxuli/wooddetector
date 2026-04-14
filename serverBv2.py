import sys
import socket
import cv2
import numpy as np
import struct
import time
import logging
import threading
from threading import Lock
from queue import SimpleQueue
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QSlider, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import os
import torch  # Thêm torch để kiểm tra CUDA

os.environ['YOLO_VERBOSE'] = 'False'

HEADER_FORMAT = 'BBBH'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_PAYLOAD_SIZE = 1400
MAX_CHUNKS = 10000
FRAME_TIMEOUT = 2.0 

# Đảm bảo đường dẫn này chính xác trên máy bạn
MODEL_PATH = r"/Users/ironm/Downloads/best.pt"

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception as e:
    HAS_YOLO = False
    logging.exception("Failed to import ultralytics: %s", e)

SERVER_IP = "0.0.0.0"
SERVER_PORT = 12345
CLIENT_IP = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class VideoServer(QWidget):
    def __init__(self):
        super().__init__()
        self.conf_thres = 0.5
        self.iou_thres = 0.5
        self.setWindowTitle("PyQt UDP Video Server - GPU Accelerated")
        self.setGeometry(100, 100, 900, 900)
        
        # --- UI Setup (Giữ nguyên layout của bạn) ---
        layout = QVBoxLayout()
        slider_layout = QHBoxLayout()
        
        conf_layout = QVBoxLayout()
        conf_label = QLabel("Confidence:")
        conf_label.setAlignment(Qt.AlignCenter)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1); self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(int(self.conf_thres*100))
        self.conf_slider.valueChanged.connect(self.update_conf)
        self.conf_value_label = QLabel(f"{self.conf_thres:.2f}")
        self.conf_value_label.setAlignment(Qt.AlignCenter)
        conf_layout.addWidget(conf_label); conf_layout.addWidget(self.conf_slider); conf_layout.addWidget(self.conf_value_label)
        
        iou_layout = QVBoxLayout()
        iou_label = QLabel("IoU:")
        iou_label.setAlignment(Qt.AlignCenter)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(1); self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(int(self.iou_thres*100))
        self.iou_slider.valueChanged.connect(self.update_iou)
        self.iou_value_label = QLabel(f"{self.iou_thres:.2f}")
        self.iou_value_label.setAlignment(Qt.AlignCenter)
        iou_layout.addWidget(iou_label); iou_layout.addWidget(self.iou_slider); iou_layout.addWidget(self.iou_value_label)
        
        slider_layout.addLayout(conf_layout); slider_layout.addSpacing(40); slider_layout.addLayout(iou_layout)
        layout.addLayout(slider_layout); layout.addSpacing(20)

        self.result_label = QLabel("Waiting for video stream...", self)
        self.result_label.setMinimumSize(800, 600)
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.info_label = QLabel("FPS: -- | Latency: -- ms", self)
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)
        self.setLayout(layout)

        # --- Network Setup ---
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((SERVER_IP, SERVER_PORT))
        self.sock.setblocking(False)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except: pass

        self.frame_buffer = {}; self.expected_chunks = {}; self.frame_timestamps = {}
        self.last_complete_frame = None
        self.lock = Lock()
        self.infer_input = None
        self.infer_lock = Lock()

        # --- Model Setup (GPU Optimized) ---
        self.model = None
        if HAS_YOLO:
            if torch.cuda.is_available():
                logging.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
                try:
                    # 1. Chuyển model sang GPU ngay khi load
                    self.model = YOLO(MODEL_PATH).to('cuda')
                    logging.info("Model loaded on GPU successfully")
                    
                    # Warmup trên GPU
                    dummy = torch.zeros((1, 3, 640, 640)).to('cuda')
                    _ = self.model(dummy, verbose=False)
                except Exception as e:
                    logging.error(f"Failed to load model on GPU: {e}")
            else:
                logging.warning("CUDA not available, falling back to CPU")
                self.model = YOLO(MODEL_PATH)

        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.recv_thread.start()

        if self.model is not None:
            self.infer_thread = threading.Thread(target=self.infer_loop, daemon=True)
            self.infer_thread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.receive_and_display)
        self.timer.start(10) # Tăng tốc độ kiểm tra hiển thị

        self.last_frame_time = time.time()
        self.frame_counter = 0; self.fps = 0

    def update_conf(self, value):
        self.conf_thres = value / 100.0
        self.conf_value_label.setText(f"{self.conf_thres:.2f}")

    def update_iou(self, value):
        self.iou_thres = value / 100.0
        self.iou_value_label.setText(f"{self.iou_thres:.2f}")

    def recv_loop(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(MAX_PAYLOAD_SIZE + HEADER_SIZE)
            except BlockingIOError:
                time.sleep(0.001); continue
            except: continue

            if len(data) < HEADER_SIZE: continue
            header = data[:HEADER_SIZE]
            payload = data[HEADER_SIZE:]
            
            try:
                frame_id, chunk_id, num_chunks, chunk_size = struct.unpack(HEADER_FORMAT, header)
                if frame_id not in self.frame_buffer:
                    self.frame_buffer[frame_id] = [None] * num_chunks
                    self.frame_timestamps[frame_id] = time.time()
                
                self.frame_buffer[frame_id][chunk_id] = payload

                if all(chunk is not None for chunk in self.frame_buffer[frame_id]):
                    frame_data = b''.join(self.frame_buffer[frame_id])
                    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.infer_lock:
                            self.infer_input = frame
                    self.frame_buffer.pop(frame_id); self.frame_timestamps.pop(frame_id)
            except: continue

            # Cleanup stale frames
            now = time.time()
            for fid in list(self.frame_timestamps.keys()):
                if now - self.frame_timestamps[fid] > FRAME_TIMEOUT:
                    self.frame_buffer.pop(fid); self.frame_timestamps.pop(fid)

    def infer_loop(self):
        """Hàm xử lý AI chính tối ưu cho GPU"""
        logging.info("Inference loop started on GPU")
        while True:
            with self.infer_lock:
                frame = self.infer_input
                self.infer_input = None
            
            if frame is None:
                time.sleep(0.001); continue

            t0 = time.time()
            try:
                # 2. Sử dụng device='0' (GPU) và tăng tốc độ bằng stream=True nếu cần
                # Ở đây mình bỏ việc chia khung hình % 15 vì GPU 1660Ti đủ mạnh để chạy liên tục
                results = self.model(frame, device='0', conf=self.conf_thres, iou=self.iou_thres, verbose=False)
                
                # Vẽ kết quả (vẫn thực hiện trên CPU để hiển thị PyQt)
                annotated = results[0].plot() 
                
                with self.lock:
                    self.last_complete_frame = annotated
                
                infer_time = (time.time() - t0) * 1000
                if len(results[0].boxes) > 0:
                    logging.info(f"GPU Detect: {len(results[0].boxes)} objects - {infer_time:.1f}ms")

            except Exception as e:
                logging.error(f"Inference error: {e}")

    def receive_and_display(self):
        with self.lock:
            frame = self.last_complete_frame
            self.last_complete_frame = None
        
        if frame is not None:
            now = time.time()
            self.frame_counter += 1
            if not hasattr(self, 'fps_last_time'): self.fps_last_time = now
            if self.frame_counter >= 10:
                elapsed = now - self.fps_last_time
                self.fps = int(10 / elapsed) if elapsed > 0 else 0
                self.fps_last_time = now; self.frame_counter = 0
            
            self.info_label.setText(f"FPS: {self.fps} | Img: {frame.shape[1]}x{frame.shape[0]}")
            self.display_frame(frame)

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.result_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.result_label.size(), Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    server = VideoServer()
    server.show()
    sys.exit(app.exec_())
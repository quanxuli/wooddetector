# Wood Defect Detector 🪵🔍

A computer vision project for real-time detection of defects on wood surfaces. The project includes dataset preprocessing/augmentation notebooks, a YOLO-based training workflow, and a multithreaded client-server application for streaming video sources over UDP and running real-time GPU-accelerated inference.

---

## 🔗 Dataset Link
The dataset used in this project is available at:
👉 **[Download Wood Detection Dataset (Google Drive)](https://drive.google.com/file/d/1qn6Hy3lERitjyNd5ZYOk7j8QahYR2i1B/)**

---

## 🚀 Key Features

* **4 Wood Defect Classes**: Real-time detection of **Crack**, **Dead Knot**, **Live Knot**, and **Marrow**.
* **Robust Preprocessing & Data Augmentation**: Complete data prep pipeline with Albumentations (pixel-level and spatial transforms) that auto-recalculates bounding boxes.
* **YOLO-based Training Pipeline**: Ready-to-use notebook for Colab using `ultralytics` with advanced hyperparameter configurations (AdamW, cosine learning rate scheduler, mosaic, mixup, and copy-paste).
* **Multi-Source Streaming Client**: Desktop application supporting webcam capture, local image file streaming, and desktop screen capture.
* **Real-time UDP Chunked Streaming**: Low-latency, custom chunked UDP frame transmission with server auto-discovery (via broadcast signals).
* **GPU-Accelerated Inference Server**: Multi-threaded GUI server that receives UDP streams, reconstructs frames, runs GPU-accelerated YOLO inference (CUDA), and provides interactive sliders to tune Confidence and IoU thresholds on-the-fly.

---

## 📁 Project Structure

```bash
wooddetector/
├── Preprocessing.ipynb   # Colab notebook for data extraction, splitting, health checks, and augmentation
├── train.ipynb           # Colab notebook for training the YOLO model
├── clientAv2.py          # PyQt5 client application for multi-source video streaming over UDP
├── serverBv2.py          # PyQt5 server application for receiving frames and running YOLO inference
├── README.md             # Project documentation (this file)
└── .gitignore            # Git ignore rules
```

---

## 🛠️ Setup & Installation

### Prerequisite Libraries
To run the client and server desktop applications, install the required python packages:

```bash
pip install PyQt5 opencv-python numpy pillow ultralytics torch
```

> [!NOTE]
> To enable GPU acceleration on the server, ensure you have a CUDA-compatible NVIDIA GPU and have installed a PyTorch build configured with CUDA support.

---

## 💻 How to Use

### 1. Preprocessing & Data Augmentation (`Preprocessing.ipynb`)
Open this notebook in Google Colab to:
* Mount Google Drive and import raw datasets (`wood.zip`).
* Automatically split the data into **Train (80%)**, **Valid (10%)**, and **Test (10%)**.
* Generate a dataset health report checking for empty labels or missing files.
* Perform image augmentations:
  * **Pixel-level**: Brightness, Contrast, CLAHE, Hue, Saturation, Value, and Gaussian Blur.
  * **Spatial-level**: Horizontal Flip, Vertical Flip, Shift/Scale/Rotate, and Random Resized Crop.
* Package the augmented dataset into a zip file (`Dataset_Augment.zip`) and export/backup.

### 2. Model Training (`train.ipynb`)
Open this notebook in Google Colab to:
* Load the augmented dataset zip file.
* Install `ultralytics` and train a YOLO model (e.g. `yolo26m.pt`) with custom parameters:
  * **Resolution**: 1024x1024
  * **Optimizer**: AdamW
  * **Epochs**: 200
  * **High Augmentations**: Mosaic (1.0), Mixup (0.3), Copy Paste (0.3)
* Train results and weights are saved automatically to your Google Drive runs folder.

### 3. Running the Client (`clientAv2.py`)
Launch the client application:
```bash
python clientAv2.py
```
* **Auto Discover**: Click this to broadcast a UDP signal `WHERE_IS_VIDEO_SERVER` on port `12346` to automatically find the running server's IP.
* **Select Source**:
  * `Use Webcam`: Stream from local web camera.
  * `Upload Image`: Stream a single image repeatedly.
  * `Capture Screen`: Stream desktop screenshot captured live.
* **Start Streaming**: Click `Start Streaming UDP` to begin transmitting to the target server IP.

### 4. Running the Server (`serverBv2.py`)
Before running, edit [serverBv2.py](file:///c:/Users/ironm/OneDrive/Desktop/wooddetector/serverBv2.py) and update the `MODEL_PATH` variable (line 26) with the path to your trained `best.pt` file.
```python
MODEL_PATH = r"path/to/your/best.pt"
```
Then launch the server application:
```bash
python serverBv2.py
```
* The server binds to port `12345` to receive incoming video frames.
* It hosts a background thread for non-blocking packet reception and frame reconstruction.
* A separate inference thread evaluates the frame using CUDA (if available) and updates the GUI frame.
* Drag the **Confidence** and **IoU** sliders to tune object detection thresholds dynamically.

---

## 📡 UDP Protocol Specifications

Standard UDP packets have a size limit. To stream high-quality JPEG frames, the client splits frames into smaller chunks and sends them with custom headers:

### Header Structure (`BBBH`)
| Field | Type | Size | Description |
|---|---|---|---|
| **Frame ID** | Byte (`B`) | 1 byte | Sequential identifier modulo 256 |
| **Chunk ID** | Byte (`B`) | 1 byte | Index of the current chunk |
| **Total Chunks** | Byte (`B`) | 1 byte | Total number of chunks in the frame |
| **Payload Size** | UShort (`H`) | 2 bytes | Size of the data payload in the packet |

* **Max Payload Size**: `1400` bytes (to fit within standard MTU limits and avoid IP fragmentation).
* **Timeout**: The server discards incomplete frames if all chunks are not received within `2.0` seconds.

---

## 🏷️ Classes & Labels
The model is trained to recognize the following classes:

| Class ID | Class Name | Description |
|:---:|---|---|
| **0** | `Crack` | Splits or fissures along the wood grain |
| **1** | `Dead_Knot` | Knots that are loose or no longer structurally integrated with the wood |
| **2** | `Live_Knot` | Sound knots structurally integrated and tight within the wood |
| **3** | `Marrow` | Pith or soft core sections of the wood |

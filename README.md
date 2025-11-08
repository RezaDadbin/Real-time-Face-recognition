# 🧠 Face Recognition GUI

## 📌 What is this project?

**Face Recognition GUI** is a desktop application written in **Python** that performs **real-time face recognition** using your computer’s webcam. It allows you to **capture and label** faces (assign names to them) and later **recognize those faces** live in front of the camera — all through a simple, graphical interface built with **Tkinter**.

This project runs **completely offline**, without using any external cloud APIs or internet access. It’s designed to be a **local, privacy-safe, and educational** implementation of how real face recognition systems work internally.

The system uses two deep learning models from the `facenet-pytorch` library:  
- **MTCNN** for detecting faces in each video frame.  
- **InceptionResnetV1 (FaceNet)** for converting faces into 512-dimensional numerical embeddings.

---

## ⚙️ How It Works

The pipeline has three main stages: **Face Detection → Face Embedding → Face Recognition**.

### 1) Face Detection (MTCNN)
Each webcam frame is processed with **MTCNN**, which returns bounding boxes around one or more faces in the image.

### 2) Face Embedding (FaceNet / InceptionResnetV1)
Each detected face is cropped and resized to **160×160**, then fed into **FaceNet**, producing a **512-dimensional embedding** (a numeric vector that represents the face). Two images of the same person produce similar embeddings; different people produce very different ones.

### 3) Local Database and Recognition
When you label faces, each image is saved locally under:
```
face_data/<person_name>/
```
The app computes embeddings for all stored images and keeps them in memory. During recognition, a new face’s embedding is compared to all stored embeddings using **Euclidean distance**. The smallest distance is taken as the match; if it’s smaller than a configurable threshold (default **1.0**), the system labels that face as known; otherwise it shows **“Unknown.”**

---

## 💻 How to Use It

### Install & Run
```bash
git clone https://github.com/RezaDadbin/face-recognition-gui.git
cd face-recognition-gui
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# or
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
python -m src.face_rec_gui.app
```

### Label Faces
1. In the main window, click **“Labeling UI.”**  
2. Enter a person’s name.  
3. The webcam opens:  
   - Press **`s`** to save the current frame to `face_data/<name>/`.  
   - Press **`q`** to quit labeling mode.  
4. After quitting, embeddings are computed for all saved images.

### Recognize Faces
1. From the main window, click **“Prediction UI.”**  
2. The webcam starts and detects faces live.  
3. Each face is embedded and compared to known ones.  
4. If the smallest distance is below the threshold, the name appears on the bounding box; otherwise it shows **Unknown**.  
5. Press **`q`** to exit.

### Configure (configs/default.yaml)
```yaml
device: "cpu"                # or "cuda" for GPU
database_path: "face_data"
recognition_threshold: 1.0   # smaller = stricter; larger = more lenient
camera_index: 0              # change if you have multiple webcams
```

---

## 🧱 Technical Overview

| Component | Purpose |
|----------|---------|
| Tkinter | Graphical user interface |
| OpenCV | Webcam frames & drawing |
| MTCNN | Face detection |
| FaceNet (InceptionResnetV1) | 512-D face embeddings |
| NumPy | Vector math & distances |
| PyTorch | Deep model execution |
| YAML | Config management |

**Recognition decision**: for a new embedding `e_new`, compute distances `d_i = ||e_new - e_stored,i||_2`. Let `d_min = min_i d_i`. If `d_min < threshold`, assign the corresponding label; else **Unknown**.

---

## 🔒 Data Privacy

All data stays **local** in `face_data/`. No external APIs or internet calls are used.

---

## 👥 Authors

- **Reza Dadbin** — GitHub: https://github.com/RezaDadbin
- **Sina Lotfi** — GitHub: https://github.com/cinaLotfi

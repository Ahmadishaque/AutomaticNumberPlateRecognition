# 🚘 Automatic Number Plate Recognition (ANPR) using Object Detection

## 📘 Overview
This project focuses on **Automatic Number Plate Recognition (ANPR)** through **object detection** using deep learning techniques. The task involves detecting vehicle registration plates from images and videos. The implementation uses **Faster R-CNN with ResNet-50 FPN** as the backbone model, trained and evaluated on a curated dataset derived from the **Open Images Dataset**.

The project aims to develop an end-to-end detection system capable of:
- Detecting number plates from vehicle images.
- Evaluating detection quality using **COCO metrics**.
- Performing **inference on both static images and videos**.

---

## 🎯 Objectives
- Build a **custom object detection dataset** with bounding boxes for license plates.
- Train a **Faster R-CNN** model to localize number plates accurately.
- Evaluate the model using **COCO detection metrics**.
- Run **real-time inference** on test images and videos.

---

## 🧠 Research Motivation
License plate detection is a foundational step in **Intelligent Transportation Systems (ITS)**, **traffic surveillance**, and **law enforcement automation**. This research-oriented implementation explores:
- Efficient **transfer learning** for specialized detection tasks.
- **Generalization of deep detection models** on small, single-class datasets.
- Benchmarking **COCO-based evaluation metrics** for fine-grained detection performance.

---

## 🧩 Dataset Description
- **Dataset Source:** Open Images Dataset (subset focusing on vehicle registration plates)
- **Training Images:** 5,308  
- **Validation Images:** 386  
- **Class:** `Vehicle registration plate`

**Directory Structure:**
```

Dataset/
├── train/
│   ├── Vehicle registration plate/
│   └── Label/
└── validation/
├── Vehicle registration plate/
└── Label/

```
Each image has an associated `.txt` label file containing bounding box coordinates in the format:
```

Vehicle registration plate xmin ymin xmax ymax

```

---

## 🧰 Tech Stack
| Category | Tools & Frameworks |
|-----------|--------------------|
| Language | Python 3.12 |
| Deep Learning | PyTorch, TorchVision |
| Model | Faster R-CNN (ResNet-50 FPN) |
| Metrics | COCO Evaluation (pycocotools) |
| Visualization | OpenCV, Matplotlib, PIL |
| Logging | TensorBoard.dev |

---

## 🧪 Methodology

### 1. Data Preparation & Visualization
- Parsed `.txt` annotations and visualized **ground truth bounding boxes** for verification.
- Ensured consistency between image and label files.
- Used OpenCV and Matplotlib for rendering bounding boxes on images.

### 2. Model Architecture
- Base Model: **Faster R-CNN** with **ResNet-50 FPN** backbone pre-trained on COCO.
- Modified classifier head to predict **2 classes** (`background`, `number plate`).

### 3. Training Configuration
- **Optimizer:** SGD (LR=0.005, Momentum=0.9, Weight Decay=0.0005)  
- **Scheduler:** StepLR (step_size=3, gamma=0.1)  
- **Epochs:** 10  
- **Batch Size:** 2  
- **Loss Function:** Built-in Faster R-CNN loss (classification + regression)
- **Device:** CUDA-enabled GPU

**Training Results:**
| Epoch | Average Loss |
|-------|---------------|
| 1 | 0.1341 |
| 5 | 0.0972 |
| 10 | 0.0916 |

> Loss consistently decreased across epochs, indicating effective convergence.

### 4. Evaluation Metrics (COCO)
- **Average Precision (AP):** 0.630 (IoU=0.50:0.95)
- **AP@0.50:** 0.884
- **AP@0.75:** 0.746
- **Average Recall (AR):** 0.687 (IoU=0.50:0.95)

The model achieved strong performance with **AP > 0.5**, exceeding the benchmark expectation.

### 5. Inference & Visualization
- Performed **bounding box prediction** on validation images.
- Displayed results with **confidence scores** above a threshold (0.5).
- Detected plates were highlighted with labeled boxes and score overlays.

### 6. Video Inference
- Integrated frame-wise inference using **OpenCV VideoCapture**.
- Processed video in real-time and rendered detection overlays.
- Saved output as `output_video.mp4`.
- Uploaded results to YouTube for demonstration.

📺 **Demo Video:** [https://youtu.be/Ip4pCRA3S94](https://youtu.be/Ip4pCRA3S94)

---

## 📊 Results Summary
| Metric | Value |
|---------|--------|
| Average Precision (AP) @[IoU=0.50:0.95] | 0.630 |
| Average Precision (AP) @[IoU=0.50] | 0.884 |
| Average Precision (AP) @[IoU=0.75] | 0.746 |
| Average Recall (AR) @[IoU=0.50:0.95] | 0.687 |
| Inference Speed (per image) | ~0.08s on GPU |

---

## 💾 Key Learnings
- Transfer learning with **Faster R-CNN** effectively handles single-class detection.
- **TensorBoard monitoring** is essential for early detection of overfitting.
- Proper **data labeling consistency** dramatically improves AP.
- Fine-tuning pretrained models on small datasets yields **high generalization**.

---

## 🧩 Future Work
- Extend detection to **multi-class scenarios** (vehicles, pedestrians, plates).
- Incorporate **OCR models (e.g., CRNN or EasyOCR)** for text extraction.
- Experiment with **YOLOv8 and DETR** for speed-performance trade-offs.
- Explore **real-time deployment** on edge devices like NVIDIA Jetson.

---

## 👨‍💻 Author

**Ahmad Ishaque Karimi**  
Graduate Student — Data Science & Computer Vision Research  
📧 ahmadishaquekarimi@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/ahmadishaquekarimi/)

---

> “Detection is not about finding objects; it’s about understanding scenes.”

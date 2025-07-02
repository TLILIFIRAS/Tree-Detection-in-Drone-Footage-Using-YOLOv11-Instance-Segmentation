# Tree Detection in Drone Footage Using YOLOv11 Instance Segmentation

![Project Demo](output.gif)  

---

## 🚀 Project Overview

This project demonstrates a **tree detection system** applied to drone footage, leveraging the latest **YOLOv11 instance segmentation** model. Despite training the model on **only two images**, it achieves reliable detection results by combining:

- Instance segmentation masks for precise tree outlines  
- Non-Maximum Suppression (NMS) to remove duplicate bounding boxes  
- Real-time video processing with mask overlay and performance metrics  

This lightweight and efficient solution can be used for environmental monitoring, forest management, agricultural analysis, and sustainability projects using drone technology.

---

## ⚙️ Features

- **YOLOv11 Instance Segmentation:** Accurate per-tree mask and bounding box detection  
- **Data Efficiency:** Model trained with just two annotated images  
- **Duplicate Box Removal:** Integrated NMS filtering to avoid overlapping detections  
- **Drone Video Input:** Processes local drone footage video files  
- **Real-time Visualization:** Displays masks, bounding boxes, FPS, and frame counts live  
- **Easy Output:** Saves processed video with all annotations  

---

## 📋 Requirements

- Python 3.8+  
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (`pip install ultralytics`)  
- OpenCV (`pip install opencv-python`)  
- NumPy (`pip install numpy`)  
- pandas (`pip install pandas`)  
- PyTorch (compatible with Ultralytics YOLOv11)  

---

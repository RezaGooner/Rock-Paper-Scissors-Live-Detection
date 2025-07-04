# Rock-Paper-Scissors Live Detection

![4](https://github.com/user-attachments/assets/778c2042-a18c-4e8a-bab9-4970d580b40e)

A real-time Rock-Paper-Scissors gesture detection system using deep learning and computer vision (YOLOv8) in Python.

## 🚀 Overview

This project implements a live hand gesture recognition model that can classify the classic Rock-Paper-Scissors game using a webcam. It leverages the YOLOv8 object detection framework for robust, real-time detection.

**Main Use Case:**  
Play Rock-Paper-Scissors against your computer using only your hand gestures in front of a camera!

---

## ✨ Features

- Real-time detection and classification: **Rock**, **Paper**, **Scissors**
- Uses YOLOv8 deep learning models for accuracy and speed
- Webcam integration for live predictions
- Simple code base: easy to extend or retrain
- Clear visualization with bounding boxes and labels

---

## 🛠️ Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy
- (Optional) CUDA-compatible GPU for faster inference

**Install dependencies:**
```bash
pip install ultralytics opencv-python numpy
```

---

## 🗂️ Project Structure

```
root/
│
├── live_detection.py       # Main script to run live detection
├── train model.ipynb       # Notebook to (re-)train model (if needed)
├── best.pt                 # Pretrained YOLO weights (.pt)

```

## 🖥️ Usage
1. Clone the repository
```
git clone https://github.com/RezaGooner/Rock-Paper-Scissors-Live-Detection
cd Rock-Paper-Scissors-Live-Detection
```
2. Download Weights
If not present, download pretrained YOLOv8 weights and place them in the weights/ directory.

You can use your own trained weights or try public YOLO models as a baseline.

3. Run Live Detection
```
python live_detection.py --weights best.pt
```
This will start your webcam and show real-time hand gesture detection.

Command-line arguments:
```
--weights: Path to the YOLOv8 weights file (.pt)
--source: (Optional) Select video/image file or use webcam (default)
```


## 🏗️ Training (Optional)
If you want to train or fine-tune the model on your own labeled data:

```
Run notebook -> train model.ipynb
```


## 📊 Results & Demo
![1](https://github.com/user-attachments/assets/3870cc0c-12ec-4199-940e-c12158a9598c)
![2](https://github.com/user-attachments/assets/a6894520-46c5-4f58-bed9-d1e12a29c8f7)
![3](https://github.com/user-attachments/assets/9d5e99bd-5dc7-426b-a4af-268ca03457b2)


## 🙏 Credits
```
YOLOv8 by Ultralytics: https://github.com/ultralytics/ultralytics
Dataset: https://universe.roboflow.com/objectdetection-ol9fv/rockpaperscissor-frthr
Author: RezaGooner
```

## 📄 License
MIT License

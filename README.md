# Object Detection using YOLOv8

This project demonstrates real-time **object detection** using the **YOLOv8 model**.  
It can detect and label objects in:
- Images
- Videos
- Live webcam stream

---

## 🚀 Features
- Detects multiple objects with bounding boxes and confidence scores.
- Works on images, video files, and live webcam feed.
- Based on **YOLOv8 (Ultralytics)** pretrained model.
- Easy to extend for custom datasets.

---

## 📌 Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/vishnu-R2005/object-detection.git
   cd object-detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install ultralytics opencv-python
   ```

---

## ▶️ Usage

### Detect objects in an **image**
```bash
python detect_image.py --image your_image.jpg
```

### Detect objects in a **video**
```bash
python detect_image.py --video your_video.mp4
```

### Detect objects using **webcam**
```bash
python detect_webcam.py --video 0
```

---

## 📸 Sample Outputs

### Image Detection Example
![Object Detection Example](asset\detected_image.png)

---

## 🎥 Demo Videos

### Video Detection Example   
![Watch the video](asset\object_detected_video.mp4)

### Detection using webcam Example 
![Watch the video](asset\object_detection_through_webcam.mp4)

---

## 📊 Model Accuracy
- Using **YOLOv8n (Nano)**: ~37.3 mAP (fast, lightweight).
- Using **YOLOv8s (Small)**: ~44.9 mAP (more accurate).
- Can be swapped depending on speed vs accuracy needs.

---

## ✨ Future Work
- Add custom dataset training.
- Deploy with Flask/Django as a web app.
- Optimize inference speed with GPU support.

---

## 👨‍💻 Author
Developed by Vishnu

# DeepShield-video
An advanced deepfake detection system that analyzes videos using a hybrid **CNN + Transformer architecture**.

The system extracts facial frames, learns spatial features using CNN, and captures temporal inconsistencies using a Transformer encoder to accurately classify videos as Real or Fake.

---

## 🧠 Key Innovations

### ✅ CNN + Transformer Temporal Modeling

* CNN (ResNet18) extracts frame-level spatial features
* Transformer captures long-range temporal dependencies across frames

### ✅ Multi-Face Handling

* Detects multiple faces per frame
* Selects primary face (largest area) for consistent tracking

### ✅ Temporal Consistency Analysis

* Sliding window sequence modeling
* Frame-wise prediction smoothing

### ✅ Audio + Video Fusion *(optional)*

* Supports audio energy-based feature fusion
* Works even if audio is absent

### ✅ Real-Time Deepfake Detection

* Live webcam inference with bounding boxes and predictions

---

## 🏗️ Project Structure

```
DeepShield-video/
│
├── models/
│   ├── cnn_model.py
│   ├── cnn_transformer_model.py
│   ├── cnn_lstm_model.py (legacy)
│
├── training/
│   └── train_lstm.py
│
├── inference/
│   ├── predict_video.py
│   └── realtime_demo.py
│
├── preprocessing/
│   ├── face_detection.py
│   └── extract_frames.py
|
├── utils/
│   ├── video_dataset.py
│   └── aggregation.py
|
├── datasets/videos/
│              ├── real/
│              ├── fake/
│
├── outputs/
├── models/
└── .gitignore
```

---

## ⚙️ Installation

```
git clone https://github.com/bhargavi-chauhan/DeepShield-video
cd DeepShield-video

pip install -r requirements.txt
```

## 🏋️ Training

```
python -m training.train_lstm --data_dir datasets/videos
```

### 🔧 Features

* Automatic train/validation split
* Mixed Precision Training (AMP)
* Early Stopping
* Best Model Saving
* Metrics per epoch:

  * Accuracy
  * Precision
  * Recall
  * F1-score

---

## 📊 Evaluation Outputs

Generated in `outputs/`:

* 📉 Loss Curve
* 📈 Accuracy Curve
* 📊 Confusion Matrix (best model)
* 📉 ROC Curve + AUC

---

## 🎬 Inference (Video)

```bash
python -m inference.predict_video --video <path/to/video>.mp4
```

### Output:

* Video Score
* Temporal Variation
* Final Prediction (REAL / FAKE)
* Timeline graph

---

## 🎥 Real-Time Demo

```bash
python -m inference.realtime_demo
```
Press Q to exit.

### Features:

* Multi-face detection
* Per-face prediction
* Smooth confidence tracking
* Real-time bounding boxes



---


# Face Expression Detector using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
---


## Overview
This project implements an end-to-end deep learning pipeline for facial expression recognition. It combines computer vision techniques with a custom-trained CNN model to classify human emotions from facial images in real-time.

### Project Highlights:
- **High Accuracy:** Achieves 85% classification accuracy on FER2013 dataset
- **Real-Time Performance:** Live webcam emotion detection using OpenCV
- **Optimized Inference:** 92% reduction in inference time through TensorFlow Lite quantization (from ~68ms to ~5ms per frame on CPU)
- **7 Emotion Categories:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

---

## Features

- Real-time emotion detection from webcam feed
- CNN-based classification (trained from scratch on FER2013)
- Face detection using Haar Cascades (OpenCV)
- TensorFlow Lite optimization for faster inference
- Comprehensive performance benchmarking
- Data preprocessing and normalization pipeline

---

## Tech Stack

**Core Technologies:**
- Python 3.8+
- TensorFlow / Keras 2.x
- OpenCV
- NumPy, Pandas
- Scikit-learn

**Model Optimization:**
- TensorFlow Lite
- Post-training quantization

---

## Model Architecture

The CNN model consists of:

```
Input (48x48x1 grayscale images)
├── Conv2D (128 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Dropout (0.4)
├── Conv2D (256 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Dropout (0.4)
├── Conv2D (512 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Dropout (0.4)
├── Conv2D (512 filters, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Dropout (0.4)
├── Flatten
├── Dense (512) + ReLU
├── Dropout (0.4)
├── Dense (256) + ReLU
├── Dropout (0.3)
└── Dense (7) + Softmax
```

**Total Parameters:** ~3-5 million  
**Training Accuracy:** 85% on FER2013 dataset

---

## Performance Benchmarks

### Inference Time Optimization

| Model Type | Inference Time (per frame) | Improvement |
|------------|----------------------------|-------------|
| Original Keras Model | 68.39 ms | Baseline |
| TensorFlow Lite (Quantized) | 5.29 ms | **92.3% faster** |

**Hardware:** CPU-only inference (Intel Core series)  
**Methodology:** Averaged over 100 inference runs

### Model Size Comparison

| Model Format | File Size |
|--------------|-----------|
| Keras (.keras) | ~50-80 MB |
| TFLite (quantized) | ~12-20 MB |

---

##  Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time detection)
- FER2013 dataset (download from Kaggle)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/okayysatyam/Face-Expression-Detector.git
cd Face-Expression-Detector
```

2. **Create virtual environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download FER2013 Dataset:**
   - Visit [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
   - Download and extract to `data/fer2013/` folder

---

##  Usage

### 1. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the FER2013 dataset
- Train the CNN model (50 epochs)
- Save the trained model as `emotiondetector.keras`

**Note:** Training may take 2-4 hours on CPU, 30-60 minutes on GPU.

### 2. Convert to TensorFlow Lite

```bash
python convert_to_tflite.py
```

This applies post-training quantization and generates `emotion_detector_quantized.tflite`.

### 3. Run Performance Benchmark

```bash
python benchmark_inference.py
```

This compares inference times between Keras and TFLite models.

### 4. Real-Time Emotion Detection

```bash
python realtimedetection.py
```

Press `ESC` to exit the webcam window.

---

## Project Structure

```
Face-Expression-Detector/
├── config/
│   └── Expression-Detector.json
├── data/                          # Dataset (download separately)
│   └── fer2013/
│       ├── train/
│       └── test/
├── models/                        # Trained models (generated after training)
│   ├── emotiondetector.keras
│   └── emotion_detector_quantized.tflite
├── src/
│   ├── train_model.py
│   ├── realtimedetection.py 
│   ├── convert_to_tflite.py
│   └── benchmark_inference.py 
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Dataset

This project uses the **FER2013 (Facial Expression Recognition 2013)** dataset:

- **Total Images:** 35,887 grayscale images
- **Image Size:** 48x48 pixels
- **Classes:** 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- **Split:** 28,709 training, 7,178 test images

**Citation:**
```
Challenges in Representation Learning: Facial Expression Recognition Challenge
Kaggle Competition, 2013
```

**Download:** [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

---

## Results & Analysis

### Classification Accuracy
- **Training Accuracy:** ~85% on FER2013 dataset
- **Validation Accuracy:** ~82-84%

### Inference Performance
- **Original Model:** 68ms per frame (14.6 FPS)
- **Optimized Model:** 5ms per frame (200 FPS)
- **Speedup:** 13x faster inference

### Key Insights
- TensorFlow Lite quantization provides significant speedup with minimal accuracy loss
- Real-time performance achieved on CPU-only systems
- Suitable for deployment on edge devices and mobile platforms

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [FER2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- [TensorFlow/Keras Documentation](https://keras.io/)
- [OpenCV Library](https://opencv.org/)
- [TensorFlow Lite Optimization Guide](https://www.tensorflow.org/lite/performance/post_training_quantization)

---

##  Author

**Satyam Kumar Pandey**

- GitHub: [@okayysatyam](https://github.com/okayysatyam)
- LinkedIn: [Satyam](https://www.linkedin.com/in/satyam-kumar-pandey/) 

---

# VeraFace – AI-Based Fake Face Detection System

VeraFace is an AI-powered face authenticity detection platform designed to verify whether facial images are authentic human photographs or synthetically generated / manipulated (deepfakes).

## Overview

The platform provides a web-based verification interface and an API backend powered by deep learning models (EfficientNet-B0) to classify uploaded facial images into two primary categories:
- **REAL**: Authentic human photograph.
- **FAKE**: Synthetically generated or manipulated face.

## Project Structure

```
veraface/
├── backend/
│   ├── api/
│   │   └── predict.py          # Prediction endpoint and image pipeline
│   ├── static/
│   │   ├── css/style.css       # Web styling
│   │   └── js/script.js        # Client-side UI interactions
│   ├── templates/
│   │   └── index.html          # Web interface
│   ├── weights/                # Model checkpoints and class mappings
│   ├── app.py                  # Flask application entry point
│   └── uploads/                # Temporary image upload directory
├── frontend/                   # Standalone frontend interface
│   ├── css/
│   ├── js/
│   └── index.html
├── training/                   # Model training and evaluation scripts
│   ├── train_efficientnet.py   # Fine-tuning EfficientNet-B0
│   ├── evaluate_efficientnet.py# Validation & test evaluation
│   └── train_random_forest.py  # Feature-based baseline
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch & Torchvision
- Flask & Flask-CORS
- OpenCV (`opencv-python`)
- Pillow, NumPy, Scikit-Learn

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/poojithajallu-debug/veraface.git
   cd veraface
   ```

2. **Set up a virtual environment:**
   ```bash
   cd backend
   python -m venv venv
   .\venv\Scripts\activate   # Windows
   # or source venv/bin/activate # Linux/macOS
   ```

3. **Install dependencies:**
   ```bash
   pip install flask flask-cors torch torchvision opencv-python pillow numpy scikit-learn
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000/`.

## API Usage

- **Endpoint**: `POST /api/predict`
- **Payload**: `multipart/form-data` with field `image`
- **Response**:
  ```json
  {
    "success": true,
    "prediction": "REAL",
    "confidence": 91.9,
    "real_probability": 91.9,
    "fake_probability": 8.1,
    "face_detected": true,
    "face_count": 1,
    "message": "Image analysis completed."
  }
  ```

## License
MIT License

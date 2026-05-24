# Accident Detection App

## Overview
This app detects traffic accidents from CCTV or dashcam images using a trained TensorFlow model.

It is built with Streamlit for a clean, evaluation-ready dashboard experience.

## Included Files
- `app.py` — Streamlit application for accident detection
- `requirements.txt` — Python dependencies for the app
- `best_accident_detection_model.keras` — Trained accident detection model
- `README.md` — This file

## Installation
1. Create and activate a Python virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App
Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Usage
1. Upload a CCTV or traffic camera image in JPEG/PNG format.
2. Wait for the model to evaluate the image.
3. View the accident detection result, confidence score, and recommendation.

## Model Details
- Model file: `best_accident_detection_model.keras`
- Input: 224×224 RGB image
- Output: Accident / No Accident classification plus confidence
- Preprocessing: RGB conversion, resizing, normalization

## Notes
- Use well-lit images with clear vehicle and road visibility.
- Avoid blurry or dark frames for best results.

## Author
Built with ❤️ by **Aatequa Ansari**

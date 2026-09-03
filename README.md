# Real-Time Facial Emotion Detection

An end-to-end facial emotion classification project built with Python, OpenCV, and TensorFlow/Keras.

It detects faces from a webcam or video, converts each face to a normalized 48×48 grayscale image, and overlays the predicted expression and confidence in real time.

## Emotions

`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

## Quick start

```powershell
cd outputs/emotion_detection_system
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run real-time inference

Place a trained Keras model at `models/emotion_model.keras`, then run:

```powershell
python app.py
```

Press `q` or `Esc` to close the camera window. Use a video file instead of the camera with:

```powershell
python app.py --source path\to\video.mp4
```

### Train the model

Organize images by emotion class (the same class names as above):

```text
data/
  train/
    happy/
    sad/
    ...
  validation/
    happy/
    sad/
    ...
```

Then run:

```powershell
python train.py --data-dir data --epochs 30
```

The best model is written to `models/emotion_model.keras` and can be used immediately by `app.py`.

## Notes

- The included Haar cascade is distributed with OpenCV; no separate detector download is needed.
- Predictions are only as reliable as the training data and should not be used for high-stakes decisions.

"""Live facial emotion recognition using OpenCV and a Keras model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
IMAGE_SIZE = (48, 48)


def preprocess_face(face: np.ndarray) -> np.ndarray:
    """Convert a cropped BGR face to the model's normalized batch format."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized[np.newaxis, ..., np.newaxis]


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-time facial emotion detection.")
    parser.add_argument("--model", default="models/emotion_model.keras", help="Path to a trained Keras model")
    parser.add_argument("--source", default="0", help="Camera index or video path (default: 0)")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Hide labels below this probability")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train one with `python train.py --data-dir data`."
        )
    model = tf.keras.models.load_model(model_path)
    detector_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(detector_path)
    if detector.empty():
        raise RuntimeError("OpenCV's Haar cascade could not be loaded.")

    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))

            for x, y, width, height in faces:
                face = frame[y : y + height, x : x + width]
                probabilities = model.predict(preprocess_face(face), verbose=0)[0]
                index = int(np.argmax(probabilities))
                confidence = float(probabilities[index])
                label = EMOTIONS[index] if confidence >= args.min_confidence else "uncertain"
                color = (55, 210, 100) if label != "uncertain" else (40, 190, 240)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.0%}", (x, max(y - 10, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            cv2.imshow("Emotion Detection — press q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

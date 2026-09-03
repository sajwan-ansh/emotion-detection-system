"""Train a compact CNN for facial emotion recognition from class-folder images."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
IMAGE_SIZE = (48, 48)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 1))
    x = inputs
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(len(EMOTIONS), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="emotion_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def make_dataset(path: Path, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        path, labels="inferred", label_mode="int", class_names=list(EMOTIONS), color_mode="grayscale",
        image_size=IMAGE_SIZE, batch_size=batch_size, shuffle=shuffle,
    )
    return dataset.map(lambda image, label: (tf.cast(image, tf.float32) / 255.0, label), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a facial emotion classifier.")
    parser.add_argument("--data-dir", required=True, help="Folder containing train/ and validation/ directories")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    root = Path(args.data_dir)
    train_dir, validation_dir = root / "train", root / "validation"
    if not train_dir.is_dir() or not validation_dir.is_dir():
        raise FileNotFoundError("Expected `train` and `validation` folders inside --data-dir.")

    train_data = make_dataset(train_dir, args.batch_size, shuffle=True)
    validation_data = make_dataset(validation_dir, args.batch_size, shuffle=False)
    output = Path("models/emotion_model.keras")
    output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output, monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=7, restore_best_weights=True),
    ]
    model = build_model()
    model.fit(train_data, validation_data=validation_data, epochs=args.epochs, callbacks=callbacks)
    print(f"Best model saved to {output}")


if __name__ == "__main__":
    main()

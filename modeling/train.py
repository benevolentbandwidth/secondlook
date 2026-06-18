# Training script for the Second Look baseline classifier.

# Typical usage:
#   from modeling.train import train_baseline
#   history = train_baseline(train_df, val_df, image_dir="data/images/")

# What this does:
#   1. Builds tf.data pipelines from split DataFrames (binary labels)
#   2. Computes positive-class-biased class weights
#   3. Trains with early stopping + LR reduction on val_loss
#   4. Saves the best checkpoint by val_loss

# After training, run evaluate.py to check WORTH_SECOND_LOOK sensitivity
# before considering the model usable.

import os
import numpy as np
import tensorflow as tf
import pandas as pd

from config.constants import INPUT_SIZE
from modeling.baseline_classifier import (
    build_baseline,
    compute_class_weights,
)
from data_pipeline.preprocessor import preprocess
from data_pipeline.quality import quality_check


def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: str,
    image_col: str = "image_path",
    label_col: str = "label",
    input_size: tuple = INPUT_SIZE,
    batch_size: int = 32,
    max_epochs: int = 50,
    checkpoint_dir: str = "checkpoints/baseline",
    freeze_backbone: bool = True,
) -> tf.keras.callbacks.History:
    """Train the baseline MobileNetV2 classifier with a binary head.

    Args:
        train_df: Training split DataFrame (from splitter.split_dataset).
        val_df: Validation split DataFrame.
        image_dir: Root directory containing image files.
        image_col: Column in DataFrames with image filenames or relative paths.
        label_col: Column with binary labels (int 0 or 1).
        input_size: (height, width) passed to build_baseline and the data pipeline.
        batch_size: Training batch size.
        max_epochs: Maximum training epochs (early stopping will halt sooner).
        checkpoint_dir: Directory to save the best model checkpoint.
        freeze_backbone: If True, only the head trains. Recommended for first run.

    Returns:
        Keras History object from model.fit().
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_ds = _build_dataset(train_df, image_dir, image_col, label_col, input_size, batch_size, shuffle=True)
    val_ds = _build_dataset(val_df, image_dir, image_col, label_col, input_size, batch_size, shuffle=False)

    model = build_baseline(input_size=input_size, freeze_backbone=freeze_backbone)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    class_weights = compute_class_weights(list(train_df[label_col]))

    callbacks = _build_callbacks(checkpoint_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print(f"\nBest model saved to: {checkpoint_dir}")
    print("Run evaluate.py on the test set before using this model.")
    return history


# ---------------------------------------------------------------------------
# Dataset pipeline
# ---------------------------------------------------------------------------

def _build_dataset(
    df: pd.DataFrame,
    image_dir: str,
    image_col: str,
    label_col: str,
    input_size: tuple,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    paths = [os.path.join(image_dir, p) for p in df[image_col]]
    labels = [int(y) for y in df[label_col]]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda path, label: _load_and_preprocess(path, label, input_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _load_and_preprocess(
    path: tf.Tensor,
    label: tf.Tensor,
    input_size: tuple,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Load an image from disk and run the preprocessing pipeline."""
    raw = tf.io.read_file(path)
    image = tf.image.decode_png(raw, channels=1)

    # Run numpy-side preprocessing (CLAHE, masking, orientation) via py_function.
    # This is acceptable for training; TF Lite inference uses the C++ pipeline.
    image = tf.py_function(
        func=lambda img: _numpy_preprocess(img.numpy(), input_size),
        inp=[image],
        Tout=tf.float32,
    )
    image.set_shape((*input_size, 1))
    # Binary head expects float32 labels.
    label = tf.cast(label, tf.float32)
    return image, label


def _numpy_preprocess(image_np: np.ndarray, input_size: tuple) -> np.ndarray:
    """Bridge from tf.py_function to the data_pipeline preprocessor."""
    passes, reason = quality_check(image_np)
    if not passes:
        # Return a zero image for bad-quality samples during training.
        # These will be filtered in production; during training they contribute
        # zero signal (not noise) and their presence can be audited via reason.
        return np.zeros((*input_size, 1), dtype=np.float32)
    return preprocess(image_np, target_size=input_size)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _build_callbacks(checkpoint_dir: str) -> list:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

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
from modeling.losses import build_loss, describe_loss
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
    checkpoint_dir: str = "gs://b2-foundation/second-look/checkpoints/baseline",
    freeze_backbone: bool = True,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    worth_weight_multiplier: float | None = None,
    loss: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    use_class_weights: bool = True,
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
        learning_rate: Adam learning rate. Keep ~1e-3 for a frozen head; drop to
                       ~1e-4/1e-5 when fine-tuning (freeze_backbone=False) so the
                       pretrained backbone weights are not destroyed.
        dropout_rate: Dropout before the classification head (regularization knob).
        worth_weight_multiplier: Extra positive-class weight (default 1.5 via
                       compute_class_weights). Pass 1.0 to disable the bias.
                       Ignored when use_class_weights=False.
        loss: "bce" (default, the validated baseline) or "focal". See
                       modeling.losses.build_loss.
        focal_gamma: Focal focusing parameter (ignored unless loss="focal").
        focal_alpha: Focal internal class balancing (ignored unless
                       loss="focal"). None leaves it off.
        use_class_weights: If False, train with NO Keras class_weight at all.
                       Use with focal_alpha to let focal own the imbalance --
                       stacking balanced class weights (~8x positive at the real
                       6%-positive distribution) ON TOP of focal alpha is how a
                       model gets pushed into low-precision over-flagging.

    Returns:
        Keras History object from model.fit().
    """
    # gfile.makedirs handles both local paths and gs:// URIs (os.makedirs would
    # create a junk local directory for a gs:// path).
    tf.io.gfile.makedirs(checkpoint_dir)

    train_ds = _build_dataset(train_df, image_dir, image_col, label_col, input_size, batch_size, shuffle=True)
    val_ds = _build_dataset(val_df, image_dir, image_col, label_col, input_size, batch_size, shuffle=False)

    model = build_baseline(
        input_size=input_size,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate,
    )
    print(f"[train] loss: {describe_loss(loss, focal_gamma, focal_alpha, use_class_weights)}")
    _compile(model, learning_rate, loss, focal_gamma, focal_alpha)

    class_weights = _resolve_class_weights(
        train_df[label_col], worth_weight_multiplier, use_class_weights
    )

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
# Two-phase fine-tuning
# ---------------------------------------------------------------------------

class _CombinedHistory:
    """Expose two phases' fit histories under one ``.history`` dict.

    Concatenates each metric across phases so callers can treat a two-phase run
    like a single Keras ``History`` (e.g. ``len(h.history['loss'])`` = total
    epochs run), while still exposing the per-phase objects as ``.phase1`` /
    ``.phase2``.
    """

    def __init__(self, phase1, phase2):
        keys = set(phase1.history) | set(phase2.history)
        self.history = {
            k: list(phase1.history.get(k, [])) + list(phase2.history.get(k, []))
            for k in keys
        }
        self.phase1 = phase1
        self.phase2 = phase2


def _find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Return the nested backbone (the only sub-Model inside the baseline)."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested backbone Model found in the baseline.")


def train_baseline_two_phase(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: str,
    image_col: str = "image_path",
    label_col: str = "label",
    input_size: tuple = INPUT_SIZE,
    batch_size: int = 32,
    phase1_epochs: int = 12,
    phase2_epochs: int = 12,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-5,
    dropout_rate: float = 0.3,
    checkpoint_dir: str = "gs://b2-foundation/second-look/checkpoints/baseline",
    worth_weight_multiplier: float | None = None,
    loss: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    use_class_weights: bool = True,
) -> "_CombinedHistory":
    """Two-phase fine-tuning: converge the head frozen, then unfreeze at low LR.

    Phase 1 trains only the classification head (MobileNetV2 frozen, BatchNorm in
    inference mode). Phase 2 unfreezes the backbone's conv weights at a LOW LR
    while KEEPING BatchNorm frozen (inference stats), the stable recipe for
    fine-tuning a small, imbalanced dataset. Training one-shot from an unfrozen
    backbone (the naive path) tends to collapse toward the positive class;
    warm-starting the head first, and not perturbing BN statistics, avoids that.

    ``best.keras`` under ``checkpoint_dir`` holds the best PHASE-2 model by
    val_auc. Returns a combined history spanning both phases.

    ``loss`` / ``focal_gamma`` / ``focal_alpha`` / ``use_class_weights`` are the
    imbalance knobs documented on ``train_baseline``; the SAME loss is used in
    both phases so phase 2 continues the objective phase 1 converged on.
    """
    tf.io.gfile.makedirs(checkpoint_dir)
    train_ds = _build_dataset(train_df, image_dir, image_col, label_col, input_size, batch_size, shuffle=True)
    val_ds = _build_dataset(val_df, image_dir, image_col, label_col, input_size, batch_size, shuffle=False)

    class_weights = _resolve_class_weights(
        train_df[label_col], worth_weight_multiplier, use_class_weights
    )
    print(f"[train] loss: {describe_loss(loss, focal_gamma, focal_alpha, use_class_weights)}")

    # Build frozen: build_baseline(freeze_backbone=True) calls the backbone with
    # training=False, baking BatchNorm into inference mode for BOTH phases.
    model = build_baseline(
        input_size=input_size, freeze_backbone=True, dropout_rate=dropout_rate
    )

    # --- Phase 1: head only (backbone frozen) ---
    print(f"\n[two-phase] PHASE 1 (head, frozen backbone): "
          f"epochs={phase1_epochs} lr={phase1_lr}")
    _compile(model, phase1_lr, loss, focal_gamma, focal_alpha)
    phase1_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
    ]
    h1 = model.fit(
        train_ds, validation_data=val_ds, epochs=phase1_epochs,
        class_weight=class_weights, callbacks=phase1_callbacks,
    )

    # --- Phase 2: unfreeze backbone conv weights, keep BatchNorm frozen ---
    # Order matters: backbone.trainable=True flips ALL sublayers trainable, so
    # re-freeze the BatchNorm layers AFTER, to hold their running stats fixed.
    backbone = _find_backbone(model)
    backbone.trainable = True
    frozen_bn = 0
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1
    print(f"\n[two-phase] PHASE 2 (fine-tune backbone): epochs={phase2_epochs} "
          f"lr={phase2_lr} (BatchNorm layers kept frozen: {frozen_bn})")
    # Recompile so the trainable-flag changes take effect, at the LOW phase-2 LR.
    _compile(model, phase2_lr, loss, focal_gamma, focal_alpha)
    phase2_callbacks = _build_callbacks(checkpoint_dir)  # best.keras by val_auc
    h2 = model.fit(
        train_ds, validation_data=val_ds, epochs=phase2_epochs,
        class_weight=class_weights, callbacks=phase2_callbacks,
    )

    print(f"\nBest fine-tuned model saved to: {checkpoint_dir}")
    print("Run evaluate.py on the test set before using this model.")
    return _CombinedHistory(h1, h2)


# ---------------------------------------------------------------------------
# Compile + class-weight helpers (shared by the single and two-phase paths)
# ---------------------------------------------------------------------------

def _compile(
    model: tf.keras.Model,
    learning_rate: float,
    loss: str,
    focal_gamma: float,
    focal_alpha: float | None,
) -> None:
    """Compile with the configured loss. One place, so the single-run and both
    two-phase compiles can never drift apart on metrics or loss resolution."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=build_loss(loss, focal_gamma=focal_gamma, focal_alpha=focal_alpha),
        # val_auc is what ModelCheckpoint/EarlyStopping monitor -- it must stay
        # in this list. AUC is also the one metric that is comparable ACROSS
        # losses, which matters when a sweep mixes bce and focal configs.
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )


def _resolve_class_weights(
    labels,
    worth_weight_multiplier: float | None,
    use_class_weights: bool,
) -> dict[int, float] | None:
    """Return the Keras class_weight dict, or None when weighting is disabled.

    compute_class_weights still validates the labels (ValueError on anything
    outside {0, 1}) even when the weights are discarded, so turning weighting
    off never turns the label check off with it.
    """
    weights = compute_class_weights(
        list(labels), positive_multiplier=worth_weight_multiplier
    )
    if not use_class_weights:
        print("[train] class_weight DISABLED (imbalance handled by the loss)")
        return None
    print(f"[train] class_weight: {weights}")
    return weights


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
    cache: bool = True,
) -> tf.data.Dataset:
    paths = [os.path.join(image_dir, p) for p in df[image_col]]
    labels = [int(y) for y in df[label_col]]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Preprocess FIRST, then cache the results. The per-image CLAHE/mask work is
    # the training bottleneck; without caching it re-runs every epoch. Training
    # preprocessing here is DETERMINISTIC (no train-time augmentation), so the
    # cache is numerically identical to recomputing — purely a speedup.
    ds = ds.map(
        lambda path, label: _load_and_preprocess(path, label, input_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Cache BEFORE shuffle so every epoch still reshuffles (caching after a
    # shuffle would freeze a single order). In-memory cache: ~200 KB/image, so a
    # ~55k-image split is ~11 GB — fits the training VM's RAM.
    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

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

def _checkpoint_path(checkpoint_dir: str, filename: str) -> str:
    """Join a checkpoint dir and filename, keeping forward slashes for gs:// URIs.

    os.path.join inserts a backslash on Windows, which corrupts gs:// paths.
    """
    if "://" in checkpoint_dir:
        return checkpoint_dir.rstrip("/") + "/" + filename
    return os.path.join(checkpoint_dir, filename)


def _build_callbacks(checkpoint_dir: str, filename: str = "best.keras") -> list:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=_checkpoint_path(checkpoint_dir, filename),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # Monitor the same metric as ModelCheckpoint (val_auc) so the weights
        # restored in memory match the best.keras written to disk. Mixing
        # val_loss here with val_auc above would let the saved checkpoint and
        # the returned model come from different epochs.
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

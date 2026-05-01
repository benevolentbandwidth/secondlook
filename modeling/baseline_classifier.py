# Baseline MobileNetV2 classifier for Second Look.

# Purpose: establish a performance floor before building the full
# ViT + CNN Detector + Fusion architecture. This model is intentionally
# simple — one backbone, one head, no fusion.

# Architecture:
#   Input (H, W, 1) grayscale
#     → 1x1 Conv projection to 3 channels  (TF Lite safe; avoids Lambda layers)
#     → MobileNetV2 backbone (frozen initially)
#     → GlobalAveragePooling
#     → Dropout
#     → Dense(1, sigmoid)  →  P(WORTH_SECOND_LOOK)

# TF Lite constraint: no Lambda layers, no custom ops.
# If conversion fails, simplify — never fight the runtime.

# Class weighting: biased toward sensitivity on the positive class.
# False reassurance (missing a WORTH_SECOND_LOOK case) is the worst failure
# mode per CLAUDE.md's failure-mode hierarchy.

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from config.constants import INPUT_SIZE


# Label index convention: 0 = NOT_WORTH_SECOND_LOOK, 1 = WORTH_SECOND_LOOK.
# Index matches Label enum values in data_pipeline.label_mapper; keeping
# this list for confusion-matrix bookkeeping in evaluate.py.
LABEL_ORDER = ["NOT_WORTH_SECOND_LOOK", "WORTH_SECOND_LOOK"]
POSITIVE_CLASS_INDEX = 1  # WORTH_SECOND_LOOK

# Minimum acceptable sensitivity for the positive class (WORTH_SECOND_LOOK)
# before a model is considered safe to use. Below this, the model risks
# false reassurance — the #1 failure mode in CLAUDE.md.
WORTH_SENSITIVITY_FLOOR = 0.80

# Extra weight applied to the positive class on top of class-balanced weights.
# Reflects the asymmetric cost of a missed WORTH case vs. a false alarm.
WORTH_WEIGHT_MULTIPLIER = 1.5


def build_baseline(
    input_size: tuple = INPUT_SIZE,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = True,
) -> tf.keras.Model:
    """Build the baseline MobileNetV2 classifier with a binary head.

    Args:
        input_size: (height, width) of the input image. Must be >= (32, 32).
        dropout_rate: Dropout before the classification head.
        freeze_backbone: If True, MobileNetV2 weights are frozen during initial
                         training. Set to False for fine-tuning after the head
                         has converged.

    Returns:
        Compiled-ready Keras Model with a sigmoid head — pair with
        binary_crossentropy in train.py.
    """
    inputs = tf.keras.Input(shape=(*input_size, 1), name="mammogram_input")

    # 1x1 conv projects grayscale → 3 channels so MobileNetV2 pretrained
    # weights apply. TF Lite safe (no Lambda/tf.repeat).
    x = layers.Conv2D(
        filters=3,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False,
        name="channel_expand",
    )(inputs)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*input_size, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = not freeze_backbone

    # Pass training=False so BatchNorm layers inside the frozen backbone
    # always run in inference mode, even during model.fit().
    x = backbone(x, training=not freeze_backbone)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Binary output: sigmoid on a single unit — P(WORTH_SECOND_LOOK).
    # Do NOT expose raw probabilities to the user. The UI maps confidence
    # to a tier via data_pipeline.label_mapper.confidence_to_tier(), then
    # renders via display_label().
    outputs = layers.Dense(1, activation="sigmoid", name="worth_second_look")(x)

    model = tf.keras.Model(inputs, outputs, name="second_look_baseline")
    return model


def compute_class_weights(labels: list[int]) -> dict[int, float]:
    """Compute class weights biased toward sensitivity on WORTH_SECOND_LOOK.

    Uses sklearn's balanced weighting as a base, then applies an additional
    multiplier to the positive class to reflect the asymmetric cost of
    false reassurance.

    Args:
        labels: List of binary labels (int 0 or 1) from the training set,
                matching the Label enum's integer values.

    Returns:
        Dict mapping class index → weight, ready for Keras class_weight arg.

    Raises:
        ValueError: If labels contain values other than 0 or 1.
    """
    from sklearn.utils.class_weight import compute_class_weight

    unknown = set(labels) - {0, 1}
    if unknown:
        raise ValueError(
            f"Unknown label values: {unknown}. Expected binary {{0, 1}}."
        )

    y = np.asarray(labels, dtype=np.int64)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y,
    )

    # Extra penalty for missing a WORTH case (false reassurance risk).
    weights[POSITIVE_CLASS_INDEX] *= WORTH_WEIGHT_MULTIPLIER

    return {i: float(w) for i, w in enumerate(weights)}

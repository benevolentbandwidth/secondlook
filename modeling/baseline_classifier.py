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
#     → Dense(3, softmax)  →  [Low, Moderate, Elevated]

# TF Lite constraint: no Lambda layers, no custom ops.
# If conversion fails, simplify — never fight the runtime.

# Class weighting: biased toward sensitivity on Elevated.
# False reassurance (missing an Elevated region) is the worst failure mode.

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# Tier order is fixed. Index 0 = Low, 1 = Moderate, 2 = Elevated.
# This must match label encoding everywhere: training, evaluation, and UI.
TIER_ORDER = ["Low", "Moderate", "Elevated"]

# Baseline uses 224x224 — standard MobileNetV2 input, fast to train.
INPUT_SIZE = (224, 224)
NUM_CLASSES = len(TIER_ORDER)

# Minimum acceptable sensitivity for the Elevated tier before a model is
# considered safe to use. Below this, the model risks false reassurance.
ELEVATED_SENSITIVITY_FLOOR = 0.80

# Extra weight applied to Elevated on top of class-balanced weights.
# Reflects the asymmetric cost of a missed Elevated region vs. a false alarm.
ELEVATED_WEIGHT_MULTIPLIER = 1.5


def build_baseline(
    input_size: tuple = INPUT_SIZE,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = True,
) -> tf.keras.Model:
    """Build the baseline MobileNetV2 classifier.

    Args:
        input_size: (height, width) of the input image. Must be >= (32, 32).
        dropout_rate: Dropout before the classification head.
        freeze_backbone: If True, MobileNetV2 weights are frozen during initial
                         training. Set to False for fine-tuning after the head
                         has converged.

    Returns:
        Compiled Keras Model ready for .fit().
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
    x = backbone(x, training=freeze_backbone is True and False or not freeze_backbone)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Output: softmax over 3 concern tiers.
    # Do NOT expose raw logits or probabilities to the user — the UI maps
    # the argmax to a display label via label_mapper.display_label().
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="concern_tier")(x)

    model = tf.keras.Model(inputs, outputs, name="second_look_baseline")
    return model


def compute_class_weights(tier_labels: list[str]) -> dict[int, float]:
    """Compute class weights biased toward sensitivity on Elevated.

    Uses sklearn's balanced weighting as a base, then applies an additional
    multiplier to Elevated to reflect the asymmetric cost of false reassurance.

    Args:
        tier_labels: List of tier strings ('Low', 'Moderate', 'Elevated')
                     from the training set.

    Returns:
        Dict mapping class index → weight, ready for Keras class_weight arg.

    Raises:
        ValueError: If tier_labels contains values not in TIER_ORDER.
    """
    from sklearn.utils.class_weight import compute_class_weight

    unknown = set(tier_labels) - set(TIER_ORDER)
    if unknown:
        raise ValueError(
            f"Unknown tier labels: {unknown}. Expected values from {TIER_ORDER}."
        )

    indices = np.array([TIER_ORDER.index(t) for t in tier_labels])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=indices,
    )

    # Extra penalty for missing an Elevated region (false reassurance risk).
    elevated_idx = TIER_ORDER.index("Elevated")
    weights[elevated_idx] *= ELEVATED_WEIGHT_MULTIPLIER

    return {i: float(w) for i, w in enumerate(weights)}

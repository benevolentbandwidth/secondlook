# Loss functions for the Second Look binary head.
#
# Why this module exists: the baseline trains with plain binary_crossentropy
# plus heavy class weighting (balanced ~= 8x on the positive class at the real
# ~6%-positive screening distribution, x1.5 on top of that). That combination
# buys WORTH sensitivity by flooding the positive side of the decision boundary
# -- the full-run weak point (WORTH precision 0.15-0.19, specificity ~0.56 at
# the 0.80 sensitivity floor).
#
# Focal loss attacks the same imbalance differently: instead of up-weighting
# every positive uniformly, it DOWN-weights examples the model already gets
# right (the (1-p_t)^gamma factor), so the gradient concentrates on the hard
# boundary cases. That is the mechanism expected to lift positive-class
# precision without giving up sensitivity.
#
# Interaction warning: focal's own alpha balancing, and Keras `class_weight`,
# and WORTH_WEIGHT_MULTIPLIER are THREE separate thumbs on the same scale.
# Stacking all of them is how a model collapses to all-positive. Pick one
# primary mechanism per run -- see build_loss's docstring.

from __future__ import annotations

import tensorflow as tf

# Supported values for the `loss` config key. Unknown names raise (never
# silently fall back to a default) per the failure-mode hierarchy in CLAUDE.md.
SUPPORTED_LOSSES = ("bce", "focal")

# Focal defaults from Lin et al. 2017 (RetinaNet). gamma=2 is the standard
# starting point; gamma=0 degenerates to (alpha-weighted) cross-entropy.
DEFAULT_FOCAL_GAMMA = 2.0
DEFAULT_FOCAL_ALPHA = 0.25


def build_loss(
    loss: str = "bce",
    focal_gamma: float = DEFAULT_FOCAL_GAMMA,
    focal_alpha: float | None = None,
):
    """Resolve a loss-name config value into a Keras loss.

    Args:
        loss: "bce" (binary crossentropy, the baseline) or "focal"
              (BinaryFocalCrossentropy).
        focal_gamma: Focusing parameter. Higher = more aggressively down-weight
              easy examples. 0 disables focusing; 2.0 is the standard default.
              Ignored when loss="bce".
        focal_alpha: Positive-class weight INSIDE the focal loss, in [0, 1].
              None (default) disables focal's internal class balancing, which
              is the right choice when Keras `class_weight` is already carrying
              the imbalance. Set it (e.g. 0.25) and turn class weighting OFF to
              let focal own the imbalance entirely. Ignored when loss="bce".

    Returns:
        A value accepted by ``model.compile(loss=...)``.

    Raises:
        ValueError: If ``loss`` is not one of SUPPORTED_LOSSES. Unknown config
            values must fail loudly rather than default to a class/behavior.
    """
    key = str(loss).lower()
    if key not in SUPPORTED_LOSSES:
        raise ValueError(
            f"Unknown loss '{loss}'. Expected one of {SUPPORTED_LOSSES}."
        )

    if key == "bce":
        return "binary_crossentropy"

    if focal_alpha is not None and not 0.0 <= float(focal_alpha) <= 1.0:
        raise ValueError(
            f"focal_alpha must be in [0, 1] (it is a class-balance weight, not "
            f"a multiplier); got {focal_alpha}."
        )
    if float(focal_gamma) < 0:
        raise ValueError(f"focal_gamma must be >= 0; got {focal_gamma}.")

    # apply_class_balancing is what switches alpha on. With alpha=None we leave
    # it off so the loss does pure hard-example focusing and the imbalance is
    # handled by exactly one mechanism (class_weight) instead of two.
    return tf.keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=focal_alpha is not None,
        alpha=DEFAULT_FOCAL_ALPHA if focal_alpha is None else float(focal_alpha),
        gamma=float(focal_gamma),
        from_logits=False,  # the head is Dense(1, activation="sigmoid")
    )


def describe_loss(
    loss: str, focal_gamma: float, focal_alpha: float | None, use_class_weights: bool
) -> str:
    """One-line human summary of the loss setup, for run logs and summaries."""
    if str(loss).lower() == "bce":
        base = "binary_crossentropy"
    else:
        alpha = "off" if focal_alpha is None else f"{focal_alpha}"
        base = f"focal(gamma={focal_gamma}, alpha={alpha})"
    return f"{base} + class_weight={'on' if use_class_weights else 'OFF'}"

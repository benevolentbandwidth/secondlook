# Evaluation for the Second Look baseline classifier.
#
# Sensitivity (recall) on the WORTH_SECOND_LOOK class is the primary metric.
# A model that misses WORTH cases causes false reassurance — the worst
# failure mode defined in CLAUDE.md.
#
# This evaluator will explicitly warn (and optionally raise) if positive-class
# sensitivity falls below the defined safety floor.
#
# Usage:
#   from modeling.evaluate import evaluate_baseline
#   results = evaluate_baseline(model, test_df, image_dir="data/images/")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    roc_curve,
)

from config.constants import INPUT_SIZE
from modeling.baseline_classifier import (
    LABEL_ORDER,
    POSITIVE_CLASS_INDEX,
    WORTH_SENSITIVITY_FLOOR,
)
from modeling.train import _build_dataset


def find_optimal_threshold(
    model: tf.keras.Model,
    val_df: pd.DataFrame,
    image_dir: str,
    image_col: str = "image_path",
    label_col: str = "label",
    input_size: tuple = INPUT_SIZE,
    batch_size: int = 32,
) -> float:
    """Find the decision threshold that maximises specificity while meeting the sensitivity floor.

    Sweeps the ROC curve on the validation set and picks the operating point where
    WORTH_SECOND_LOOK sensitivity >= WORTH_SENSITIVITY_FLOOR and FPR (false-alarm rate)
    is minimised. Falls back to 0.5 with a warning if no threshold meets the floor.

    Args:
        model: Trained Keras model (or path string to a saved .keras file).
        val_df: Validation split DataFrame — must NOT be the test set.
        image_dir: Root directory containing image files.
        image_col: Column with image filenames.
        label_col: Column with binary labels (int 0 or 1).
        input_size: Must match the size used during training.
        batch_size: Inference batch size.

    Returns:
        Optimal threshold float in (0, 1).
    """
    if isinstance(model, str):
        model = tf.keras.models.load_model(model)

    val_ds = _build_dataset(
        val_df, image_dir, image_col, label_col, input_size, batch_size, shuffle=False
    )
    true_labels = np.asarray([int(y) for y in val_df[label_col]])
    probabilities = model.predict(val_ds, verbose=0).ravel()

    # roc_curve returns fpr, tpr (=sensitivity), and the corresponding thresholds.
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities, pos_label=POSITIVE_CLASS_INDEX)

    # Keep only operating points that satisfy the sensitivity floor.
    valid_mask = tpr >= WORTH_SENSITIVITY_FLOOR
    if not valid_mask.any():
        print(
            f"WARNING: No threshold achieves sensitivity >= {WORTH_SENSITIVITY_FLOOR}. "
            "Falling back to 0.5. Consider retraining with stronger class weighting."
        )
        return 0.5

    # Among valid points, choose the one with the lowest FPR (fewest false alarms).
    best_idx = np.argmin(fpr[valid_mask])
    optimal = float(thresholds[valid_mask][best_idx])

    achieved_sensitivity = float(tpr[valid_mask][best_idx])
    achieved_specificity = float(1.0 - fpr[valid_mask][best_idx])
    print(
        f"\nOptimal threshold: {optimal:.3f} "
        f"(val sensitivity: {achieved_sensitivity:.3f}, "
        f"val specificity: {achieved_specificity:.3f})"
    )
    return optimal


def evaluate_baseline(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    image_dir: str,
    image_col: str = "image_path",
    label_col: str = "label",
    input_size: tuple = INPUT_SIZE,
    batch_size: int = 32,
    threshold: float = 0.5,
    raise_on_unsafe: bool = False,
) -> dict:
    """Evaluate the baseline model on the test set.

    Prints per-class sensitivity and a 2x2 confusion matrix.
    Flags explicitly if WORTH_SECOND_LOOK sensitivity is below the safety floor.

    Args:
        model: Trained Keras model (or path string to a saved .keras file).
        test_df: Test split DataFrame from splitter.split_dataset().
        image_dir: Root directory containing image files.
        image_col: Column with image filenames.
        label_col: Column with binary labels (int 0 or 1).
        input_size: Must match the size used during training.
        batch_size: Inference batch size.
        threshold: Sigmoid output threshold for positive classification.
                   Lower values favor sensitivity; higher values favor specificity.
        raise_on_unsafe: If True, raises RuntimeError when WORTH sensitivity is
                         below WORTH_SENSITIVITY_FLOOR. Use this in CI.

    Returns:
        Dict with keys: per_class_sensitivity, confusion_matrix, report_str,
        worth_sensitivity, passed_safety_floor, threshold.
    """
    if isinstance(model, str):
        model = tf.keras.models.load_model(model)

    test_ds = _build_dataset(
        test_df, image_dir, image_col, label_col, input_size, batch_size, shuffle=False
    )

    true_labels = np.asarray([int(y) for y in test_df[label_col]])
    probabilities = model.predict(test_ds, verbose=1).ravel()
    predicted_labels = (probabilities >= threshold).astype(np.int64)

    per_class_sensitivity = recall_score(
        true_labels, predicted_labels, average=None, labels=[0, 1]
    )
    worth_sensitivity = per_class_sensitivity[POSITIVE_CLASS_INDEX]

    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=LABEL_ORDER,
        digits=3,
        zero_division=0,
    )

    _print_results(per_class_sensitivity, worth_sensitivity, cm, report, threshold)

    passed = worth_sensitivity >= WORTH_SENSITIVITY_FLOOR
    if not passed:
        msg = (
            f"\nSAFETY WARNING: WORTH_SECOND_LOOK sensitivity is "
            f"{worth_sensitivity:.3f}, below the required floor of "
            f"{WORTH_SENSITIVITY_FLOOR}. This model risks false reassurance "
            f"and must not be deployed."
        )
        print(msg)
        if raise_on_unsafe:
            raise RuntimeError(msg)

    return {
        "per_class_sensitivity": {
            LABEL_ORDER[i]: float(s) for i, s in enumerate(per_class_sensitivity)
        },
        "confusion_matrix": cm,
        "report_str": report,
        "worth_sensitivity": float(worth_sensitivity),
        "passed_safety_floor": bool(passed),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_results(
    per_class_sensitivity: np.ndarray,
    worth_sensitivity: float,
    cm: np.ndarray,
    report: str,
    threshold: float,
) -> None:
    print("\n" + "=" * 60)
    print("SECOND LOOK — BASELINE EVALUATION")
    print("=" * 60)

    print(f"\nDecision threshold: {threshold:.2f}")

    print("\nSensitivity (Recall) per class:")
    for i, name in enumerate(LABEL_ORDER):
        marker = " ← PRIMARY METRIC" if i == POSITIVE_CLASS_INDEX else ""
        print(f"  {name:25s}: {per_class_sensitivity[i]:.3f}{marker}")

    floor = WORTH_SENSITIVITY_FLOOR
    status = "PASS" if worth_sensitivity >= floor else f"FAIL (floor: {floor})"
    print(f"\nWORTH_SECOND_LOOK sensitivity floor check: {status}")

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    header = f"{'':25s}" + "".join(f"{n:>25s}" for n in LABEL_ORDER)
    print(header)
    for i, name in enumerate(LABEL_ORDER):
        row = f"{name:25s}" + "".join(f"{cm[i, j]:>25d}" for j in range(len(LABEL_ORDER)))
        print(row)

    print("\nFull Classification Report:")
    print(report)
    print("=" * 60)
    print(
        "NOTE: Accuracy is not the primary metric here. A model that catches "
        "WORTH cases at the cost of some false alarms is preferable to one "
        "that misses WORTH cases. Review sensitivity first."
    )

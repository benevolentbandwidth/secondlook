# Evaluation for the Second Look baseline classifier.
#
# Sensitivity (recall) on the Elevated tier is the primary metric.
# A model that misses Elevated regions causes false reassurance — the worst
# failure mode defined in CLAUDE.md.
#
# This evaluator will explicitly warn (and optionally raise) if Elevated
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
)

from modeling.baseline_classifier import (
    TIER_ORDER,
    ELEVATED_SENSITIVITY_FLOOR,
    INPUT_SIZE,
)
from modeling.train import _build_dataset


def evaluate_baseline(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    image_dir: str,
    image_col: str = "image_path",
    tier_col: str = "concern_tier",
    input_size: tuple = INPUT_SIZE,
    batch_size: int = 32,
    raise_on_unsafe: bool = False,
) -> dict:
    """Evaluate the baseline model on the test set.

    Prints a full sensitivity/specificity breakdown per tier.
    Flags explicitly if Elevated sensitivity is below the safety floor.

    Args:
        model: Trained Keras model (or path string to a saved .keras file).
        test_df: Test split DataFrame from splitter.split_dataset().
        image_dir: Root directory containing image files.
        image_col: Column with image filenames.
        tier_col: Column with concern tier strings.
        input_size: Must match the size used during training.
        batch_size: Inference batch size.
        raise_on_unsafe: If True, raises RuntimeError when Elevated sensitivity
                         is below ELEVATED_SENSITIVITY_FLOOR. Use this in CI.

    Returns:
        Dict with keys: per_class_sensitivity, confusion_matrix, report_str,
        elevated_sensitivity, passed_safety_floor.
    """
    if isinstance(model, str):
        model = tf.keras.models.load_model(model)

    test_ds = _build_dataset(
        test_df, image_dir, image_col, tier_col, input_size, batch_size, shuffle=False
    )

    true_labels = np.array([TIER_ORDER.index(t) for t in test_df[tier_col]])
    predictions = model.predict(test_ds, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)

    per_class_sensitivity = recall_score(
        true_labels, predicted_labels, average=None, labels=list(range(len(TIER_ORDER)))
    )
    elevated_idx = TIER_ORDER.index("Elevated")
    elevated_sensitivity = per_class_sensitivity[elevated_idx]

    cm = confusion_matrix(true_labels, predicted_labels, labels=list(range(len(TIER_ORDER))))
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=TIER_ORDER,
        digits=3,
    )

    _print_results(per_class_sensitivity, elevated_sensitivity, cm, report)

    passed = elevated_sensitivity >= ELEVATED_SENSITIVITY_FLOOR
    if not passed:
        msg = (
            f"\nSAFETY WARNING: Elevated tier sensitivity is {elevated_sensitivity:.3f}, "
            f"below the required floor of {ELEVATED_SENSITIVITY_FLOOR}. "
            "This model risks false reassurance and must not be deployed."
        )
        print(msg)
        if raise_on_unsafe:
            raise RuntimeError(msg)

    return {
        "per_class_sensitivity": {TIER_ORDER[i]: float(s) for i, s in enumerate(per_class_sensitivity)},
        "confusion_matrix": cm,
        "report_str": report,
        "elevated_sensitivity": float(elevated_sensitivity),
        "passed_safety_floor": passed,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_results(
    per_class_sensitivity: np.ndarray,
    elevated_sensitivity: float,
    cm: np.ndarray,
    report: str,
) -> None:
    print("\n" + "=" * 60)
    print("SECOND LOOK — BASELINE EVALUATION")
    print("=" * 60)

    print("\nSensitivity (Recall) per Concern Tier:")
    for i, tier in enumerate(TIER_ORDER):
        marker = " ← PRIMARY METRIC" if tier == "Elevated" else ""
        print(f"  {tier:12s}: {per_class_sensitivity[i]:.3f}{marker}")

    floor = ELEVATED_SENSITIVITY_FLOOR
    status = "PASS" if elevated_sensitivity >= floor else f"FAIL (floor: {floor})"
    print(f"\nElevated sensitivity floor check: {status}")

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    header = f"{'':12s}" + "".join(f"{t:>12s}" for t in TIER_ORDER)
    print(header)
    for i, tier in enumerate(TIER_ORDER):
        row = f"{tier:12s}" + "".join(f"{cm[i, j]:>12d}" for j in range(len(TIER_ORDER)))
        print(row)

    print("\nFull Classification Report:")
    print(report)
    print("=" * 60)
    print(
        "NOTE: Accuracy is not the primary metric here. A model that catches "
        "all Elevated regions at the cost of some false alarms is preferable "
        "to one that misses Elevated regions. Review sensitivity first."
    )

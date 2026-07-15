# Evaluation for the Second Look baseline classifier.
#
# Sensitivity (recall) on the WORTH_SECOND_LOOK class is the primary metric.
# A model that misses WORTH cases causes false reassurance — the worst
# failure mode defined in CLAUDE.md.
#
# This evaluator reports the sensitivity-first protocol:
#   1. AUROC — the honest, threshold-INDEPENDENT discrimination metric. On
#      imbalanced data (CBIS is ~87% positive) raw accuracy/sensitivity at a
#      fixed 0.5 threshold are misleading; AUROC is not.
#   2. Operating point at the sensitivity floor — instead of a hard-coded 0.5,
#      pick the threshold that MAXIMIZES specificity subject to WORTH
#      sensitivity >= WORTH_SENSITIVITY_FLOOR (0.80). This is the operating
#      point we would actually deploy, per the failure-mode hierarchy.
#   3. Calibration — Brier score, Expected Calibration Error, and an optional
#      reliability diagram. Needed before confidence can drive the UX tiers.
#
# It still prints per-class sensitivity + confusion at the reference threshold
# and explicitly warns (and optionally raises) if positive-class sensitivity
# falls below the safety floor.
#
# Usage:
#   from modeling.evaluate import evaluate_baseline
#   results = evaluate_baseline(model, test_df, image_dir="data/images/")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config.constants import INPUT_SIZE
from modeling.baseline_classifier import (
    LABEL_ORDER,
    POSITIVE_CLASS_INDEX,
    WORTH_SENSITIVITY_FLOOR,
)
from modeling.train import _build_dataset

# Number of bins for the reliability diagram / Expected Calibration Error.
CALIBRATION_BINS = 10


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
    sensitivity_floor: float = WORTH_SENSITIVITY_FLOOR,
    output_dir: str | None = None,
) -> dict:
    """Evaluate the baseline model on the test set (sensitivity-first protocol).

    Reports AUROC, the operating point that meets the WORTH sensitivity floor
    at maximum specificity, calibration (Brier + ECE), and — at the reference
    ``threshold`` — per-class sensitivity and a 2x2 confusion matrix. Flags
    explicitly if WORTH_SECOND_LOOK sensitivity at ``threshold`` is below the
    safety floor.

    Args:
        model: Trained Keras model (or path string to a saved .keras file).
        test_df: Test split DataFrame.
        image_dir: Root directory containing image files (use "" if image_col
                   already holds absolute paths).
        image_col: Column with image filenames/paths.
        label_col: Column with binary labels (int 0 or 1).
        input_size: Must match the size used during training.
        batch_size: Inference batch size.
        threshold: Reference threshold for the confusion matrix + floor check.
                   NOTE: the deployable operating point is chosen separately
                   from the sensitivity floor (see operating_point in the
                   return value); this fixed threshold is kept for continuity
                   and as a sanity reference.
        raise_on_unsafe: If True, raises RuntimeError when WORTH sensitivity at
                         ``threshold`` is below the floor. Use this in CI.
        sensitivity_floor: Minimum acceptable WORTH sensitivity (default 0.80).
        output_dir: If given, write a reliability-diagram PNG here. Accepts a
                    local path or a gs:// URI (written via tf.io.gfile).

    Returns:
        Dict with keys: per_class_sensitivity, confusion_matrix, report_str,
        worth_sensitivity, passed_safety_floor, threshold, auroc,
        operating_point, brier_score, ece, reliability_diagram_path.
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

    # Threshold-independent + operating-point + calibration metrics.
    auroc = _compute_auroc(true_labels, probabilities)
    operating_point = _operating_point_at_floor(
        true_labels, probabilities, sensitivity_floor
    )
    brier, ece, curve = _calibration(true_labels, probabilities)
    diagram_path = None
    if output_dir is not None and curve is not None:
        diagram_path = _save_reliability_diagram(curve, brier, ece, output_dir)

    _print_results(
        per_class_sensitivity, worth_sensitivity, cm, report, threshold,
        auroc, operating_point, brier, ece, sensitivity_floor,
    )

    passed = worth_sensitivity >= sensitivity_floor
    if not passed:
        msg = (
            f"\nSAFETY WARNING: WORTH_SECOND_LOOK sensitivity is "
            f"{worth_sensitivity:.3f}, below the required floor of "
            f"{sensitivity_floor}. This model risks false reassurance "
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
        "auroc": auroc,
        "operating_point": operating_point,
        "brier_score": brier,
        "ece": ece,
        "reliability_diagram_path": diagram_path,
    }


# ---------------------------------------------------------------------------
# Threshold-independent + operating-point + calibration metrics
# ---------------------------------------------------------------------------

def _both_classes_present(true_labels: np.ndarray) -> bool:
    """AUROC / ROC-sweep need at least one sample of each class."""
    return len(np.unique(true_labels)) >= 2


def _compute_auroc(true_labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    """AUROC — the honest, threshold-independent metric. None if single-class."""
    if not _both_classes_present(true_labels):
        print(
            "\nNOTE: test set has a single class; AUROC is undefined and "
            "reported as None. (Sensitivity/specificity trade-offs are "
            "meaningless without both classes present.)"
        )
        return None
    return float(roc_auc_score(true_labels, probabilities))


def _operating_point_at_floor(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    sensitivity_floor: float,
) -> dict | None:
    """Pick the threshold with max specificity s.t. WORTH sensitivity >= floor.

    Walks the ROC curve (thresholds descending, sensitivity non-decreasing) and
    takes the FIRST point that clears the floor — that point has the smallest
    false-positive rate (highest specificity) among all points that satisfy the
    sensitivity requirement. This is the deployable operating point per the
    failure-mode hierarchy: meet the sensitivity floor first, then minimize
    false alarms. Returns None if no threshold meets the floor (or single-class).
    """
    if not _both_classes_present(true_labels):
        return None

    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    meets = tpr >= sensitivity_floor
    if not meets.any():
        print(
            f"\nNOTE: no threshold reaches the {sensitivity_floor:.2f} "
            f"sensitivity floor on this test set; no safe operating point exists."
        )
        return None

    idx = int(np.argmax(meets))  # first index where sensitivity clears the floor
    thr = float(thresholds[idx])
    # sklearn prepends an infinite threshold; clamp to 1.0 for a usable value.
    if not np.isfinite(thr):
        thr = 1.0
    return {
        "threshold": thr,
        "sensitivity": float(tpr[idx]),
        "specificity": float(1.0 - fpr[idx]),
    }


def _calibration(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> tuple[float, float, dict | None]:
    """Return (brier_score, expected_calibration_error, reliability_curve).

    ECE is the sample-weighted average gap between mean predicted probability
    and observed positive rate across equal-width probability bins. The curve
    dict holds per-bin (mean_predicted, observed_rate, count) for plotting.
    """
    brier = float(brier_score_loss(true_labels, probabilities))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index per sample; clip the right edge into the last bin.
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1)

    n = len(probabilities)
    ece = 0.0
    mean_pred, obs_rate, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(probabilities[mask].mean())
        acc = float(true_labels[mask].mean())
        ece += abs(acc - conf) * count / n
        mean_pred.append(conf)
        obs_rate.append(acc)
        counts.append(count)

    curve = {
        "mean_predicted": mean_pred,
        "observed_rate": obs_rate,
        "counts": counts,
    }
    return brier, float(ece), curve


def _save_reliability_diagram(
    curve: dict, brier: float, ece: float, output_dir: str
) -> str | None:
    """Write a reliability-diagram PNG to output_dir (local or gs://)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless — no display on training VMs
        import matplotlib.pyplot as plt
    except ImportError:
        print("NOTE: matplotlib unavailable; skipping reliability diagram.")
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax.plot(curve["mean_predicted"], curve["observed_rate"], "o-", label="model")
    ax.set_xlabel("Mean predicted P(WORTH_SECOND_LOOK)")
    ax.set_ylabel("Observed WORTH rate")
    ax.set_title(f"Reliability diagram (Brier={brier:.3f}, ECE={ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()

    dest = output_dir.rstrip("/") + "/reliability_diagram.png"
    if "://" in output_dir:
        # Write locally then copy to the remote filesystem via tf.io.gfile.
        import os
        import tempfile
        tf.io.gfile.makedirs(output_dir)
        tmp = os.path.join(tempfile.gettempdir(), "reliability_diagram.png")
        fig.savefig(tmp, dpi=120)
        plt.close(fig)
        tf.io.gfile.copy(tmp, dest, overwrite=True)
    else:
        tf.io.gfile.makedirs(output_dir)
        fig.savefig(dest, dpi=120)
        plt.close(fig)
    print(f"Reliability diagram written to: {dest}")
    return dest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_results(
    per_class_sensitivity: np.ndarray,
    worth_sensitivity: float,
    cm: np.ndarray,
    report: str,
    threshold: float,
    auroc: float | None,
    operating_point: dict | None,
    brier: float,
    ece: float,
    sensitivity_floor: float,
) -> None:
    print("\n" + "=" * 60)
    print("SECOND LOOK - BASELINE EVALUATION")
    print("=" * 60)

    # Threshold-independent discrimination first — the honest headline number.
    auroc_str = f"{auroc:.3f}" if auroc is not None else "N/A (single-class test set)"
    print(f"\nAUROC (threshold-independent): {auroc_str}")

    print("\nDeployable operating point (max specificity at the sensitivity floor):")
    if operating_point is None:
        print(f"  NONE — no threshold meets the {sensitivity_floor:.2f} "
              f"sensitivity floor on this test set.")
    else:
        print(f"  threshold  : {operating_point['threshold']:.3f}")
        print(f"  sensitivity: {operating_point['sensitivity']:.3f} "
              f"(WORTH; floor {sensitivity_floor:.2f})")
        print(f"  specificity: {operating_point['specificity']:.3f}")

    print("\nCalibration:")
    print(f"  Brier score: {brier:.3f}  (lower is better)")
    print(f"  ECE        : {ece:.3f}  (expected calibration error, lower better)")

    print("\n" + "-" * 60)
    print(f"Reference threshold: {threshold:.2f}")

    print("\nSensitivity (Recall) per class:")
    for i, name in enumerate(LABEL_ORDER):
        marker = " <-- PRIMARY METRIC" if i == POSITIVE_CLASS_INDEX else ""
        print(f"  {name:25s}: {per_class_sensitivity[i]:.3f}{marker}")

    status = "PASS" if worth_sensitivity >= sensitivity_floor else f"FAIL (floor: {sensitivity_floor})"
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
        "NOTE: Accuracy is not the primary metric here. On imbalanced data, "
        "read AUROC + the operating point at the sensitivity floor first. A "
        "model that catches WORTH cases at the cost of some false alarms is "
        "preferable to one that misses WORTH cases."
    )

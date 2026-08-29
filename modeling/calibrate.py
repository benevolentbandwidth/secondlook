# Probability calibration for the Second Look binary head.
#
# The full combined run reported ECE ~= 0.36: the model's sigmoid output is not
# a probability in any usable sense. That is fine for ranking (AUROC is
# unaffected) but blocks the UX layer, where label_mapper.confidence_to_tier
# turns a confidence into Low / Moderate / Elevated. Those thresholds are
# explicit placeholders "pending calibration" -- this module is that calibration.
#
# Method: TEMPERATURE SCALING (Guo et al. 2017). One scalar T divides the logit
# before the sigmoid. Chosen over Platt/isotonic because it is:
#   * monotonic  -> AUROC, the ROC curve, and therefore the operating point
#                   chosen at the 0.80 sensitivity floor are ALL unchanged.
#                   Calibration here never trades away sensitivity.
#   * one parameter -> essentially cannot overfit the validation split.
#   * trivially portable -> a single float the on-device/TF Lite path can apply
#                   after the sigmoid, with no graph surgery.
#
# T is fit on the VALIDATION split and applied to test. Fitting on test would
# leak, and the resulting ECE would be meaningless.

from __future__ import annotations

import numpy as np

# Clip probabilities away from 0/1 before the logit so the transform stays
# finite on saturated sigmoid outputs (which this model produces plenty of).
# Kept tight (not 1e-6) because clipping is what turns distinct tail scores into
# TIES: everything below the clip maps to one logit. Ties in the far tail are
# harmless for calibration but needlessly coarsen the ranking, so give the tail
# as much room as float64 comfortably allows. log(1e-12/(1-1e-12)) ~= -27.6.
_EPS = 1e-12

# Search grid for T. Log-spaced coarse pass then a local refine; a closed-form
# solution does not exist and this avoids adding a scipy dependency to the
# Vertex training package.
_T_MIN, _T_MAX = 0.05, 20.0
_COARSE_STEPS = 200
_REFINE_ROUNDS = 4

CALIBRATION_BINS = 10


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    """Invert the sigmoid, clipping to keep the result finite."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale sigmoid outputs by a temperature: sigmoid(logit(p) / T).

    T > 1 softens confidence toward 0.5 (the usual fix for an over-confident
    model); T < 1 sharpens it. T = 1 is a no-op.

    The map is NON-DECREASING, which is the property that matters: the ROC curve
    and therefore the operating point chosen at the sensitivity floor are
    unchanged, so calibration can never cost WORTH sensitivity. It is not
    strictly increasing at the extremes, where _EPS clipping merges
    already-indistinguishable scores into ties.
    """
    t = float(temperature)
    if t <= 0:
        raise ValueError(f"temperature must be > 0; got {temperature}.")
    logits = probabilities_to_logits(probabilities) / t
    return 1.0 / (1.0 + np.exp(-logits))


def negative_log_likelihood(
    labels: np.ndarray, probabilities: np.ndarray
) -> float:
    """Mean binary NLL -- the objective temperature scaling minimizes."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), _EPS, 1.0 - _EPS)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature(
    labels: np.ndarray,
    probabilities: np.ndarray,
    verbose: bool = True,
) -> float:
    """Fit the temperature that minimizes NLL on a held-out split (use VAL).

    Args:
        labels: Binary ground truth (0/1) for the validation split.
        probabilities: The model's sigmoid outputs on that same split.
        verbose: Print the NLL/ECE improvement.

    Returns:
        The fitted temperature T (float). 1.0 means already calibrated.

    Raises:
        ValueError: If the inputs disagree in length, are empty, or the split
            contains a single class (calibration is undefined without both).
    """
    labels = np.asarray(labels).ravel()
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    if len(labels) != len(probabilities):
        raise ValueError(
            f"labels ({len(labels)}) and probabilities ({len(probabilities)}) "
            f"must be the same length."
        )
    if len(labels) == 0:
        raise ValueError("Cannot fit a temperature on an empty split.")
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Calibration split contains a single class; temperature scaling is "
            "undefined. (Do not fall back to T=1 silently -- an uncalibrated "
            "model must not be reported as calibrated.)"
        )

    logits = probabilities_to_logits(probabilities)
    y = labels.astype(np.float64)

    def nll_at(t: float) -> float:
        p = 1.0 / (1.0 + np.exp(-logits / t))
        p = np.clip(p, _EPS, 1.0 - _EPS)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    # Coarse log-spaced scan, then successively narrower local refinements.
    lo, hi = np.log(_T_MIN), np.log(_T_MAX)
    best_t = 1.0
    for _ in range(_REFINE_ROUNDS):
        grid = np.exp(np.linspace(lo, hi, _COARSE_STEPS))
        losses = [nll_at(t) for t in grid]
        idx = int(np.argmin(losses))
        best_t = float(grid[idx])
        span = (hi - lo) / _COARSE_STEPS
        lo, hi = np.log(best_t) - span * 2, np.log(best_t) + span * 2

    if verbose:
        before_nll, after_nll = nll_at(1.0), nll_at(best_t)
        before_ece = expected_calibration_error(labels, probabilities)
        after_ece = expected_calibration_error(
            labels, apply_temperature(probabilities, best_t)
        )
        print(f"\n[calibrate] fitted temperature T = {best_t:.4f} (on val)")
        print(f"[calibrate]   NLL {before_nll:.4f} -> {after_nll:.4f}")
        print(f"[calibrate]   ECE {before_ece:.4f} -> {after_ece:.4f}")
        print("[calibrate]   NOTE: monotonic, so AUROC and the operating point "
              "at the sensitivity floor are unchanged by design.")
    return best_t


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> float:
    """Sample-weighted mean gap between predicted confidence and observed rate."""
    _, ece, _ = calibration_curve(labels, probabilities, n_bins=n_bins)
    return ece


def calibration_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> tuple[float, float, dict]:
    """Return (brier, ece, curve) over equal-width probability bins.

    The curve dict holds per-bin (mean_predicted, observed_rate, count) for the
    reliability diagram. Shared by modeling.evaluate so the evaluation protocol
    and the calibration module cannot drift apart.
    """
    labels = np.asarray(labels).ravel().astype(np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()

    brier = float(np.mean((probabilities - labels) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index per sample; clip the right edge into the last bin.
    bin_ids = np.clip(
        np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1
    )

    n = len(probabilities)
    ece = 0.0
    mean_pred, obs_rate, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(probabilities[mask].mean())
        acc = float(labels[mask].mean())
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

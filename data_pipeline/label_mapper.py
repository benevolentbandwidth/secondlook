# Maps dataset-specific labels to a unified "Second Look" decision.

# Design rationale:
#   Each dataset encodes outcomes differently:
#       - CBIS-DDSM → pathology (post-biopsy)
#       - RSNA      → cancer classification (0/1)
#       - VinDr     → BI-RADS categories (radiologist assessment)
#   These are not directly comparable, so we map them into a shared,
#   non-clinical decision space.


# Mapping principle:
#   WORTH_SECOND_LOOK     → requires recall, follow-up, or biopsy
#   NOT_WORTH_SECOND_LOOK → confidently non-actionable


# Dataset-specific interpretation:
#
#   CBIS-DDSM (biopsy-driven):
#       MALIGNANT               → WORTH_SECOND_LOOK
#       BENIGN                  → WORTH_SECOND_LOOK  (biopsy performed → suspicious)
#       BENIGN_WITHOUT_CALLBACK → NOT_WORTH_SECOND_LOOK
#
#   RSNA (screening-scale):
#       cancer = 1 → WORTH_SECOND_LOOK
#       cancer = 0 → NOT_WORTH_SECOND_LOOK
#
#   VinDr-Mammo (BI-RADS-based):
#       BI-RADS 1–3 → NOT_WORTH_SECOND_LOOK
#       BI-RADS 4–5 → WORTH_SECOND_LOOK
#       (Matches the INbreast rule; unified across BI-RADS datasets.)

# Safety note:
#   Unknown labels raise ValueError rather than defaulting silently.
#   Per the failure mode hierarchy, unrecognized input must never
#   produce confident output.

from enum import Enum
from typing import Any


class Label(Enum):
    """
    Canonical binary label used across all datasets.

    Attributes:
        WORTH_SECOND_LOOK:
            Case warrants additional diagnostic attention.

        NOT_WORTH_SECOND_LOOK:
            Case is confidently non-actionable.
    """
    WORTH_SECOND_LOOK = 1
    NOT_WORTH_SECOND_LOOK = 0


# --- CBIS-DDSM ---
CBIS_MAP = {
    "MALIGNANT": Label.WORTH_SECOND_LOOK,
    "BENIGN": Label.WORTH_SECOND_LOOK,
    "BENIGN_WITHOUT_CALLBACK": Label.NOT_WORTH_SECOND_LOOK,
}


def map_cbis(pathology: str) -> Label:
    """
    Map CBIS-DDSM pathology to Label.

    MALIGNANT and BENIGN → WORTH_SECOND_LOOK
    BENIGN_WITHOUT_CALLBACK → NOT_WORTH_SECOND_LOOK

    Raises:
        ValueError: If label is unknown.
    """
    # Normalize string to match dictionary keys
    key = pathology.strip().upper().replace(" ", "_")

    if key not in CBIS_MAP:
        raise ValueError(f"Unknown CBIS label: {pathology}")

    return CBIS_MAP[key]


# --- RSNA ---
def map_rsna(cancer: int) -> Label:
    """
    Map RSNA cancer label (0/1) to Label.

    Raises:
        ValueError: If input is not 0 or 1.
    """
    if cancer not in (0, 1):
        raise ValueError(f"Invalid RSNA label: {cancer}")

    return Label.WORTH_SECOND_LOOK if cancer == 1 else Label.NOT_WORTH_SECOND_LOOK


# --- VinDr ---
def map_vindr(birads) -> Label:
    """
    Map BI-RADS (e.g. 'BI-RADS 4') to Label.

    BI-RADS 4–5 → WORTH_SECOND_LOOK
    BI-RADS 1–3 → NOT_WORTH_SECOND_LOOK

    Unified with the INbreast rule so BI-RADS datasets share a single
    binary cut-point.
    """
    # Handle string format like "BI-RADS 4"
    if isinstance(birads, str):
        # Extract digits only
        digits = "".join(filter(str.isdigit, birads))
        if not digits:
            raise ValueError(f"Invalid BI-RADS string: {birads}")
        birads = int(digits)

    if birads < 0 or birads > 6:
        raise ValueError(f"Invalid BI-RADS value: {birads}. Expected 0–6.")

    return Label.WORTH_SECOND_LOOK if birads >= 4 else Label.NOT_WORTH_SECOND_LOOK


# --- INbreast ---
# Project decision: BI-RADS 1–3 → NOT_WORTH, 4–5 → WORTH.
# This is stricter than the VinDr rule (which places BI-RADS 3 on the WORTH side)
# and is specific to INbreast per the project owner.
def map_inbreast(birads) -> Label:
    """
    Map INbreast BI-RADS (int or 'BI-RADS N' string) to Label.

    BI-RADS 4–5 → WORTH_SECOND_LOOK
    BI-RADS 1–3 → NOT_WORTH_SECOND_LOOK
    """
    if isinstance(birads, str):
        digits = "".join(filter(str.isdigit, birads))
        if not digits:
            raise ValueError(f"Invalid BI-RADS string: {birads}")
        birads = int(digits)

    if birads not in (1, 2, 3, 4, 5):
        raise ValueError(f"Invalid INbreast BI-RADS value: {birads}. Expected 1–5.")

    return Label.WORTH_SECOND_LOOK if birads >= 4 else Label.NOT_WORTH_SECOND_LOOK


def map_dataset(dataset: str, value) -> Label:
    """
    Map a dataset-specific label to the unified Label.

    Args:
        dataset: One of {'cbis', 'rsna', 'vindr', 'inbreast'}.
        value: Raw label value for that dataset.

    Returns:
            Label enum.

    Raises:
        ValueError: If dataset is unknown or input is invalid.
    """
    dataset = dataset.lower()  # normalize input

    if dataset == "cbis":
        return map_cbis(value)
    elif dataset == "rsna":
        return map_rsna(value)
    elif dataset == "vindr":
        return map_vindr(value)
    elif dataset == "inbreast":
        return map_inbreast(value)
    else:
        # Explicit failure for unknown dataset
        raise ValueError(f"Unknown dataset: {dataset}")


def to_int(label: Label) -> int:
    """
    Convert Label enum to integer (0/1).

    Returns:
        1 for WORTH_SECOND_LOOK, 0 for NOT_WORTH_SECOND_LOOK.

    Notes:
        Use this for model training (e.g., TensorFlow, PyTorch),
        where numeric targets are required.
    """
    return label.value


# --- UX tier rendering (decoupled from model labels per CLAUDE.md) ---
#
# The model head is binary. The three tiers below are a UX-layer
# presentation concern derived from model confidence, NOT from the model
# output directly. Keeping these functions in the same module (but in a
# clearly separated block) so the single call site
# `display_label(confidence_to_tier(prob))` is discoverable without
# importing from two places.

VALID_TIERS = {"Low", "Moderate", "Elevated"}

TIER_DISPLAY_LABELS = {
    "Low": "Low Area of Interest",
    "Moderate": "Moderate Area of Interest",
    "Elevated": "Elevated Area of Interest",
}

# Tier cut-points on the positive-class probability:
#   prob < low_max       → Low
#   prob < moderate_max  → Moderate
#   otherwise            → Elevated
#
# PROVISIONAL — these evenly-spaced defaults are placeholders so the UI has
# something functional to render during integration. They are NOT clinically
# calibrated. calibrate_thresholds() is the home for replacing them with
# asymmetric, sensitivity-favoring cuts once validation data exists.
TIER_THRESHOLDS = {"low_max": 0.33, "moderate_max": 0.66}


def confidence_to_tier(prob: float, thresholds: dict[str, float] = TIER_THRESHOLDS) -> str:
    """Map a positive-class probability to a UX concern tier.

    The cut-points come from ``thresholds`` (defaults to the provisional
    ``TIER_THRESHOLDS``). Pass a calibrated dict from ``calibrate_thresholds``
    once validation data is available; the defaults are NOT clinically
    calibrated and the real cuts will almost certainly be asymmetric (pushing
    the Elevated threshold lower to favor sensitivity per the failure-mode
    hierarchy).

    Args:
        prob: Positive-class probability in [0.0, 1.0] — the sigmoid output
              of the binary Second Look model.
        thresholds: Mapping with 'low_max' and 'moderate_max' cut-points.

    Returns:
        One of 'Low', 'Moderate', 'Elevated'.

    Raises:
        ValueError: If prob is outside [0.0, 1.0].
    """
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"Probability out of range [0, 1]: {prob}")

    if prob < thresholds["low_max"]:
        return "Low"
    if prob < thresholds["moderate_max"]:
        return "Moderate"
    return "Elevated"


def calibrate_thresholds(val_df: Any, model: Any) -> dict[str, float]:
    """Derive calibrated tier cut-points from validation data. NOT YET IMPLEMENTED.

    This is the designated home for replacing the provisional, evenly-spaced
    ``TIER_THRESHOLDS`` with data-driven cut-points. The intended contract:

    1. Run ``model`` over the validation images in ``val_df`` to get a
       positive-class probability per case, paired with its true binary label.
    2. Choose ``moderate_max`` (the Moderate→Elevated boundary) as the
       operating point that meets the positive-class sensitivity requirement —
       ``WORTH_SENSITIVITY_FLOOR`` in ``modeling.baseline_classifier`` (0.80).
       Per the failure-mode hierarchy, this cut is deliberately ASYMMETRIC:
       pushed lower than a balanced 0.66 so borderline cases escalate to
       Elevated rather than risk false reassurance.
    3. Choose ``low_max`` (the Low→Moderate boundary) to keep the Low tier
       high-specificity — only confidently non-actionable cases land in Low.

    Args:
        val_df: Validation split (e.g. a manifest DataFrame) with image paths
                and binary ``canonical_label`` values.
        model: Trained binary classifier exposing a ``predict``-style call that
               returns positive-class probabilities.

    Returns:
        A dict with 'low_max' and 'moderate_max', drop-in compatible with
        ``confidence_to_tier(prob, thresholds=...)``.

    Raises:
        NotImplementedError: Always, until calibration is implemented against a
            trained model and a built validation split.
    """
    raise NotImplementedError(
        "Tier threshold calibration is not implemented yet. "
        "confidence_to_tier() currently uses the provisional TIER_THRESHOLDS. "
        "See this function's docstring for the intended calibration contract."
    )


def display_label(tier: str) -> str:
    """Return the UI-safe display label for a concern tier.

    Use this whenever rendering tier text in the app or logs.
    Never expose raw tier strings or clinical terms to the user.

    Args:
        tier: One of 'Low', 'Moderate', 'Elevated'.

    Returns:
        A calm, non-diagnostic display string.

    Raises:
        ValueError: If tier is not a valid concern tier.
    """
    if tier not in VALID_TIERS:
        raise ValueError(
            f"'{tier}' is not a valid concern tier. Expected one of: {sorted(VALID_TIERS)}"
        )
    return TIER_DISPLAY_LABELS[tier]

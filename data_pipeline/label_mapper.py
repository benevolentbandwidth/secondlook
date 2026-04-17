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
#       BI-RADS 1–2 → NOT_WORTH_SECOND_LOOK
#       BI-RADS 3–6 → WORTH_SECOND_LOOK
#       (BI-RADS 3 included to prioritize sensitivity)

# Safety note:
#   Unknown labels raise ValueError rather than defaulting silently.
#   Per the failure mode hierarchy, unrecognized input must never
#   produce confident output.

from enum import Enum


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
    Map BI-RADS (e.g. 'BI-RADS 3') to Label.

    BI-RADS ≥ 3 → WORTH_SECOND_LOOK
    BI-RADS ≤ 2 → NOT_WORTH_SECOND_LOOK
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

    # Decision threshold (>=3)
    return Label.WORTH_SECOND_LOOK if birads >= 3 else Label.NOT_WORTH_SECOND_LOOK


def map_dataset(dataset: str, value) -> Label:
    """
    Map a dataset-specific label to the unified Label.

    Args:
        dataset: One of {'cbis', 'rsna', 'vindr'}.
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

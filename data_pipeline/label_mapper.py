# Maps CBIS-DDSM pathology labels to Second Look concern tiers.

# Design rationale:
#   CBIS-DDSM pathology values reflect radiologist assessment at time of biopsy.
#   We map them to three coarse, non-clinical tiers. These tiers are intentionally
#   vague — they communicate relative concern without implying diagnosis.

# Tier assignments:
#   MALIGNANT               → Elevated  (confirmed malignancy in source data)
#   BENIGN                  → Moderate  (benign, but radiologist requested follow-up)
#   BENIGN_WITHOUT_CALLBACK → Low       (benign, no follow-up indicated)

# Safety note: Unknown labels raise ValueError rather than defaulting silently.
# Per the failure mode hierarchy, unrecognized input must never produce confident output.

PATHOLOGY_TO_TIER = {
    "MALIGNANT": "Elevated",
    "BENIGN": "Moderate",
    "BENIGN_WITHOUT_CALLBACK": "Low",
}

VALID_TIERS = {"Low", "Moderate", "Elevated"}

# These are the only strings allowed to appear in model outputs and UI labels.
# Do not add clinical synonyms here — use the tier names exclusively.
TIER_DISPLAY_LABELS = {
    "Low": "Low Area of Interest",
    "Moderate": "Moderate Area of Interest",
    "Elevated": "Elevated Area of Interest",
}

def map_label(pathology: str) -> str:
    """Map a single CBIS-DDSM pathology string to a concern tier.

    Args:
        pathology: Raw pathology value from CBIS-DDSM CSV
                   (e.g. 'MALIGNANT', 'BENIGN', 'BENIGN_WITHOUT_CALLBACK').

    Returns:
        One of: 'Low', 'Moderate', 'Elevated'.

    Raises:
        ValueError: If the label is not one of the three known CBIS-DDSM values.
    """
    key = pathology.strip().upper().replace(" ", "_")
    if key not in PATHOLOGY_TO_TIER:
        raise ValueError(
            f"Unknown pathology label: '{pathology}'. "
            f"Expected one of: {sorted(PATHOLOGY_TO_TIER.keys())}. "
            "Do not add a default fallback — unknown labels must surface explicitly."
        )
    return PATHOLOGY_TO_TIER[key]

def map_labels(series) -> "pd.Series":
    """Apply map_label across a pandas Series of pathology strings.

    Args:
        series: pd.Series of raw pathology strings.

    Returns:
        pd.Series of concern tier strings ('Low', 'Moderate', 'Elevated').

    Raises:
        ValueError: If any value in the Series is not a known pathology label.
    """
    return series.apply(map_label)

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

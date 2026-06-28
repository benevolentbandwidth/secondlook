# Input-quality gate for Second Look.
#
# Rejects images that cannot be analyzed reliably (blank, low-resolution,
# low-contrast, or mostly background). Per CLAUDE.md's failure-mode
# hierarchy, rejection reasons must be surfaced to the caller — never
# silently discarded.
#
# Two related but distinct gates live here:
#   - quality_check: raw-input gate, run on the image before preprocessing
#                    returns (bool, reason_str).
#   - quality_gate:  augmentation-output realism gate (added after the
#                    augmentation-module merge). Returns a qualitative label.

import cv2
import numpy as np

from data_pipeline._imaging_utils import breast_mask, to_grayscale


def quality_check(image: np.ndarray) -> tuple[bool, str]:
    """Gate to reject images that cannot be analyzed reliably.

    Checks for:
    - Sufficient breast tissue coverage (not mostly background)
    - Minimum contrast (not overexposed or blank)
    - Minimum resolution

    Args:
        image: Raw image array (grayscale or RGB).

    Returns:
        (True, "") if the image passes, or (False, reason_string) if rejected.
        Callers must surface the reason string to the user — never silently discard.
    """
    if image is None or image.size == 0:
        return False, "Image is empty or could not be loaded."

    gray = to_grayscale(image)

    h, w = gray.shape
    if h < 256 or w < 256:
        return False, (
            "Image resolution is too low for reliable analysis. "
            "Please provide a clearer photo or scan."
        )

    mask = breast_mask(gray)
    tissue_fraction = mask.sum() / (255.0 * mask.size)
    if tissue_fraction < 0.10:
        return False, (
            "Too little breast tissue detected. The image may be cropped, "
            "overexposed, or not a mammogram. Please retake."
        )

    # Contrast check: std dev of tissue pixels should be meaningful.
    tissue_pixels = gray[mask > 0]
    if tissue_pixels.std() < 8.0:
        return False, (
            "Image contrast is too low for analysis. "
            "This may be caused by glare, overexposure, or a very low-quality photo. "
            "Please retake with better lighting."
        )

    return True, ""


def quality_gate(img: np.ndarray) -> str:
    """Augmentation-realism gate.

    Run on an augmented image to decide whether it is realistic enough to
    keep as a training sample. Distinct from `quality_check`: this gate
    evaluates post-augmentation quality with a qualitative label instead of
    a rejection boolean.

    Checks sharpness (Laplacian variance), mean brightness extremes, and
    the fraction of near-black pixels. Each failing check counts as one
    issue; the label is the bucket the issue count falls into.

    Returns:
        'USABLE' (0 issues), 'BORDERLINE' (1), or 'UNUSABLE' (2+).
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    issues = 0
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 8:
        issues += 1
    if np.mean(gray) < 20:
        issues += 1
    if np.mean(gray) > 240:
        issues += 1
    if np.mean(gray < 15) > 0.75:
        issues += 1
    return 'USABLE' if issues == 0 else ('BORDERLINE' if issues == 1 else 'UNUSABLE')

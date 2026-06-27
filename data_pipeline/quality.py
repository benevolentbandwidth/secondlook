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

    gray = _to_grayscale(image)

    h, w = gray.shape
    if h < 256 or w < 256:
        return False, (
            "Image resolution is too low for reliable analysis. "
            "Please provide a clearer photo or scan."
        )

    mask = _breast_mask(gray)
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


# ---------------------------------------------------------------------------
# Internal helpers (duplicated from preprocessor to avoid a circular import;
# both modules need cheap grayscale + breast-mask primitives, and the
# preprocessor is a heavier dependency to pull in from the quality gate.)
# ---------------------------------------------------------------------------

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        # Single-channel 3D (e.g. tf.image.decode_png(channels=1)) is already
        # grayscale; drop the trailing axis to a 2D (H, W) array.
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _breast_mask(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels < 2:
        return binary
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

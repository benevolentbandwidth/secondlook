"""Shared low-level imaging primitives for the data pipeline.

Thin, dependency-light helpers (cv2 + numpy only — no pipeline imports) used by
both ``preprocessor`` and ``quality``. Extracted so the grayscale + breast-mask
logic lives in exactly one place; previously each module kept a verbatim copy to
dodge a circular import, which risked the two copies silently diverging as the
masking logic evolves.
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse an image to a single-channel 2D array.

    Accepts 2D grayscale, (H, W, 1), 3-channel BGR, or 4-channel BGRA. The
    source bit depth (uint8 / uint16) is preserved.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return np.ascontiguousarray(image[:, :, 0])
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def breast_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask isolating breast tissue from background.

    Uses Otsu thresholding + largest-connected-component selection + a
    morphological close to fill small holes. Returns a uint8 mask where
    255 = tissue, 0 = background.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep only the largest connected component (the breast).
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels < 2:
        # Fallback: return the full image as mask if segmentation fails.
        return binary

    # Label 0 is background; find the largest foreground component.
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Morphological close to fill small holes in tissue.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Preprocessing pipeline for Second Look.

# Produces normalized arrays ready for training / TF Lite inference.

# Steps applied in order:
#   1. Convert to grayscale (mammograms are single-channel)
#   2. CLAHE — improves local contrast without amplifying noise globally
#   3. Breast masking — zeros out background (air), leaving only tissue
#   4. Pectoral muscle removal — removes the dense triangle in MLO views
#   5. Orientation normalization — flips so the breast faces right consistently
#   6. Resize + normalize to [0, 1] float32

# Output shape: (H, W, 1) — single channel, float32.

# Input-quality gating lives in data_pipeline.quality (quality_check).

from pathlib import Path

import cv2
import numpy as np

from config.constants import INPUT_SIZE
from data_pipeline._imaging_utils import breast_mask, to_grayscale


# Default target size: MobileNetV2 / EfficientNetB0 standard input, per CLAUDE.md.
DEFAULT_SIZE = INPUT_SIZE  # (224, 224)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(image: np.ndarray, target_size: tuple = DEFAULT_SIZE) -> np.ndarray:
    """Full preprocessing pipeline for a single mammogram image.

    Args:
        image: Raw image as a numpy array (grayscale or RGB, uint8 or uint16).
        target_size: (width, height) to resize to after all preprocessing steps.

    Returns:
        Float32 numpy array of shape (H, W, 1), values in [0, 1].

    Raises:
        ValueError: If the input array is empty or has an unsupported number of channels.
    """
    if image is None or image.size == 0:
        raise ValueError("Received an empty image array.")

    gray = to_grayscale(image)
    clahe = _apply_clahe(gray)
    mask = breast_mask(clahe)
    masked = cv2.bitwise_and(clahe, clahe, mask=mask)
    no_pec = _remove_pectoral(masked, mask)
    oriented = _normalize_orientation(no_pec, mask)
    resized = cv2.resize(oriented, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized[:, :, np.newaxis]  # (H, W, 1)


def load_image(path: str | Path) -> np.ndarray:
    """Load a mammogram image from disk into a numpy array.

    Preserves the source bit depth (8- or 16-bit) so downstream CLAHE can
    enhance from the full dynamic range. CBIS-DDSM PNGs converted from
    DICOM are often 16-bit grayscale.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")
    image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to decode image (corrupt or unsupported format): {file_path}")
    return image


def load_and_preprocess(path: str | Path, target_size: tuple = DEFAULT_SIZE) -> np.ndarray:
    """Convenience: load an image file and run the full preprocessing pipeline."""
    return preprocess(load_image(path), target_size=target_size)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the native bit depth, then normalize to uint8.

    CLAHE runs on the original depth (OpenCV handles uint16 natively) so the
    contrast step uses the full dynamic range of DICOM-converted PNGs before
    any quantization. Only after enhancement do we normalize down to uint8.

    The CLAHE object is created per call rather than shared at module level:
    OpenCV's CLAHE instances hold internal buffers and are NOT thread-safe, and
    preprocessing runs in parallel (tf.data num_parallel_calls). A shared
    instance would produce subtle, hard-to-reproduce corruption under
    concurrency. Per-call construction is cheap relative to the pipeline.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    if enhanced.dtype != np.uint8:
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return enhanced


def _remove_pectoral(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Suppress the pectoral muscle region in MLO views.

    Strategy: the pectoral muscle is a bright triangle in the upper corner.
    We detect it via edge-based line fitting and zero out that region.
    This is a heuristic — it degrades gracefully (does nothing) on CC views
    where no pectoral triangle is present.
    """
    h, w = gray.shape

    # Only attempt removal in the upper 40% of the image where pectoral appears.
    roi_h = int(h * 0.4)
    roi = gray[:roi_h, :]

    edges = cv2.Canny(roi, 30, 100)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=50, minLineLength=roi_h // 4, maxLineGap=20
    )

    result = gray.copy()
    if lines is None:
        return result

    # Find the line most consistent with a pectoral edge (steep diagonal).
    pectoral_line = _find_pectoral_line(lines, roi_h, w)
    if pectoral_line is None:
        return result

    # Build a mask for the pectoral triangle and zero it out.
    x1, y1, x2, y2 = pectoral_line
    pec_mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[0, 0], [x2, y2], [x1, y1], [0, y1]], dtype=np.int32)
    cv2.fillPoly(pec_mask, [pts], 255)

    # False-positive guard: a real pectoral triangle is a corner wedge. If the
    # detected region covers an implausibly large fraction of the image, the
    # line is more likely an artifact (e.g. an MLO scan line) and zeroing it
    # would silently destroy real tissue. Skip removal rather than risk that.
    pec_fraction = (pec_mask > 0).sum() / pec_mask.size
    if pec_fraction > 0.25:
        return result

    result[pec_mask > 0] = 0
    return result


def _find_pectoral_line(lines, roi_h: int, w: int):
    """Select the line most likely to be the pectoral muscle edge.

    Criteria: steep negative slope (upper-left to lower-right), starts near
    the top edge, and spans a meaningful length.
    """
    best = None
    best_score = -1

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)

        if dx == 0:
            continue
        slope = dy / dx

        # Pectoral edge has a positive slope (going down-right from top-left).
        if slope <= 0.3 or slope > 5.0:
            continue

        # Should start near the top of the image.
        top_y = min(y1, y2)
        if top_y > roi_h * 0.3:
            continue

        score = length
        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)

    return best


def _normalize_orientation(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flip the image so the breast always faces right.

    Strategy: find the centroid of the breast mask. If it's in the left half,
    flip horizontally. A fixed reference frame helps downstream models that
    rely on positional embeddings.
    """
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return gray  # Cannot determine orientation — leave as-is.

    cx = moments["m10"] / moments["m00"]
    if cx < gray.shape[1] / 2:
        return cv2.flip(gray, 1)
    return gray

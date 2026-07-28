"""Kotlin-facing entry points for the Second Look Python pipeline.

The app runs the *actual* research preprocessing code on-device via Chaquopy,
rather than a hand-ported copy, so what the model sees at inference time is
byte-for-byte what it saw during training. Everything below is a thin adapter:
no image logic lives here.

Chaquopy type mapping used at this boundary:
    Java String  <-> Python str
    Python bytes  -> Java byte[]   (via PyObject.toJava(byte[]::class))
    Python float  -> Java double
"""

import cv2
import numpy as np

from config.constants import INPUT_SIZE
from data_pipeline.label_mapper import confidence_to_tier
from data_pipeline.preprocessor import load_image, preprocess

# Degrees -> the cv2 rotate code that undoes them. Android hands us the EXIF
# rotation because cv2.imread(IMREAD_UNCHANGED) deliberately ignores EXIF, and
# a sideways mammogram would break the pectoral/orientation heuristics.
_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def input_size() -> int:
    """Model input edge length, from the shared config (INPUT_SIZE is square)."""
    return int(INPUT_SIZE[0])


def preprocess_file(path: str, rotation: int = 0) -> bytes:
    """Preprocess an image file into the model's input tensor.

    Args:
        path: Absolute path to the image the user picked or captured.
        rotation: Clockwise EXIF rotation in degrees (0, 90, 180 or 270).

    Returns:
        Raw little-endian float32 bytes for a [1, INPUT_SIZE, INPUT_SIZE, 1]
        tensor, values in [0, 1].

    Raises:
        FileNotFoundError / ValueError: propagated from the pipeline when the
        image is missing, corrupt, or in an unsupported format.
    """
    image = load_image(path)

    rotate_code = _ROTATIONS.get(rotation % 360)
    if rotate_code is not None:
        image = cv2.rotate(image, rotate_code)

    tensor = preprocess(image, target_size=INPUT_SIZE)
    return np.ascontiguousarray(tensor, dtype="<f4").tobytes()


def tier_for(probability: float) -> str:
    """Concern tier for a positive-class probability ('Low'/'Moderate'/'Elevated').

    Delegates to the shared cut-points so the app never carries its own copy of
    them — they are provisional and will change when calibration lands.
    """
    return confidence_to_tier(float(probability))

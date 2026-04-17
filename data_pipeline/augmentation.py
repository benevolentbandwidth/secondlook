# Real-world capture augmentations for Second Look.
#
# These simulate the degradations a phone-camera photo of a printed or
# on-screen mammogram would introduce — glare, motion blur, JPEG compression,
# moiré, brightness/contrast drift, partial occlusion, perspective skew,
# and rotation. Training against them is what makes on-device capture
# tolerable for the downstream classifier.
#
# Function bodies are kept intact from the original Colab script; only
# the Colab-specific scaffolding (pip install, files.upload, matplotlib
# grid, zip/download, hard-coded /content paths, module-level RNG seeding)
# has been removed. Masks propagate through geometric augs so segmentation
# labels stay aligned.
#
# Reproducibility: call seed_rng(seed) explicitly when you need deterministic
# output. Module import no longer mutates the global RNG.

import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def seed_rng(seed: int = 42) -> None:
    """Seed both `random` and `numpy.random` for reproducible augmentation.

    Explicit opt-in — the module no longer seeds on import, which would
    silently clobber any caller-established RNG state.
    """
    random.seed(seed)
    np.random.seed(seed)


def add_glare(img, mask=None):
    # Glare affects only intensity, so segmentation masks remain valid
    out = img.copy().astype(np.float32)
    h, w = out.shape[:2]
    cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
    rx, ry = random.randint(w//10, w//4), random.randint(h//10, h//4)
    glare_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(glare_mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    glare_mask = cv2.GaussianBlur(glare_mask, (0, 0), sigmaX=max(rx//2, 1))

    # Scale glare based on brightness so darker images are not overexposed
    intensity = 0.3 + (0.3 * (np.mean(out) / 255.0))
    out += glare_mask * 255 * intensity

    out = np.clip(out, 0, 255).astype(np.uint8)
    return (out, mask) if mask is not None else out


def add_motion_blur(img, mask=None):
    # Simulates real motion artifacts without shifting object positions
    size = 15
    angle = random.uniform(0, 180)
    kernel = np.zeros((size, size))
    kernel[size//2, :] = 1.0 / size
    M = cv2.getRotationMatrix2D((size//2, size//2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    out = cv2.filter2D(img, -1, kernel)
    return (out, mask) if mask is not None else out


def add_compression(img, mask=None):
    # Mimics real-world quality loss from storage/transmission pipelines
    quality = random.randint(10, 35)
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return (img.copy(), mask) if mask is not None else img.copy()
    out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return (out, mask) if mask is not None else out


def add_moire(img, mask=None):
    # Adds frequency-based noise pattern seen in scanning/display artifacts
    h, w = img.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    wave = np.sin(2 * np.pi * 0.08 * (X * np.cos(0.5) + Y * np.sin(0.5)))
    wave = ((wave + 1) / 2 * 25).astype(np.float32)
    out = np.clip(img.astype(np.float32) + wave, 0, 255).astype(np.uint8)
    return (out, mask) if mask is not None else out


def add_brightness(img, mask=None):
    # Tests robustness under poor exposure (common in medical imaging)
    factor = random.uniform(0.25, 0.65)
    out = np.array(ImageEnhance.Brightness(Image.fromarray(img)).enhance(factor)).astype(np.uint8)
    return (out, mask) if mask is not None else out


def add_contrast(img, mask=None):
    # Makes subtle structures harder to detect to stress-test models
    factor = random.uniform(0.25, 0.60)
    out = np.array(ImageEnhance.Contrast(Image.fromarray(img)).enhance(factor)).astype(np.uint8)
    return (out, mask) if mask is not None else out


def add_occlusion(img, mask=None):
    # Simulates partial visibility while keeping labels usable
    out = img.copy().astype(np.float32)
    h, w = out.shape[:2]
    occ_mask = np.zeros((h, w), dtype=np.float32)
    rx, ry = random.randint(w//6, w//3), random.randint(h//8, h//5)
    cx, cy = random.randint(0, rx), random.randint(0, ry)
    cv2.ellipse(occ_mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    occ_mask = cv2.GaussianBlur(occ_mask, (0, 0), sigmaX=20)
    out = np.clip(out * (1 - occ_mask), 0, 255).astype(np.uint8)
    return (out, mask) if mask is not None else out


def add_perspective(img, mask=None):
    # Must apply identical transform to maintain image–mask alignment
    h, w = img.shape[:2]
    j = int(min(h, w)*0.08)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, j),     random.randint(0, j)],
        [w-random.randint(0, j),   random.randint(0, j)],
        [w-random.randint(0, j),   h-random.randint(0, j)],
        [random.randint(0, j),     h-random.randint(0, j)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if mask is not None:
        mask_out = cv2.warpPerspective(mask, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return out, mask_out
    return out


def add_rotation(img, mask=None):
    # Keeps annotation consistent by rotating image and mask together
    h, w = img.shape[:2]
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if mask is not None:
        mask_out = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return out, mask_out
    return out


def normalize(img):
    # Enforces a consistent format to prevent downstream pipeline errors
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.clip(img, 0, 255).astype(np.uint8)


# Central registry simplifies random selection and extensibility.
AUGMENTATIONS = {
    'glare': add_glare,
    'motion_blur': add_motion_blur,
    'compression': add_compression,
    'moire': add_moire,
    'brightness': add_brightness,
    'contrast': add_contrast,
    'occlusion': add_occlusion,
    'perspective': add_perspective,
    'rotation': add_rotation,
}


def apply_random_combo(img, mask=None):
    # Limited stacking preserves realism while increasing variability
    selected = random.sample(list(AUGMENTATIONS.items()), k=random.randint(1, 3))
    out = img.copy()
    out_mask = mask
    names = []
    for name, fn in selected:
        result = fn(out, out_mask)
        if isinstance(result, tuple):
            out, out_mask = result
        else:
            out = result
        out = normalize(out)
        names.append(name)
    if mask is not None:
        return out, out_mask, '+'.join(names)
    return out, '+'.join(names)

"""Tests for the PNG loader path used by the CBIS-DDSM pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from data_pipeline.preprocessor import (
    DEFAULT_SIZE,
    load_and_preprocess,
    load_image,
    preprocess,
)


def _synthetic_breast_png(path, *, dtype=np.uint16, size=(512, 512)):
    """Write a synthetic mammogram-like PNG: dark background, bright blob on right."""
    h, w = size
    img = np.zeros((h, w), dtype=dtype)
    max_val = np.iinfo(dtype).max
    yy, xx = np.ogrid[:h, :w]
    blob = (xx - int(w * 0.7)) ** 2 + (yy - h // 2) ** 2 < (h // 4) ** 2
    img[blob] = max_val
    cv2.imwrite(str(path), img)


def test_load_image_preserves_16bit_depth(tmp_path):
    png_path = tmp_path / "case.png"
    _synthetic_breast_png(png_path, dtype=np.uint16)

    arr = load_image(png_path)
    assert arr.dtype == np.uint16
    assert arr.ndim == 2


def test_load_image_handles_8bit(tmp_path):
    png_path = tmp_path / "case8.png"
    _synthetic_breast_png(png_path, dtype=np.uint8)

    arr = load_image(png_path)
    assert arr.dtype == np.uint8


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


def test_load_and_preprocess_produces_model_ready_tensor(tmp_path):
    png_path = tmp_path / "case.png"
    _synthetic_breast_png(png_path, dtype=np.uint16)

    tensor = load_and_preprocess(png_path)

    assert tensor.shape == (*DEFAULT_SIZE, 1)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_preprocess_accepts_grayscale_array_directly():
    arr = (np.random.rand(256, 256) * 255).astype(np.uint8)
    tensor = preprocess(arr)
    assert tensor.shape == (*DEFAULT_SIZE, 1)
    assert tensor.dtype == np.float32

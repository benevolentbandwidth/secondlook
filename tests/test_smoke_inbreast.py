"""End-to-end smoke test on a small INbreast sample.

Gated by the INBREAST_ROOT env var so CI without the dataset skips cleanly.
Expected layout under INBREAST_ROOT:
    birads1/*.png
    birads2/*.png
    birads3/*.png
    birads4/*.png
    birads5/*.png

Scope (stops at preprocessed tensor shape per project decision):
  1. Walk birads{1..5}/*.png, derive BI-RADS from parent dir.
  2. Take a balanced sample of 5 per class (25 total).
  3. Per image: cv2.imread -> quality_check -> preprocess -> shape assertion.
  4. One augmentation -> quality_gate still returns a non-empty verdict string.
  5. Label round-trip: map_dataset("inbreast", n) -> to_int -> expected 0/1.

No model forward pass, no splitter in this smoke — those are separate steps.
"""

import os
import random
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.label_mapper import Label, map_dataset, to_int
from config.constants import INPUT_SIZE


INBREAST_ROOT_ENV = "INBREAST_ROOT"
SAMPLE_PER_CLASS = 5
SEED = 42

# BI-RADS -> expected binary int under the INbreast rule.
EXPECTED_LABEL = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}


def _inbreast_root() -> Path | None:
    raw = os.environ.get(INBREAST_ROOT_ENV)
    if not raw:
        return None
    root = Path(raw)
    return root if root.exists() else None


def _sampled_paths(root: Path) -> list[tuple[int, Path]]:
    rng = random.Random(SEED)
    picked: list[tuple[int, Path]] = []
    for birads in (1, 2, 3, 4, 5):
        class_dir = root / f"birads{birads}"
        if not class_dir.exists():
            pytest.skip(f"Missing class dir: {class_dir}")
        pngs = sorted(class_dir.glob("*.png"))
        if len(pngs) < SAMPLE_PER_CLASS:
            pytest.skip(f"Need >={SAMPLE_PER_CLASS} images in {class_dir}, got {len(pngs)}")
        picked.extend((birads, p) for p in rng.sample(pngs, SAMPLE_PER_CLASS))
    return picked


@pytest.mark.skipif(
    _inbreast_root() is None,
    reason=f"Set {INBREAST_ROOT_ENV} to the INbreast root (containing birads1..birads5/) to run.",
)
def test_inbreast_end_to_end_smoke():
    import cv2

    from data_pipeline.preprocessor import preprocess
    from data_pipeline.quality import quality_check, quality_gate
    from data_pipeline.augmentation import apply_random_combo, seed_rng

    seed_rng(SEED)
    root = _inbreast_root()
    samples = _sampled_paths(root)
    assert len(samples) == SAMPLE_PER_CLASS * 5

    for birads, path in samples:
        # Use np.fromfile + cv2.imdecode so Unicode paths work on Windows
        # (cv2.imread goes through a narrow-char API and returns None for
        # non-ASCII paths like this dataset's "İnbreast" folder).
        raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert raw is not None, f"imdecode returned None for {path}"

        passes, reason = quality_check(raw)
        assert passes, f"quality_check failed on {path}: {reason}"

        tensor = preprocess(raw)
        assert tensor.dtype == np.float32, f"dtype for {path}: {tensor.dtype}"
        assert tensor.shape == (*INPUT_SIZE, 1), f"shape for {path}: {tensor.shape}"
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0, (
            f"range for {path}: [{tensor.min()}, {tensor.max()}]"
        )

        augmented, _combo = apply_random_combo(raw)
        verdict = quality_gate(augmented)
        assert isinstance(verdict, str) and verdict, (
            f"quality_gate returned empty verdict for augmented {path}"
        )

        expected = EXPECTED_LABEL[birads]
        assert to_int(map_dataset("inbreast", birads)) == expected, (
            f"INbreast label mismatch for BI-RADS {birads} ({path})"
        )


def test_inbreast_mapper_roundtrip():
    """INbreast mapping: 1-3 -> NOT_WORTH (0), 4-5 -> WORTH (1). Unknown raises."""
    assert to_int(map_dataset("inbreast", 1)) == 0
    assert to_int(map_dataset("inbreast", 2)) == 0
    assert to_int(map_dataset("inbreast", 3)) == 0
    assert to_int(map_dataset("inbreast", 4)) == 1
    assert to_int(map_dataset("inbreast", 5)) == 1

    assert map_dataset("inbreast", "BI-RADS 2") == Label.NOT_WORTH_SECOND_LOOK
    assert map_dataset("inbreast", "BI-RADS 4") == Label.WORTH_SECOND_LOOK

    with pytest.raises(ValueError):
        map_dataset("inbreast", 0)
    with pytest.raises(ValueError):
        map_dataset("inbreast", 6)
    with pytest.raises(ValueError):
        map_dataset("inbreast", "not a birads")

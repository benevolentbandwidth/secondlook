"""Smoke test for the data pipeline.

What this verifies end-to-end per CLAUDE.md's "done" criteria:
  1. Load one image (via local fixture),
  2. Pass quality_check,
  3. Run preprocess,
  4. (After augmentation-module merge:) apply an augmentation and check
     quality_gate does not fail catastrophically,
  5. Produce a (224, 224, 1) float32 tensor,
  6. Confirm binary label mapping round-trips correctly.

The image step is gated on a local fixture because we do not yet have a
retriever (GCS download + local cache) — that's a separate PR. Without a
fixture image present, the image-dependent assertions are skipped; the
label-mapper round-trip still runs and is the primary thing we can verify
today.
"""

from pathlib import Path

import numpy as np
import pytest

from data_pipeline.label_mapper import (
    Label,
    map_dataset,
    to_int,
    confidence_to_tier,
    display_label,
)


FIXTURE_IMAGE = Path(__file__).parent / "fixtures" / "sample_mammogram.png"


def test_label_mapper_roundtrip():
    """Binary mapper covers all three datasets and preserves ValueError safety."""
    assert to_int(map_dataset("cbis", "MALIGNANT")) == 1
    assert to_int(map_dataset("cbis", "BENIGN_WITHOUT_CALLBACK")) == 0
    assert to_int(map_dataset("rsna", 1)) == 1
    assert to_int(map_dataset("rsna", 0)) == 0
    assert to_int(map_dataset("vindr", "BI-RADS 4")) == 1
    assert to_int(map_dataset("vindr", "BI-RADS 2")) == 0

    # Unknown labels must raise — never silently default.
    with pytest.raises(ValueError):
        map_dataset("cbis", "UNKNOWN_PATHOLOGY")
    with pytest.raises(ValueError):
        map_dataset("rsna", 2)
    with pytest.raises(ValueError):
        map_dataset("mystery_dataset", "anything")


def test_confidence_to_tier_and_display():
    """UX helpers map probability → tier → display string."""
    assert confidence_to_tier(0.1) == "Low"
    assert confidence_to_tier(0.5) == "Moderate"
    assert confidence_to_tier(0.9) == "Elevated"
    assert display_label("Elevated") == "Elevated Area of Interest"

    with pytest.raises(ValueError):
        confidence_to_tier(1.5)
    with pytest.raises(ValueError):
        display_label("NotATier")


def test_label_enum_values():
    """Positive class is 1, matches the training label convention."""
    assert Label.WORTH_SECOND_LOOK.value == 1
    assert Label.NOT_WORTH_SECOND_LOOK.value == 0


@pytest.mark.skipif(
    not FIXTURE_IMAGE.exists(),
    reason=(
        f"No local fixture image at {FIXTURE_IMAGE}. "
        "Drop a sample mammogram there to enable the full image pipeline "
        "smoke test. A retriever that fetches from GCS is a separate PR."
    ),
)
def test_preprocess_pipeline_shape():
    """Image pipeline produces a (224, 224, 1) float32 tensor in [0, 1]."""
    import cv2

    from data_pipeline.preprocessor import preprocess
    from data_pipeline.quality import quality_check
    from config.constants import INPUT_SIZE

    raw = cv2.imread(str(FIXTURE_IMAGE), cv2.IMREAD_UNCHANGED)
    assert raw is not None, f"cv2.imread returned None for {FIXTURE_IMAGE}"

    passes, reason = quality_check(raw)
    assert passes, f"quality_check failed on fixture: {reason}"

    tensor = preprocess(raw)

    assert tensor.dtype == np.float32
    assert tensor.shape == (*INPUT_SIZE, 1)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0

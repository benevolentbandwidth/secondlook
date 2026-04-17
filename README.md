# Second Look

Second Look is a privacy-preserving, on-device mammogram analysis prototype. The target deliverable is a TF Lite model that runs entirely on mobile or in the browser — it flags regions of interest, renders a coarse concern tier derived from model confidence, and stores/transmits nothing. No image or result ever leaves the device.

## Current status

This branch (`integration/data-pipeline`) is the Phase 1 integration: label mapping, preprocessing, input-quality gating, stratified splitting, and a MobileNetV2 baseline classifier (binary head). Not yet implemented in this branch: the GCS retriever + local cache, the `manifest.csv` builder, the `scripts/build_dataset.py` CLI, and the augmentation module (merge pending). Next milestone is end-to-end dataset construction: retrieve from GCS → preprocess → write manifest → train. See `second_look_work_breakdown_structure.docx` for the full 12-week WBS.

## Installation

Python 3.10 or newer. There is no `requirements.txt` yet — install the runtime deps directly:

```bash
git clone <repo-url> b2-secondlook
cd b2-secondlook

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install "tensorflow>=2.15,<2.20" opencv-python pillow scikit-learn pandas numpy
pip install pytest              # for the smoke test
```

TensorFlow pins the upper bound of supported Python versions; if `pip install tensorflow` fails, check that your interpreter is 3.10–3.12.

## Usage

All examples assume the repo root is on `PYTHONPATH` (running from the repo root via `python -m` or `pytest` handles this).

### Label mapping

The mapper normalizes dataset-specific labels (CBIS-DDSM pathologies, RSNA cancer flags, VinDr BI-RADS categories) into the binary `Label` enum that the model trains on. Unknown labels raise `ValueError` rather than defaulting — silent fallbacks on unrecognized input are the second-worst failure mode after false reassurance.

```python
from data_pipeline.label_mapper import (
    map_dataset, to_int, Label,
    confidence_to_tier, display_label,
)

to_int(map_dataset("cbis", "MALIGNANT"))         # -> 1 (WORTH_SECOND_LOOK)
to_int(map_dataset("rsna", 0))                   # -> 0 (NOT_WORTH_SECOND_LOOK)
to_int(map_dataset("vindr", "BI-RADS 4"))        # -> 1

map_dataset("cbis", "UNKNOWN_PATHOLOGY")         # raises ValueError
```

The UX-layer tier helpers are deliberately separate from the model head. The classifier outputs a single positive-class probability; the app renders that probability as a concern tier. The model never "knows" about tiers.

```python
prob = 0.87
display_label(confidence_to_tier(prob))          # -> "Elevated Area of Interest"
display_label(confidence_to_tier(0.2))           # -> "Low Area of Interest"

confidence_to_tier(1.5)                          # raises ValueError
display_label("NotATier")                        # raises ValueError
```

The `confidence_to_tier` thresholds (`<0.33 / <0.66 / else`) are provisional defaults pending calibration — they will almost certainly become asymmetric once real validation data is available.

### Preprocessing

`preprocess()` runs the full pipeline: grayscale → CLAHE → breast masking → pectoral muscle removal → orientation normalization → resize to 224×224 → normalize to `[0, 1]` float32. Output shape is `(H, W, 1)`.

```python
import cv2
from data_pipeline.preprocessor import preprocess

raw = cv2.imread("path/to/mammogram.png", cv2.IMREAD_UNCHANGED)
tensor = preprocess(raw)                         # (224, 224, 1) float32 in [0, 1]
```

### Quality gates

Two distinct gates live in `data_pipeline.quality`:

- `quality_check(image)` — **input gate.** Rejects raw mammograms that cannot be analyzed reliably (blank, too small, low contrast, too little tissue). Returns `(bool, reason_string)`. Run this before preprocessing; surface the reason to the user on failure.
- `quality_gate(image)` — **augmentation-realism gate.** Classifies an augmented image as `USABLE | BORDERLINE | UNUSABLE` so degraded augmentations don't pollute training. Ships with the augmentation module (merge pending).

```python
from data_pipeline.quality import quality_check

passes, reason = quality_check(raw)
if not passes:
    raise RuntimeError(f"Image rejected: {reason}")
```

### Augmentation

Not merged yet.

### Splitting a dataset

Stratified 70/15/15 train/val/test split on the binary `label` column, seed 42 by default. Reproducible across machines.

```python
import pandas as pd
from data_pipeline.splitter import split_dataset, summarize_splits, save_splits

df = pd.read_csv("manifest.csv")                 # columns: image_path, label, ...
train_df, val_df, test_df = split_dataset(df)    # stratified on "label"

print(summarize_splits(train_df, val_df, test_df))
save_splits(train_df, val_df, test_df, output_dir="data/splits/")
```

`split_dataset` raises `ValueError` if either class has fewer than 3 samples (stratification would be undefined) — collect more data rather than disabling the check.

### Training the baseline

MobileNetV2 backbone (ImageNet weights, frozen initially) with a `Dense(1, sigmoid)` head. Grayscale input is projected to 3 channels via a 1×1 Conv rather than a Lambda layer, which keeps the model TF Lite convertible. Class weights are balanced, then the positive class (`WORTH_SECOND_LOOK`) gets an extra 1.5× multiplier — missing a worth-a-second-look case is the worst failure mode, and the weighting reflects that asymmetry.

Real image data is not yet wired in (the retriever and manifest builder are the next milestone), so this example assumes you've already produced split DataFrames with `image_path` and `label` columns.

```python
from modeling.train import train_baseline

history = train_baseline(
    train_df,
    val_df,
    image_dir="data/images/",
    image_col="image_path",
    label_col="label",
    batch_size=32,
    max_epochs=50,
    checkpoint_dir="checkpoints/baseline",
    freeze_backbone=True,
)
```

Best checkpoint (by `val_loss`) is written to `{checkpoint_dir}/best.keras`. Early stopping patience is 7 epochs; `ReduceLROnPlateau` kicks in at patience 3.

### Evaluation

Sensitivity on `WORTH_SECOND_LOOK` is the primary metric — not accuracy. `evaluate_baseline` prints the 2×2 confusion matrix and a per-class sensitivity breakdown, and flags whether the model cleared the `WORTH_SENSITIVITY_FLOOR = 0.80` safety floor. Pass `raise_on_unsafe=True` in CI to hard-fail when the floor is breached.

```python
from modeling.evaluate import evaluate_baseline

results = evaluate_baseline(
    "checkpoints/baseline/best.keras",          # or a live tf.keras.Model
    test_df,
    image_dir="data/images/",
    threshold=0.5,                              # lower favors sensitivity
    raise_on_unsafe=False,
)

print(results["worth_sensitivity"])             # float in [0, 1]
print(results["passed_safety_floor"])           # bool
```

Lowering `threshold` below 0.5 trades specificity for sensitivity — preferable here given the failure-mode asymmetry.

# Modeling Changes

## Branch: feature/modeling-improvements

---

### 1. AUC-based checkpointing (`train.py`)

**What changed:** `ModelCheckpoint` and `EarlyStopping` now monitor `val_auc` (maximise) instead of `val_loss` (minimise). Added `AUC` as a compiled metric. `ReduceLROnPlateau` stays on `val_loss` — this benefits from the smoother, more continuous signal.

**Why:** `val_loss` (binary cross-entropy) rewards confidence, not ranking quality. A model can drive loss down by being very certain about easy negatives while still fumbling borderline WORTH cases. AUC directly measures whether the model ranks WORTH above NOT_WORTH across all thresholds — a much stronger signal when sensitivity is the priority. Checkpointing on best AUC gives us the most flexibility when choosing a decision threshold that meets the safety floor.

---

### 2. Validation-set threshold optimisation (`evaluate.py`)

**What changed:** Added `find_optimal_threshold(model, val_df, image_dir, ...)` to `evaluate.py`, exported from `__init__.py`. The 0.5 default in `evaluate_baseline` is intentionally untouched — callers are expected to run `find_optimal_threshold` on the validation set first and pass the result in explicitly.

**Why:** The sigmoid output is a ranking score, not a calibrated probability. Defaulting to 0.5 ignores class imbalance and the asymmetric cost of missing a WORTH case. Instead, we sweep the ROC curve on the validation set to find the lowest threshold where sensitivity meets `WORTH_SENSITIVITY_FLOOR` (0.80) and specificity is as high as possible — catching every necessary case while keeping patient callbacks to a minimum. The test set is never touched during this step.

**Usage:**

```python
threshold = find_optimal_threshold(model, val_df, image_dir="data/images/")
results = evaluate_baseline(model, test_df, image_dir="data/images/", threshold=threshold)
```

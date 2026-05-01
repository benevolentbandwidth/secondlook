# Second Look

Second Look is a privacy-preserving, on-device mammogram analysis prototype. This branch (`integration/data-pipeline`) integrates the Phase 1 pipeline: a binary label mapper (CBIS-DDSM, RSNA, VinDr, INbreast → `WORTH_SECOND_LOOK` / `NOT_WORTH_SECOND_LOOK`) with UX tier helpers, a preprocessor (CLAHE, breast masking, pectoral removal, orientation, 224×224 float32 output), an input `quality_check` gate, a stratified 70/15/15 splitter, and a MobileNetV2 baseline classifier with a sigmoid head, binary-crossentropy training, and sensitivity-floor evaluation. Tests cover label round-trips, a fixture-gated preprocessing smoke test, and an end-to-end smoke test on a sampled INbreast subset. Not yet merged: augmentation module. Not yet written: GCS retriever, manifest builder, `build_dataset` CLI. See `second_look_work_breakdown_structure.docx`.

## Setup

Requires Python 3.9–3.13. TensorFlow's deeply-nested wheels need Windows long paths enabled (one-time, elevated PowerShell):

```
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Then:

```
py -3.13 -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Testing a dataset

Smoke tests are gated on an env var pointing at the dataset root. This keeps CI green without the data and lets contributors run the full pipeline locally.

**INbreast** — expected layout: `<root>/birads{1..5}/*.png`. Samples 5 images per class (25 total), runs `quality_check` → `preprocess` → shape assert → augmentation → `quality_gate` → label round-trip.

Change according to where you have your data stored, but example is:
```
INBREAST_ROOT="../../data/INbreast dataset - BI-RADS classification/İnbreast" \
  pytest tests/test_smoke_inbreast.py -v
``` 

Runtime ~4 min (augmentation runs on full-res images). Expect `2 passed`.

**Generic fixture** — drop a single mammogram at `tests/fixtures/sample_mammogram.png` and run:

```
pytest tests/test_smoke.py -v
```

**All tests** — `pytest tests/ -v`. Anything without its env var or fixture skips cleanly.

## Adding a new dataset

1. Add a mapper function in `data_pipeline/label_mapper.py` and register it in `map_dataset`. Unknown inputs must raise `ValueError` — never default silently.
2. Add a `tests/test_smoke_<dataset>.py` modeled on [tests/test_smoke_inbreast.py](tests/test_smoke_inbreast.py), gated on a `<DATASET>_ROOT` env var.
3. On Windows, use `cv2.imdecode(np.fromfile(path, dtype=np.uint8), ...)` instead of `cv2.imread` if the path may contain non-ASCII characters.

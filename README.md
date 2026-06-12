# Second Look

Second Look is a privacy-preserving, on-device mammogram analysis prototype. This branch (`integration/data-pipeline`) integrates the Phase 1 pipeline: a binary label mapper (CBIS-DDSM, RSNA, VinDr, INbreast to `WORTH_SECOND_LOOK` / `NOT_WORTH_SECOND_LOOK`) with UX tier helpers, a preprocessor (CLAHE, breast masking, pectoral removal, orientation, 224x224 float32 output), an input `quality_check` gate, a stratified splitter (CBIS's official train/test boundary, or a patient-grouped 70/15/15 for RSNA), GCS retrievers for CBIS-DDSM, RSNA, and VinDr-Mammo, and a MobileNetV2 baseline classifier with a sigmoid head, binary-crossentropy training, and sensitivity-floor evaluation. **Trainable datasets:** CBIS-DDSM, RSNA Screening Mammography, and VinDr-Mammo are all wired end-to-end through `scripts/build_dataset.py` and produce a unified `data/manifest.csv` consumed by `modeling/train.py`. Tests cover label round-trips, a fixture-gated preprocessing smoke test, mocked-GCS retriever tests, and an end-to-end smoke test on a sampled INbreast subset. See `second_look_work_breakdown_structure.docx`.

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

## Getting the CBIS-DDSM dataset

The data lives in `gs://b2-foundation/second-look/DDSM/`. The bucket holds the four CBIS case description CSVs at the top level and the converted PNG images under `cbis_ddsm/`. You need read access to the bucket.

### One-time setup

Install and initialize the gcloud SDK, then authenticate for Python:

```bash
gcloud auth application-default login
```

This is separate from `gcloud auth login`. The first one writes the credentials file that the Python `google-cloud-storage` library reads from. If you only run `gcloud auth login`, the build script will fail with a `DefaultCredentialsError`.

To silence the quota project warning (optional):

```bash
gcloud auth application-default set-quota-project <your-gcp-project-id>
```

### Building the dataset

The build script downloads the four case CSVs, joins them with the image folder structure in GCS, fetches the PNGs to a local cache, applies the official CBIS train/test split (val is carved out of train at 15%), and writes a training manifest.

Full build (about 10 GB on disk, takes 20 to 40 minutes depending on bandwidth):

```bash
python scripts/build_dataset.py --use-gcs --datasets cbis --cache-dir data/cache
```

Outputs:
- `data/manifest.csv` is the training manifest, one row per case folder, with `split`, `canonical_label`, and `image_local_path` columns.
- `data/manifest_patients.csv` is a patient-level audit view.
- `data/image_download_report.csv` lists the outcome of every download.

The cache is keyed by case folder, so re-running the script is cheap. Already-downloaded PNGs are skipped.

### Small local test before the full pull

If you want to confirm everything works before committing to the full download, run with a row cap:

```bash
python scripts/build_dataset.py --use-gcs --datasets cbis --cache-dir data/cache --limit 100
```

That pulls about 100 cases (a few hundred MB) sampled across the train and test splits. Useful for verifying auth, paths, and downstream training code before pulling the full dataset.

### Running on Vertex AI instead of your laptop

Training the full model on a laptop is fine for smoke tests but slow for real runs. The bucket lives in GCP, so the natural place to train is a Vertex AI Workbench notebook or a custom training job in the same region. Intra-region reads from GCS are free and fast, and you avoid storing the dataset locally. Point `--cache-dir` at a local path on the VM (or skip caching entirely and stream from `gs://` URIs through `tf.io.gfile`) and the rest of the pipeline works the same.

This will be implemented soon.

## Getting the RSNA dataset

RSNA Screening Mammography Breast Cancer Detection (Kaggle 2022) lives at `gs://b2-foundation/second-look/RSNA/rsna-breast-cancer-detection/`. Layout:

- `train.csv` — labels (`cancer` 0/1) and metadata for 54,706 images across 11,913 patients.
- `train_images/{patient_id}/{image_id}.png` — pre-converted PNGs.
- `test.csv` and `test_images/` — Kaggle's hidden competition split, no labels. Ignored by the build script.

Because the test labels are unavailable, the build uses a patient-grouped, stratified 70/15/15 split on the labeled train set. All images from one patient (typically 4: L-CC, L-MLO, R-CC, R-MLO) stay in the same partition, so no patient leaks across train/val/test. RSNA labels are per-breast — a patient with unilateral cancer contributes both a positive and a negative image, both landing in the same split. The auth setup is identical to CBIS (ADC).

Full RSNA build:

```bash
python scripts/build_dataset.py --use-gcs --datasets rsna --cache-dir data/cache
```

Small smoke test (~100 images):

```bash
python scripts/build_dataset.py --use-gcs --datasets rsna --cache-dir data/cache --limit 100
```

All datasets at once:

```bash
python scripts/build_dataset.py --use-gcs --datasets cbis rsna vindr --cache-dir data/cache
```

The image manifest gains an `image_id` column (CBIS rows leave it empty; RSNA and VinDr fill it with the PNG filename stem). The download report is split per-dataset: `image_download_report_cbis.csv`, `image_download_report_rsna.csv`, `image_download_report_vindr.csv`.

## Getting the VinDr-Mammo dataset

VinDr-Mammo (PhysioNet 1.0.0) lives at `gs://b2-foundation/second-look/VinDR/physionet.org/files/vindr-mammo/1.0.0/`. The bucket already holds pre-converted PNGs, so no DICOM dependency is needed. Layout:

- `breast-level_annotations.csv` — labels (`breast_birads` strings like `"BI-RADS 4"`) and metadata for 20,000 images across 5,000 studies (one study per patient, four images per study: L-CC, L-MLO, R-CC, R-MLO). Also publishes an official `split` column with values `training` and `test`.
- `images/{study_id}/{image_id}.png` — pre-converted PNGs.
- `finding_annotations.csv` — finding-level bounding boxes. Not consumed by the binary pipeline; the build script reads `breast-level_annotations.csv` only.

**Label mapping.** `breast_birads` values span `BI-RADS 1` through `BI-RADS 5` (no 0, no 6). The retriever parses the string into the integer N before mapping. BI-RADS 4 and 5 map to `WORTH_SECOND_LOOK` (1); BI-RADS 1–3 map to `NOT_WORTH_SECOND_LOOK` (0). Configured in `config/label_maps.yaml`.

**Split.** VinDr's official train/test boundary is honored via `official_split_train_val`. Val is carved out of train at 15% **grouped by `patient_id`** (= `study_id`) so a single study's four images cannot straddle the train/val line. Empirically, every VinDr study has uniform BI-RADS across all four images (L/R and CC/MLO match), so the per-breast / per-image distinction does not introduce asymmetric labels the way RSNA does. The auth setup is identical to CBIS (ADC).

Full VinDr build:

```bash
python scripts/build_dataset.py --use-gcs --datasets vindr --cache-dir data/cache
```

Small smoke test (~100 images):

```bash
python scripts/build_dataset.py --use-gcs --datasets vindr --cache-dir data/cache --limit 100
```

### Useful flags

- `--no-official-split` falls back to a stratified 70/15/15 split if you do not want CBIS's canonical split.
- `--skip-image-download` builds the manifest without pulling images. Handy for inspecting splits and label distributions.
- `--max-workers N` controls the concurrent download pool (default 8).
- `--dry-run` prints the resolved source configuration and exits.

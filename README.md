# Second Look

Second Look is a privacy-preserving, on-device mammogram analysis prototype. This branch (`integration/data-pipeline`) integrates the Phase 1 pipeline: a binary label mapper (CBIS-DDSM, RSNA, VinDr, INbreast to `WORTH_SECOND_LOOK` / `NOT_WORTH_SECOND_LOOK`) with UX tier helpers, a preprocessor (CLAHE, breast masking, pectoral removal, orientation, 224x224 float32 output), an input `quality_check` gate, a stratified splitter that honors CBIS's official train/test boundary, a GCS retriever for CBIS-DDSM, and a MobileNetV2 baseline classifier with a sigmoid head, binary-crossentropy training, and sensitivity-floor evaluation. Tests cover label round-trips, a fixture-gated preprocessing smoke test, mocked-GCS retriever tests, and an end-to-end smoke test on a sampled INbreast subset. See `second_look_work_breakdown_structure.docx`.

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

### Useful flags

- `--no-official-split` falls back to a stratified 70/15/15 split if you do not want CBIS's canonical split.
- `--skip-image-download` builds the manifest without pulling images. Handy for inspecting splits and label distributions.
- `--max-workers N` controls the concurrent download pool (default 8).
- `--dry-run` prints the resolved source configuration and exits.

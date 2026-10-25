# Second Look — Developer Instructions

Operational guide for setting up the environment, building the datasets, running
the tests, and training on Vertex AI. For a project overview and results, see the
[README](README.md).

**Contents**

1. [Setup](#setup)
2. [Running the tests](#running-the-tests)
3. [GCS authentication](#gcs-authentication)
4. [Building the datasets](#building-the-datasets) — [CBIS-DDSM](#cbis-ddsm) · [RSNA](#rsna) · [VinDr-Mammo](#vindr-mammo) · [all at once](#all-datasets-at-once) · [flags](#useful-build-flags)
5. [Training on Vertex AI](#training-on-vertex-ai)
6. [Running a sweep](#running-a-sweep)
7. [Evaluation protocol](#evaluation-protocol)

---

## Setup

Requires Python 3.9–3.13. On Windows, TensorFlow's deeply nested wheels need long
paths enabled (one time, elevated PowerShell):

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Then:

```bash
py -3.13 -m venv .venv
.venv/Scripts/activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements-train.txt` holds the pinned set the Vertex training package installs.

---

## Running the tests

```bash
pytest tests/ -v
```

Tests that need data are gated on an env var or fixture and **skip cleanly**
without it, so the suite stays green on a fresh clone.

**INbreast smoke test.** Expected layout: `<root>/birads{1..5}/*.png`. Samples 5
images per class (25 total) and runs `quality_check` → `preprocess` → shape
assert → augmentation → `quality_gate` → label round-trip. Point the env var at
wherever your copy lives:

```bash
INBREAST_ROOT="../../data/INbreast dataset - BI-RADS classification/İnbreast" \
  pytest tests/test_smoke_inbreast.py -v
```

Runtime is about 4 minutes (augmentation runs on full-resolution images). Expect `2 passed`.

**Generic fixture.** Drop a single mammogram at
`tests/fixtures/sample_mammogram.png` and run:

```bash
pytest tests/test_smoke.py -v
```

---

## GCS authentication

All datasets live in the access-controlled `gs://b2-foundation` bucket. You need
read access to it.

Install and initialize the gcloud SDK, then authenticate **for Python**:

```bash
gcloud auth application-default login
```

This is separate from `gcloud auth login`. The command above writes the
Application Default Credentials (ADC) file that the `google-cloud-storage`
library reads. If you only run `gcloud auth login`, the build script fails with
`DefaultCredentialsError`. The ADC identity must be an account with bucket
access; the wrong identity fails with a 403 on `storage.objects.list`.

Optional, to silence the quota-project warning:

```bash
gcloud auth application-default set-quota-project <your-gcp-project-id>
```

**Windows only:** TensorFlow's `tf.io.gfile` does not auto-discover ADC. To write
checkpoints to `gs://` from a local machine, set `GOOGLE_APPLICATION_CREDENTIALS`
to the ADC JSON path, or the write authenticates as "Anonymous caller" (401).

> Never commit credential files. `.gitignore` already excludes `*.json`
> credentials and `.env` files.

---

## Building the datasets

`scripts/build_dataset.py` downloads the metadata CSVs, joins them with the image
folders in GCS, fetches PNGs to a local cache, assigns patient-disjoint splits,
and writes a unified training manifest. The cache is skip-if-present, so re-runs
are cheap.

Outputs (under `data/`):

- `manifest.csv` — the training manifest: one row per image with `split`,
  `canonical_label`, `image_local_path`, and `image_id`. This is the **single
  source of truth** for split assignment; the pipeline never re-splits.
- `manifest_patients.csv` — patient-level audit view.
- `image_download_report_<dataset>.csv` — outcome of every download, per dataset.

### CBIS-DDSM

Lives at `gs://b2-foundation/second-look/DDSM/`: the four CBIS case-description
CSVs at the top level, converted PNGs under `cbis_ddsm/`. The build honors CBIS's
official train/test boundary and carves validation out of train at 15%.

Full build (about 10 GB on disk, 20–40 minutes depending on bandwidth):

```bash
python scripts/build_dataset.py --use-gcs --datasets cbis --cache-dir data/cache
```

Small test first (about 100 cases, a few hundred MB, sampled across splits),
useful for verifying auth and paths:

```bash
python scripts/build_dataset.py --use-gcs --datasets cbis --cache-dir data/cache --limit 100
```

### RSNA

RSNA Screening Mammography Breast Cancer Detection (Kaggle 2022) lives at
`gs://b2-foundation/second-look/RSNA/rsna-breast-cancer-detection/`:

- `train.csv` — labels (`cancer` 0/1) and metadata for 54,706 images across 11,913 patients.
- `train_images/{patient_id}/{image_id}.png` — pre-converted PNGs.
- `test.csv` / `test_images/` — Kaggle's hidden split with no labels; ignored.

Because test labels are unavailable, the build uses a **patient-grouped,
stratified 70/15/15 split** on the labeled set. All of a patient's images
(typically L-CC, L-MLO, R-CC, R-MLO) stay in one partition. RSNA labels are
per-breast, so a patient with unilateral cancer contributes both positive and
negative images, all in the same split.

```bash
python scripts/build_dataset.py --use-gcs --datasets rsna --cache-dir data/cache             # full (~120 GB)
python scripts/build_dataset.py --use-gcs --datasets rsna --cache-dir data/cache --limit 100 # smoke
```

### VinDr-Mammo

VinDr-Mammo (PhysioNet 1.0.0) lives at
`gs://b2-foundation/second-look/VinDR/physionet.org/files/vindr-mammo/1.0.0/`.
The bucket holds pre-converted PNGs, so no DICOM dependency is needed.

- `breast-level_annotations.csv` — labels (`breast_birads`, e.g. `"BI-RADS 4"`)
  for 20,000 images across 5,000 studies (one study per patient, four views each),
  plus an official `split` column (`training` / `test`).
- `images/{study_id}/{image_id}.png` — pre-converted PNGs.
- `finding_annotations.csv` — finding-level bounding boxes. Not yet consumed by
  the binary pipeline.

**Labels.** BI-RADS 4–5 → `WORTH_SECOND_LOOK` (1); BI-RADS 1–3 →
`NOT_WORTH_SECOND_LOOK` (0). The retriever parses the string to an integer
before mapping. Configured in `config/label_maps.yaml`.

**Split.** The official train/test boundary is honored; validation is carved out
of train at 15%, **grouped by `study_id`** so a study's four images never
straddle train/val.

```bash
python scripts/build_dataset.py --use-gcs --datasets vindr --cache-dir data/cache             # full (~48 GB)
python scripts/build_dataset.py --use-gcs --datasets vindr --cache-dir data/cache --limit 100 # smoke
```

### All datasets at once

```bash
python scripts/build_dataset.py --use-gcs --datasets cbis rsna vindr --cache-dir data/cache
```

The full combined pull is roughly 178 GB. Note that `--limit` applies **per
dataset** and stratified-samples, so a limited build's label balance is not
representative of the real ~6%-positive distribution (see the README's results
section for why that matters).

### Useful build flags

| Flag | Effect |
|---|---|
| `--limit N` | Sample N rows per dataset for a quick build. |
| `--no-official-split` | Use a stratified 70/15/15 split instead of the dataset's official boundary. |
| `--skip-image-download` | Build the manifest without pulling images (inspect splits and label balance). |
| `--max-workers N` | Concurrent download pool size (default 8). |
| `--dry-run` | Print the resolved source configuration and exit. |

---

## Training on Vertex AI

Training at full scale on a laptop is impractical. The bucket is regional in
**`us-east1`**, so run Vertex AI Custom Jobs **in that region** — intra-region
reads are free and fast. `scripts/train_vertex.py` runs the whole loop on the VM:
build → train → upload the best checkpoint to `gs://` → optional eval.

**Fixed facts**

| Thing | Value |
|---|---|
| Project | `b2-second-look` |
| Region | `us-east1` (matches the `gs://b2-foundation` bucket) |
| Training service account | `vertex-training-b2-second-look@b2-second-look.iam.gserviceaccount.com` (reads the bucket; writes only under `.../checkpoints/`) |
| Checkpoints prefix | `gs://b2-foundation/second-look/checkpoints/` (the only SA-writable location) |
| Container (CPU) | `us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest` |
| Container (GPU) | `us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-17.py310:latest` |

On Vertex the attached service account is the credential: **no key file, and do
not set `GOOGLE_APPLICATION_CREDENTIALS`** (that is a local-Windows-only quirk).

**Prerequisite: `actAs` on the service account.** A `b2-second-look` admin must
grant the submitting user this once; users cannot self-grant:

```bash
gcloud iam service-accounts add-iam-policy-binding \
  vertex-training-b2-second-look@b2-second-look.iam.gserviceaccount.com \
  --member="user:<your-email>" \
  --role="roles/iam.serviceAccountUser" \
  --project=b2-second-look
```

Without it, `custom-jobs create` fails with *"You do not have permission to act
as service_account …"*.

### 1. Package and stage the code

No Docker build: a prebuilt TF container runs the package. Re-run whenever the
code changes:

```bash
python setup.py sdist --dist-dir dist
gsutil cp dist/second_look_training-0.1.0.tar.gz \
  gs://b2-foundation/second-look/vertex-staging/packages/second_look_training-0.1.0.tar.gz
```

### 2. Smoke run (CPU, ~200 CBIS cases, 1 epoch)

Validates the whole path and confirms a checkpoint lands in GCS. Use `--limit 200`
or more: CBIS is ~87% positive, so a smaller sample can leave a split with fewer
than 3 negatives and the splitter (correctly) refuses to stratify. Expect a poor
model; the smoke run tests the pipeline, not model quality.

```bash
gcloud ai custom-jobs create \
  --project=b2-second-look --region=us-east1 \
  --display-name="second-look-smoke-$(date +%Y%m%d-%H%M%S)" \
  --service-account=vertex-training-b2-second-look@b2-second-look.iam.gserviceaccount.com \
  --python-package-uris=gs://b2-foundation/second-look/vertex-staging/packages/second_look_training-0.1.0.tar.gz \
  --worker-pool-spec=machine-type=e2-standard-4,replica-count=1,executor-image-uri=us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-17.py310:latest,python-module=scripts.train_vertex \
  --args=--datasets=cbis,--limit=200,--max-epochs=1,--checkpoint-dir=gs://b2-foundation/second-look/checkpoints/smoke-vertex,--run-eval

gsutil ls -l gs://b2-foundation/second-look/checkpoints/smoke-vertex/   # confirm best.keras
```

> **PowerShell caveat:** unquoted commas are PowerShell's array operator and will
> corrupt `--worker-pool-spec` / `--args`. Put those values in variables and quote
> them (`"--args=$jobArgs"`), or run from Git Bash / Cloud Shell as written.

### 3. Single-dataset GPU run (full CBIS)

Drop `--limit`, raise epochs, and use a GPU pool. GPUs need an **N1** machine
type (E2 cannot attach accelerators). Full CBIS (~10 GB) fits on the default
disk. Add `--no-freeze-backbone` to `--args` to fine-tune the backbone.

```bash
gcloud ai custom-jobs create \
  --project=b2-second-look --region=us-east1 \
  --display-name="second-look-baseline-$(date +%Y%m%d-%H%M%S)" \
  --service-account=vertex-training-b2-second-look@b2-second-look.iam.gserviceaccount.com \
  --python-package-uris=gs://b2-foundation/second-look/vertex-staging/packages/second_look_training-0.1.0.tar.gz \
  --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,executor-image-uri=us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-17.py310:latest,python-module=scripts.train_vertex \
  --args=--datasets=cbis,--max-epochs=12,--checkpoint-dir=gs://b2-foundation/second-look/checkpoints/baseline,--run-eval
```

Combined-dataset runs need a much larger boot disk, which `--worker-pool-spec`
cannot set. Use the YAML configs in [Running a sweep](#running-a-sweep) instead.

### 4. Monitor

```bash
gcloud ai custom-jobs list --project=b2-second-look --region=us-east1 --limit=5
gcloud ai custom-jobs stream-logs <JOB_ID> --project=b2-second-look --region=us-east1
```

---

## Running a sweep

`scripts/sweep_vertex.py` builds the dataset **once**, then trains and evaluates
every config in a named sweep inside the same job. Each config gets its own
`<checkpoint-dir>/<name>/best.keras`, reliability diagram, and
`calibration.json`. A ranked `sweep_summary.csv` is re-uploaded after every
config, so partial results survive a crash, and one failing config never sinks
the run.

Sweeps are defined in the `SWEEPS` dict at the top of `scripts/sweep_vertex.py`
(currently `imbalance`, `seeds`, `resolution`) and picked with `--sweep NAME`.
The docstring there documents every per-config key (`mode`, `loss`,
`focal_gamma`, `use_class_weights`, `max_neg_per_pos`, `seed`, `input_size`, …).

Each job is described by a YAML config under `vertex/`. The YAML is the only way
to set the boot disk size, which the full combined pull needs (500 GB):

| Config | Purpose |
|---|---|
| `vertex/sweep-full-combined.yaml` | Full CBIS + RSNA + VinDr on 1× T4, 500 GB disk |
| `vertex/sweep-subset-imbalance.yaml` | Cheap `--limit` subset sweep to validate code paths |
| `vertex/sweep-seeds.yaml` | Same recipe, three seeds: measures the run-to-run noise floor |
| `vertex/sweep-resolution.yaml` | 224 vs 288 vs 320 input size, paired in one job |

Submit (after staging the sdist; **re-stage after any edit to `SWEEPS`**):

```bash
gcloud ai custom-jobs create \
  --project=b2-second-look --region=us-east1 \
  --display-name="second-look-sweep-$(date +%Y%m%d-%H%M%S)" \
  --service-account=vertex-training-b2-second-look@b2-second-look.iam.gserviceaccount.com \
  --config=vertex/sweep-full-combined.yaml
```

Each YAML header has its own submit command, expected cost, and what to read in
the output.

**Rules for reading a sweep:**

- Put the control **in the same job** as the treatment. Never compare against a
  number from an earlier run; identical recipes have differed by 0.03 AUROC
  between runs.
- Precision and specificity depend on prevalence, so a `--limit` subset cannot
  estimate them. Decide anything aimed at the false-alarm rate on a full run.
- Judge class-imbalance levers on `op_specificity` / `op_precision`, not AUROC;
  reweighting moves the boundary without re-ranking.

---

## Evaluation protocol

`modeling/evaluate.py` reports, in order:

1. **AUROC**: threshold-independent, the honest headline on imbalanced data.
2. **Macro-F1** plus per-class F1: catches a model that collapses to one class.
3. **Operating point**: the threshold that maximizes specificity subject to
   `WORTH_SECOND_LOOK` sensitivity ≥ 0.80, with specificity, precision, and
   WORTH-F1 at that threshold.
4. **Calibration**: Brier score and ECE, plus a reliability diagram when
   `output_dir` is set. `modeling/calibrate.py` fits temperature scaling on
   validation and applies it to test.

It still enforces the 0.80 sensitivity floor (`WORTH_SENSITIVITY_FLOOR` in
`modeling/baseline_classifier.py`) and prints the confusion matrix at a
reference threshold.

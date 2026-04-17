# Second Look — Project Context for Claude Code

## What this project is

Second Look is a privacy-preserving, on-device mammogram analysis prototype.
The 12-week goal is a TF Lite model that runs entirely on-device (mobile or
browser), flags regions of interest, displays a coarse concern tier, and
stores/transmits nothing. See `second_look_work_breakdown_structure.docx`
in the project root for the full WBS.

## Current phase

Phase 1: Data pipeline. Three teammates each built a component on separate
branches. This integration branch combines them into a clean, working pipeline.

Branches being integrated:
- `feature/img_retriever` — preprocessor, splitter, a 3-tier label mapper,
  and a full `modeling/` directory (baseline classifier, train, evaluate).
  NOTE: despite the name, this branch does NOT contain an actual image
  retriever. That still needs to be written.
- `label-mapper-module` — a competing label mapper (binary output,
  multi-dataset: CBIS-DDSM + RSNA + VinDr).
- `augmentation-module` — 9 real-world capture augmentations + a quality
  gate, currently trapped inside a Colab demo script.

## Core design decisions (non-negotiable)

**Binary, matches original design:**
The model output is BINARY: `WORTH_SECOND_LOOK` (1) vs `NOT_WORTH_SECOND_LOOK` (0).
The UX-layer concern tiers (Low / Moderate / Elevated) are a separate
presentation concern derived from model confidence, NOT from the model head.
Conflating the two is a design error. Use `label-mapper-module/label_mapper.py`
(binary, multi-dataset) as the source of truth for training labels. Keep the
`display_label` / `TIER_DISPLAY_LABELS` strings from the other mapper as the
UX-layer tier rendering.


### Other fixed decisions
- **Input size:** 224×224 (EfficientNetB0 / MobileNetV2 compatible, TF Lite friendly)
- **Backbone:** MobileNetV2 (already used in `modeling/baseline_classifier.py`).
  This overrides the older EfficientNetB0 plan — MobileNetV2 is a better fit
  for mobile deployment and the team has already built around it.
- **Seed:** 42, fixed everywhere, for reproducibility.
- **Split:** 70 / 15 / 15 train/val/test, stratified by label.
- **Manifest-driven:** A `manifest.csv` is the single source of truth for
  split assignment. The pipeline never re-splits on reconnect. Splitter
  writes the manifest once; everything downstream reads from it.
- **Data source:** Google Cloud Storage. The retriever downloads to a local
  cache directory and must never re-download on every run.
- **CPU-only for dataset construction** (`tf.device('/CPU:0')`). GPU
  operations in Colab have been a persistent source of instability.
  Training itself can use GPU, but tf.data ops stay on CPU.

## Failure mode hierarchy

In decreasing order of severity:
1. **False reassurance** on Elevated / WORTH cases — the worst failure.
   Sensitivity on the positive class is the primary metric. See
   `ELEVATED_SENSITIVITY_FLOOR = 0.80` in `baseline_classifier.py`.
2. **Silent fallbacks** on unknown inputs. Unknown labels must raise
   `ValueError`, never default to a class. This principle is already
   embedded in both label mappers — preserve it.
3. False alarms. Preferred over false reassurance.

## Target directory structure
b2-secondlook/
├── config/
│   └── constants.py           # INPUT_SIZE, SEED, paths, label constants
├── data_pipeline/
│   ├── init.py
│   ├── retriever.py           # NEW — GCS download + local cache
│   ├── label_mapper.py        # unified per decision above
│   ├── preprocessor.py        # from feature/img_retriever, tuned to 224
│   ├── augmentation.py        # refactored from augmentation-module
│   ├── quality.py             # merged quality_check + quality_gate
│   ├── splitter.py            # from feature/img_retriever
│   └── manifest.py            # NEW — builds/loads manifest.csv
├── modeling/
│   ├── init.py
│   ├── baseline_classifier.py # updated for the chosen label schema
│   ├── train.py
│   └── evaluate.py
├── scripts/
│   └── build_dataset.py       # CLI: retrieve → preprocess → manifest
├── demos/
│   └── augmentation_grid.py   # the visualization extracted from Colab
├── tests/
│   └── fixtures/              # the CSV metadata files from label-mapper-module
├── CLAUDE.md
├── README.md
├── requirements.txt
└── .gitignore

## Integration ground rules

- Preserve git authorship. When files are moved from a teammate's branch,
  use `git mv` or at minimum note the original author in the commit message
  so credit is preserved.
- One merge at a time, in this order: `label-mapper-module` first, then
  `feature/img_retriever`, then `augmentation-module`. Commit and verify
  after each.
- Never strip the safety behaviors (ValueError on unknown labels, sensitivity
  floor check, quality gate) when refactoring.
- The augmentation module functions themselves are good — keep the function
  bodies mostly intact. The changes are structural (remove Colab/matplotlib/
  zip code, turn it into a real importable module).
- Do not commit GCS credentials, service account keys, or any `.json`
  credential files. `.gitignore` must cover these.

## What "done" looks like for this integration

A smoke test in `tests/` can: load one image from GCS (or a local fixture),
preprocess it, apply augmentation, pass the quality gate, produce a tensor
of shape `(224, 224, 1)` or `(224, 224, 3)` depending on classifier input,
and confirm label mapping round-trips correctly. No training yet — that's
the next milestone.
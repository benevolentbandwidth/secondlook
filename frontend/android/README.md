# SecondLook (Android)

An Android app that runs on-device triage of mammogram images. A user selects or
captures a mammogram, the image is preprocessed and passed through a bundled
TensorFlow Lite model, and the app returns a probability and a coarse tier
(Low / Moderate / Elevated).

This is the Android counterpart of the iOS app in `frontend/`, with the same
four-screen flow, the same palette (Apple's system colors), and the same model.

## ⚠️ Not a medical device

SecondLook is a research and educational prototype. It is **not** a diagnostic
tool, is **not** FDA/CE cleared, and must not be used to make clinical or
personal health decisions. The tier cut-points are provisional and uncalibrated.
The app shows a disclaimer gate on launch for this reason.

## How it works

1. **Capture / select** an image — `ui/SecondLookFlow.kt` (Compose): disclaimer
   gate → capture → scanning → results.
2. **Preprocess** it by running the research pipeline's **actual Python code**
   on-device through [Chaquopy](https://chaquo.com/chaquopy/): grayscale →
   CLAHE → breast mask → pectoral-muscle removal → orientation normalization →
   resize to 224×224 → float32 in `[0, 1]`.
3. **Classify** with `second_look.tflite` via `MammogramClassifier.kt`,
   producing `P(worth second look)` and a tier.

Nothing is uploaded and nothing is written to shared storage: camera captures go
to the app's cache directory and the preprocessing scratch file is deleted after
each scan.

### Why Python instead of a port

The iOS app hand-ports the pipeline to Objective-C++/OpenCV, which means two
implementations that can silently drift apart — and a drift here feeds the model
inputs it was never trained on. Android can embed CPython, so this app runs the
training-time code verbatim:

```
app/src/main/python/
  config/constants.py               ┐
  data_pipeline/_imaging_utils.py   ├ copied verbatim from the research repo
  data_pipeline/preprocessor.py     │
  data_pipeline/label_mapper.py     ┘
  data_pipeline/__init__.py         trimmed: the repo's version imports the
                                    whole training stack (pandas, GCS, TF)
  second_look_bridge.py             the only Android-specific Python
```

Refresh the copied files after changing them upstream:

```bash
./gradlew :app:syncPythonPipeline        # defaults to the repo root (../..)
./gradlew :app:syncPythonPipeline -PsecondLookRepo=/path/to/secondlook
```

## Requirements

- Android Studio (AGP 9.2.x — pinned because Chaquopy 17 supports AGP ≤ 9.2)
- JDK 21 (Android Studio's bundled JBR is fine)
- **Python 3.10 on the build machine**, for Chaquopy's `buildPython`:

  ```bash
  brew install python@3.10        # macOS
  ```

  3.10 specifically: it is the only version with OpenCV wheels in Chaquopy's
  Android package repository, and the pipeline is cv2-based. Chaquopy finds
  `python3.10` on `PATH`; if Android Studio does not inherit your shell `PATH`,
  set an explicit path instead:

  ```properties
  # local.properties or gradle.properties
  chaquopyBuildPython=/opt/homebrew/bin/python3.10
  ```

Minimum device: API 24, `arm64-v8a` or `x86_64` (the ABIs Chaquopy's NumPy and
OpenCV wheels are built for — see `abiFilters`).

## Build & run

```bash
./gradlew :app:installDebug          # build + install on a running device
./gradlew :app:connectedAndroidTest  # pipeline + inference tests, on-device
```

The first build downloads the CPython runtime plus NumPy and OpenCV wheels, and
the first launch unpacks them, so both are slower than usual.

Note: the Gradle configuration cache is disabled in `gradle.properties` —
Chaquopy shells out to `buildPython` during configuration, which the cache
forbids.

## The model

`app/src/main/assets/second_look.tflite` (~4.4 MB) is the trained classifier,
bundled into the APK and memory-mapped at runtime (hence `noCompress += "tflite"`).
It is trained separately in the Python pipeline. If you retrain, replace this
file and keep the 224×224×1 input contract — `MammogramClassifier` verifies the
preprocessed tensor against the model's declared input size and fails loudly if
they disagree.

## Project layout

```
app/src/main/java/com/hexogen/secondlook/
  SecondLookApplication.kt     Starts the CPython runtime once per process
  MainActivity.kt              Entry point
  ui/SecondLookFlow.kt         The four screens (disclaimer → capture → results)
  ui/theme/                    Colors, typography, tier palette
  MammogramPreprocessor.kt     Kotlin ↔ Python bridge
  MammogramClassifier.kt       TFLite interpreter wrapper + tiering
  ImageLoading.kt              EXIF orientation, preview decoding, capture URIs
app/src/main/python/           The preprocessing pipeline (see above)
app/src/main/assets/           second_look.tflite
app/src/androidTest/           On-device pipeline + inference tests
```

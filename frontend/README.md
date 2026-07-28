# SecondLook

An iOS app that runs on-device triage of mammogram images. A user selects or
captures a mammogram, the image is preprocessed and passed through a bundled
TensorFlow Lite model, and the app returns a probability and a coarse tier
(Low / Moderate / Elevated).

## ⚠️ Not a medical device

SecondLook is a research and educational prototype. It is **not** a diagnostic
tool, is **not** FDA/CE cleared, and must not be used to make clinical or
personal health decisions. The tier cut-points are provisional and uncalibrated.
The app shows a disclaimer gate on launch for this reason.

## How it works

1. **Capture / select** an image (`ContentView.swift`, SwiftUI flow).
2. **Preprocess** it in `MammogramPreprocessor.mm` (Objective-C++ / OpenCV):
   grayscale → CLAHE → breast mask → pectoral-muscle removal → orientation
   normalization → resize to 224×224 → float32 in `[0, 1]`. This mirrors the
   Python training pipeline so on-device inputs match training inputs.
3. **Classify** with `second_look.tflite` via `MammogramClassifier.swift`
   (TensorFlow Lite), producing `P(worth second look)` and a tier.

## Requirements

- Xcode 15+
- iOS 15+ device or simulator
- [CocoaPods](https://cocoapods.org) (for TensorFlow Lite)

## Build & run

```bash
pod install
open SecondLook.xcworkspace   # NOT the .xcodeproj
```

Then select a simulator or device and run. Dependencies:

- **TensorFlow Lite** — via CocoaPods (`Podfile`). `Pods/` is not committed;
  run `pod install` after cloning to fetch it (versions are pinned in
  `Podfile.lock`).
- **OpenCV** — via Swift Package Manager (`opencv-spm`), resolved automatically
  by Xcode from `Package.resolved`.

## The model

`SecondLook/second_look.tflite` (~4.4 MB) is the trained classifier, bundled
into the app. It is trained separately in a Python pipeline (not part of this
repo). If you retrain, replace this file and keep the 224×224×1 input contract.

## Project layout

```
SecondLook/
  SecondLookApp.swift          App entry point
  ContentView.swift            SwiftUI screens (disclaimer → capture → results)
  MammogramClassifier.swift    TFLite interpreter wrapper + tiering
  MammogramPreprocessor.h/.mm  OpenCV preprocessing (ObjC++)
  SecondLook-Bridging-Header.h ObjC++ ↔ Swift bridge
  second_look.tflite           Bundled model
SecondLookTests/               Unit tests
SecondLookUITests/             UI tests
```

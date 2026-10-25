"""Second Look data pipeline — on-device subset.

This is NOT the research repo's ``data_pipeline/__init__.py``. That one eagerly
imports the whole training pipeline (pandas, PyYAML, google-cloud-storage,
TensorFlow), none of which is available — or wanted — inside the app. Only the
inference-time modules are vendored here:

    _imaging_utils.py   grayscale + breast mask primitives
    preprocessor.py     the preprocessing pipeline the model was trained with
    label_mapper.py     probability -> concern tier

Those three files are copied verbatim from the research repo; keep them that
way so training-time and on-device preprocessing cannot diverge. Refresh them
with ``./gradlew :app:syncPythonPipeline``.
"""

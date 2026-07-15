"""Packaging for the Second Look training source distribution.

This exists so the code can be shipped to a Vertex AI Custom Job as a Python
source distribution (``python setup.py sdist`` -> a ``.tar.gz`` on GCS), run by
a PREBUILT TensorFlow training container. The container already provides
tensorflow/keras/numpy; ``install_requires`` lists only the extras it lacks.

We deliberately do NOT depend on tensorflow here (the container pins a
CUDA-matched build) and use ``opencv-python-headless`` (no libGL) because the
training VM is headless. gradio / pytest / matplotlib-for-demos are omitted —
they are not needed to train.

Entrypoint module: ``scripts.train_vertex`` (pass as ``python-module`` to the
Vertex worker-pool spec).
"""

from setuptools import setup

setup(
    name="second-look-training",
    version="0.1.0",
    description="Second Look mammogram baseline — Vertex AI training package",
    python_requires=">=3.9",
    # Explicit package list. These dirs carry __init__.py so setuptools treats
    # them as regular packages and imports resolve after install.
    packages=["config", "data_pipeline", "modeling", "scripts"],
    # Ship the YAML configs the build step reads (config/sources.yaml,
    # config/label_maps.yaml). MANIFEST.in also lists them so they land in the
    # sdist tarball itself.
    package_data={"config": ["*.yaml"]},
    include_package_data=True,
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "opencv-python-headless>=4.8",
        "Pillow>=10.0",
        "PyYAML>=6.0.2",
        "google-cloud-storage>=2.18",
        "matplotlib>=3.7",
    ],
)

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.manifest import (
    MANIFEST_COLUMNS,
    build_patient_manifest,
    load_label_maps_config,
    map_raw_label,
)
from data_pipeline.retriever import load_sources_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_map_raw_label_rules():
    label_maps = load_label_maps_config(REPO_ROOT / "config" / "label_maps.yaml")

    assert map_raw_label("cbis", "MALIGNANT", label_maps) == 1
    assert map_raw_label("cbis", "BENIGN_WITHOUT_CALLBACK", label_maps) == 0
    assert map_raw_label("rsna", 1, label_maps) == 1
    assert map_raw_label("rsna", 0, label_maps) == 0
    assert map_raw_label("vindr", "BI-RADS 4", label_maps) == 1
    assert map_raw_label("vindr", "BI-RADS 3", label_maps) == 0

    with pytest.raises(ValueError):
        map_raw_label("cbis", "UNKNOWN", label_maps)


def test_build_patient_manifest_allows_missing_metadata_when_requested(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)

    (cfg_dir / "sources.yaml").write_text(
        """
datasets:
  cbis:
    bucket: "b"
    metadata_gcs_uri: "gs://b/cbis.csv"
    metadata_local_path: "does/not/exist.csv"
    columns:
      patient_id: "patient_id"
      raw_label: "pathology"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (cfg_dir / "label_maps.yaml").write_text(
        """
datasets:
  cbis:
    type: "categorical"
    mappings:
      MALIGNANT: 1
      BENIGN: 1
      BENIGN_WITHOUT_CALLBACK: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    sources = load_sources_config(cfg_dir / "sources.yaml")
    label_maps = load_label_maps_config(cfg_dir / "label_maps.yaml")

    manifest = build_patient_manifest(
        repo_root=tmp_path,
        selected_datasets=["cbis"],
        sources=sources,
        label_maps=label_maps,
        allow_missing_metadata=True,
    )
    assert isinstance(manifest, pd.DataFrame)
    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert manifest.empty

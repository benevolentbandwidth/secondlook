"""Tests for the CBIS image-level manifest builder and official split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.manifest import (
    IMAGE_MANIFEST_COLUMNS,
    build_cbis_image_manifest,
    load_label_maps_config,
)
from data_pipeline.retriever import load_sources_config
from data_pipeline.splitter import official_split_train_val


REPO_ROOT = Path(__file__).resolve().parents[1]


def _cbis_metadata_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["P_00001", "P_00001", "P_00200", "P_00141"],
            "pathology": [
                "MALIGNANT", "MALIGNANT", "BENIGN_WITHOUT_CALLBACK", "BENIGN"
            ],
            "image file path": [
                "Mass-Training_P_00001_LEFT_CC/u/v/a.dcm",
                "Mass-Training_P_00001_LEFT_MLO/u/v/a.dcm",
                "Mass-Test_P_00200_LEFT_MLO/u/v/a.dcm",
                "Calc-Test_P_00141_LEFT_CC/u/v/a.dcm",
            ],
            "case_folder": [
                "Mass-Training_P_00001_LEFT_CC",
                "Mass-Training_P_00001_LEFT_MLO",
                "Mass-Test_P_00200_LEFT_MLO",
                "Calc-Test_P_00141_LEFT_CC",
            ],
            "split": ["train", "train", "test", "test"],
            "abnormality_type": ["mass", "mass", "mass", "calc"],
        }
    )


def test_build_cbis_image_manifest_shape_and_labels():
    sources = load_sources_config(REPO_ROOT / "config" / "sources.yaml")
    label_maps = load_label_maps_config(REPO_ROOT / "config" / "label_maps.yaml")

    manifest = build_cbis_image_manifest(_cbis_metadata_df(), sources["cbis"], label_maps)

    assert list(manifest.columns) == IMAGE_MANIFEST_COLUMNS
    assert len(manifest) == 4
    assert set(manifest["split"]) == {"train", "test"}
    assert set(manifest["canonical_label"].unique()).issubset({0, 1})
    p200 = manifest[manifest["case_folder"] == "Mass-Test_P_00200_LEFT_MLO"].iloc[0]
    assert int(p200["canonical_label"]) == 0  # BENIGN_WITHOUT_CALLBACK


def test_official_split_train_val_preserves_test_partition():
    df = pd.DataFrame(
        {
            "case_folder": [f"c{i}" for i in range(20)],
            "canonical_label": [0, 1] * 10,
            "split": ["train"] * 14 + ["test"] * 6,
        }
    )

    train, val, test = official_split_train_val(
        df, label_column="canonical_label", split_column="split", val_fraction=0.2, seed=42
    )

    assert len(test) == 6
    assert len(train) + len(val) == 14
    # Stratified val keeps ~equal class balance.
    assert set(val["canonical_label"]).issubset({0, 1})


def test_official_split_rejects_unexpected_split_values():
    df = pd.DataFrame(
        {
            "canonical_label": [0, 1, 0, 1],
            "split": ["train", "train", "validation", "test"],
        }
    )
    with pytest.raises(ValueError, match="Unexpected values"):
        official_split_train_val(df, label_column="canonical_label")

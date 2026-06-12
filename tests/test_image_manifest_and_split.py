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
from data_pipeline.splitter import official_split_train_val, split_dataset


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


def test_split_dataset_grouped_keeps_patient_in_one_split():
    # 40 patients, 4 images each (RSNA-shaped). Half the patients are positive
    # in their left breast only (mixed labels within the patient).
    rows = []
    for pid in range(40):
        left_label = 1 if pid < 20 else 0
        rows.append({"patient_id": f"P{pid:03d}", "laterality": "L", "view": "CC", "canonical_label": left_label})
        rows.append({"patient_id": f"P{pid:03d}", "laterality": "L", "view": "MLO", "canonical_label": left_label})
        rows.append({"patient_id": f"P{pid:03d}", "laterality": "R", "view": "CC", "canonical_label": 0})
        rows.append({"patient_id": f"P{pid:03d}", "laterality": "R", "view": "MLO", "canonical_label": 0})
    df = pd.DataFrame(rows)

    train, val, test = split_dataset(
        df, label_column="canonical_label", group_column="patient_id"
    )

    train_pat = set(train["patient_id"])
    val_pat = set(val["patient_id"])
    test_pat = set(test["patient_id"])
    assert train_pat.isdisjoint(val_pat)
    assert train_pat.isdisjoint(test_pat)
    assert val_pat.isdisjoint(test_pat)
    assert len(train_pat | val_pat | test_pat) == 40
    # Every patient contributes exactly 4 rows to exactly one split.
    for partition in (train, val, test):
        sizes = partition.groupby("patient_id").size()
        assert (sizes == 4).all()


def test_official_split_train_val_with_group_column_keeps_patients_intact():
    """VinDr publishes a study-level official split: all 4 images of a study
    are in 'training' or 'test' together. Carving val out of train must not
    split a study across train and val. Build 30 training studies + 10 test
    studies, each with 4 images, and assert no study straddles train/val."""
    rows = []
    rng = list(range(40))
    for i in rng:
        label = 1 if i % 2 == 0 else 0
        split = "train" if i < 30 else "test"
        for view in range(4):
            rows.append(
                {
                    "patient_id": f"study_{i:03d}",
                    "image_id": f"study_{i:03d}_img{view}",
                    "canonical_label": label,
                    "split": split,
                }
            )
    df = pd.DataFrame(rows)

    train, val, test = official_split_train_val(
        df,
        label_column="canonical_label",
        split_column="split",
        val_fraction=0.2,
        seed=42,
        group_column="patient_id",
    )

    train_pat = set(train["patient_id"])
    val_pat = set(val["patient_id"])
    test_pat = set(test["patient_id"])
    assert train_pat.isdisjoint(val_pat)
    assert train_pat.isdisjoint(test_pat)
    assert val_pat.isdisjoint(test_pat)
    assert len(test_pat) == 10
    # Every study in train or val contributes all 4 rows to one side.
    for partition in (train, val, test):
        sizes = partition.groupby("patient_id").size()
        assert (sizes == 4).all()


def test_official_split_rejects_unexpected_split_values():
    df = pd.DataFrame(
        {
            "canonical_label": [0, 1, 0, 1],
            "split": ["train", "train", "validation", "test"],
        }
    )
    with pytest.raises(ValueError, match="Unexpected values"):
        official_split_train_val(df, label_column="canonical_label")

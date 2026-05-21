"""Manifest builder for patient-level canonical labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data_pipeline.retriever import (
    DatasetSourceConfig,
    load_cbis_case_metadata,
    load_rsna_case_metadata,
    resolve_metadata_local_path,
)


MANIFEST_COLUMNS = [
    "dataset",
    "patient_id",
    "canonical_label",
    "raw_label_values",
    "record_count",
    "source_refs",
]

IMAGE_MANIFEST_COLUMNS = [
    "dataset",
    "patient_id",
    "case_folder",
    "image_id",
    "abnormality_type",
    "raw_label",
    "canonical_label",
    "split",
    "image_local_path",
]


def load_label_maps_config(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate label-map config from YAML."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ValueError(f"Label map config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    datasets = raw.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Label map config must define a non-empty 'datasets' mapping.")

    result: dict[str, dict[str, Any]] = {}
    for name, spec in datasets.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Dataset '{name}' label map must be an object.")
        if "type" not in spec:
            raise ValueError(f"Dataset '{name}' label map requires a 'type' field.")
        result[name.lower()] = spec
    return result


def map_raw_label(dataset: str, raw_value: Any, label_maps: dict[str, dict[str, Any]]) -> int:
    """Map one raw metadata value into canonical binary label."""
    ds = dataset.lower()
    if ds not in label_maps:
        raise ValueError(f"No label map configured for dataset '{dataset}'.")

    spec = label_maps[ds]
    map_type = spec["type"]

    if map_type == "categorical":
        mappings = spec.get("mappings", {})
        normalized = str(raw_value).strip().upper().replace(" ", "_")
        if normalized not in mappings:
            raise ValueError(f"Unknown categorical label for {dataset}: {raw_value}")
        return int(mappings[normalized])

    if map_type == "integer":
        mappings = spec.get("mappings", {})
        key = str(_coerce_int(raw_value))
        if key not in mappings:
            raise ValueError(f"Unknown integer label for {dataset}: {raw_value}")
        return int(mappings[key])

    if map_type == "birads_range":
        birads = _parse_birads(raw_value)
        pos_min = int(spec["positive_min"])
        pos_max = int(spec["positive_max"])
        neg_min = int(spec["negative_min"])
        neg_max = int(spec["negative_max"])
        if neg_min <= birads <= neg_max:
            return 0
        if pos_min <= birads <= pos_max:
            return 1
        raise ValueError(f"BI-RADS value out of configured range for {dataset}: {raw_value}")

    raise ValueError(f"Unsupported mapping type '{map_type}' for dataset '{dataset}'.")


def build_patient_manifest_for_dataset(
    dataset: str,
    metadata_df: pd.DataFrame,
    source_cfg: DatasetSourceConfig,
    label_maps: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build one patient-level manifest DataFrame for a dataset."""
    _require_columns(
        metadata_df,
        source_cfg.name,
        [source_cfg.patient_id_column, source_cfg.raw_label_column],
    )

    source_ref_series = (
        metadata_df[source_cfg.source_ref_column].astype(str)
        if source_cfg.source_ref_column and source_cfg.source_ref_column in metadata_df.columns
        else pd.Series([""] * len(metadata_df))
    )

    working = pd.DataFrame(
        {
            "patient_id": metadata_df[source_cfg.patient_id_column].astype(str),
            "raw_label": metadata_df[source_cfg.raw_label_column],
            "source_ref": source_ref_series,
        }
    )
    working["mapped_label"] = working["raw_label"].map(
        lambda value: map_raw_label(dataset, value, label_maps)
    )
    working["raw_label_text"] = working["raw_label"].astype(str)

    grouped = (
        working.groupby("patient_id", dropna=False)
        .agg(
            canonical_label=("mapped_label", "max"),
            raw_label_values=("raw_label_text", lambda s: "|".join(sorted(set(s)))),
            record_count=("mapped_label", "size"),
            source_refs=("source_ref", lambda s: "|".join(sorted({v for v in s if v}))),
        )
        .reset_index()
    )
    grouped.insert(0, "dataset", dataset)
    return grouped[MANIFEST_COLUMNS]


def build_cbis_image_manifest(
    metadata_df: pd.DataFrame,
    source_cfg: DatasetSourceConfig,
    label_maps: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a per-case-folder manifest for CBIS-DDSM.

    Expects ``metadata_df`` produced by ``load_cbis_case_metadata`` (has
    ``split``, ``abnormality_type``, ``case_folder`` columns added). One CBIS
    row already corresponds to one case folder (full mammogram + view), so we
    deduplicate on case_folder but warn if labels disagree across duplicates.
    """
    required = (
        source_cfg.patient_id_column,
        source_cfg.raw_label_column,
        "case_folder",
        "split",
        "abnormality_type",
    )
    _require_columns(metadata_df, source_cfg.name, list(required))

    working = pd.DataFrame(
        {
            "dataset": source_cfg.name,
            "patient_id": metadata_df[source_cfg.patient_id_column].astype(str),
            "case_folder": metadata_df["case_folder"].astype(str),
            "abnormality_type": metadata_df["abnormality_type"].astype(str),
            "raw_label": metadata_df[source_cfg.raw_label_column].astype(str),
            "split": metadata_df["split"].astype(str),
        }
    )
    working = working[working["case_folder"] != ""].copy()
    working["canonical_label"] = working["raw_label"].map(
        lambda v: map_raw_label(source_cfg.name, v, label_maps)
    )

    # Deduplicate: take the max canonical_label per case_folder (positive wins).
    grouped = (
        working.groupby(["dataset", "patient_id", "case_folder"], as_index=False)
        .agg(
            abnormality_type=("abnormality_type", lambda s: "|".join(sorted(set(s)))),
            raw_label=("raw_label", lambda s: "|".join(sorted(set(s)))),
            canonical_label=("canonical_label", "max"),
            split=("split", "first"),
        )
    )
    grouped["image_id"] = ""
    grouped["image_local_path"] = ""
    return grouped[IMAGE_MANIFEST_COLUMNS]


def build_rsna_image_manifest(
    metadata_df: pd.DataFrame,
    source_cfg: DatasetSourceConfig,
    label_maps: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a per-image manifest for RSNA.

    Each row in the input train.csv is already one image; the GCS layout is
    ``train_images/{patient_id}/{image_id}.png`` so ``case_folder`` is the
    patient_id and ``image_id`` is the blob filename stem. No official split
    column (test.csv is the unlabeled Kaggle holdout); ``split`` stays empty
    here and gets filled by the stratified splitter downstream.
    """
    required = (
        source_cfg.patient_id_column,
        source_cfg.raw_label_column,
        source_cfg.source_ref_column or "image_id",
    )
    _require_columns(metadata_df, source_cfg.name, list(required))
    image_id_col = source_cfg.source_ref_column or "image_id"

    working = pd.DataFrame(
        {
            "dataset": source_cfg.name,
            "patient_id": metadata_df[source_cfg.patient_id_column].astype(str),
            "case_folder": metadata_df[source_cfg.patient_id_column].astype(str),
            "image_id": metadata_df[image_id_col].astype(str),
            "abnormality_type": "",
            "raw_label": metadata_df[source_cfg.raw_label_column].astype(str),
        }
    )
    working = working[(working["case_folder"] != "") & (working["image_id"] != "")].copy()
    working["canonical_label"] = working["raw_label"].map(
        lambda v: map_raw_label(source_cfg.name, v, label_maps)
    )
    working = working.drop_duplicates(subset=["case_folder", "image_id"]).reset_index(drop=True)
    working["split"] = ""
    working["image_local_path"] = ""
    return working[IMAGE_MANIFEST_COLUMNS]


def build_patient_manifest(
    repo_root: str | Path,
    selected_datasets: list[str],
    sources: dict[str, DatasetSourceConfig],
    label_maps: dict[str, dict[str, Any]],
    *,
    allow_missing_metadata: bool = False,
    use_gcs: bool = False,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build patient manifest across all selected datasets.

    With ``use_gcs=True`` (and ``cache_dir`` set), CBIS metadata is pulled from
    the four case-description CSVs in GCS instead of the local fixture. Other
    datasets continue to read from their local CSVs until their GCS loaders land.
    """
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    if use_gcs and cache_dir is None:
        raise ValueError("use_gcs=True requires cache_dir to be set.")

    for dataset in selected_datasets:
        ds = dataset.lower()
        if ds not in sources:
            raise ValueError(f"Dataset '{dataset}' not found in sources config.")
        if ds not in label_maps:
            raise ValueError(f"Dataset '{dataset}' not found in label-map config.")

        source_cfg = sources[ds]

        if use_gcs and ds == "cbis":
            metadata_df = load_cbis_case_metadata(source_cfg, cache_dir)
        elif use_gcs and ds == "rsna":
            metadata_df = load_rsna_case_metadata(source_cfg, cache_dir)
        else:
            metadata_path = resolve_metadata_local_path(repo_root, source_cfg)
            if not metadata_path or not metadata_path.exists():
                missing.append(ds)
                continue
            metadata_df = pd.read_csv(metadata_path)

        frames.append(build_patient_manifest_for_dataset(ds, metadata_df, source_cfg, label_maps))

    if missing and not allow_missing_metadata:
        raise ValueError(
            "Missing metadata CSV for datasets: "
            f"{sorted(missing)}. Set metadata_local_path or use --allow-missing-metadata."
        )

    if not frames:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    manifest = pd.concat(frames, ignore_index=True)
    return manifest.sort_values(["dataset", "patient_id"]).reset_index(drop=True)


def _require_columns(df: pd.DataFrame, dataset: str, required: list[str]) -> None:
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{dataset}' metadata missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer-like label, got: {value}") from exc


def _parse_birads(value: Any) -> int:
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            raise ValueError(f"Invalid BI-RADS value: {value}")
        return int(digits)
    return _coerce_int(value)

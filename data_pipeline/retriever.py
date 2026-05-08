"""Config + access helpers for manifest-first dataset retrieval.

This module intentionally supports a manifest-only phase where metadata can be
validated and transformed without downloading images from GCS yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSourceConfig:
    """Resolved source configuration for one dataset."""

    name: str
    bucket: str
    metadata_gcs_uri: str
    metadata_local_path: str | None
    patient_id_column: str
    raw_label_column: str
    source_ref_column: str | None


def load_sources_config(path: str | Path) -> dict[str, DatasetSourceConfig]:
    """Load dataset source config from YAML and validate required fields."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ValueError(f"Sources config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    datasets = raw.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Sources config must define a non-empty 'datasets' mapping.")

    result: dict[str, DatasetSourceConfig] = {}
    for dataset_name, spec in datasets.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Dataset '{dataset_name}' must map to an object.")

        columns = spec.get("columns")
        if not isinstance(columns, dict):
            raise ValueError(f"Dataset '{dataset_name}' is missing a valid 'columns' object.")

        required = ("patient_id", "raw_label")
        missing_cols = [k for k in required if not columns.get(k)]
        if missing_cols:
            raise ValueError(
                f"Dataset '{dataset_name}' missing required column mappings: {missing_cols}"
            )

        bucket = spec.get("bucket")
        metadata_gcs_uri = spec.get("metadata_gcs_uri")
        if not bucket or not metadata_gcs_uri:
            raise ValueError(
                f"Dataset '{dataset_name}' requires both 'bucket' and 'metadata_gcs_uri'."
            )

        result[dataset_name.lower()] = DatasetSourceConfig(
            name=dataset_name.lower(),
            bucket=str(bucket),
            metadata_gcs_uri=str(metadata_gcs_uri),
            metadata_local_path=(
                str(spec["metadata_local_path"]) if spec.get("metadata_local_path") else None
            ),
            patient_id_column=str(columns["patient_id"]),
            raw_label_column=str(columns["raw_label"]),
            source_ref_column=(str(columns["source_ref"]) if columns.get("source_ref") else None),
        )

    return result


def resolve_metadata_local_path(
    repo_root: str | Path, dataset_cfg: DatasetSourceConfig
) -> Path | None:
    """Resolve metadata_local_path against repo root."""
    if not dataset_cfg.metadata_local_path:
        return None
    return (Path(repo_root) / dataset_cfg.metadata_local_path).resolve()


def list_gcs_objects(
    bucket_name: str,
    prefix: str = "",
    *,
    project: str | None = None,
    max_results: int = 100,
) -> list[str]:
    """List object names from GCS using google-cloud-storage SDK.

    This is optional for manifest-only mode and can be enabled once object-level
    permissions are granted.
    """
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-storage is required for live GCS listing. "
            "Install dependencies from requirements.txt."
        ) from exc

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket, prefix=prefix, max_results=max_results)
    return [blob.name for blob in blobs]


def summarize_sources_for_dry_run(
    repo_root: str | Path,
    sources: dict[str, DatasetSourceConfig],
    selected_datasets: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return a dry-run summary table for selected datasets."""
    chosen = (
        sorted(d.lower() for d in selected_datasets)
        if selected_datasets
        else sorted(sources.keys())
    )

    rows: list[dict[str, Any]] = []
    for dataset in chosen:
        if dataset not in sources:
            raise ValueError(f"Dataset '{dataset}' not found in sources config.")
        spec = sources[dataset]
        local_path = resolve_metadata_local_path(repo_root, spec)
        rows.append(
            {
                "dataset": dataset,
                "bucket": spec.bucket,
                "metadata_gcs_uri": spec.metadata_gcs_uri,
                "metadata_local_path": str(local_path) if local_path else "",
                "metadata_exists": bool(local_path and local_path.exists()),
                "patient_id_column": spec.patient_id_column,
                "raw_label_column": spec.raw_label_column,
                "source_ref_column": spec.source_ref_column or "",
            }
        )

    return rows

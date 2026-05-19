"""Config + access helpers for manifest-first dataset retrieval.

This module intentionally supports a manifest-only phase where metadata can be
validated and transformed without downloading images from GCS yet.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd
import yaml


DEFAULT_IMAGE_DOWNLOAD_WORKERS = 8


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
    metadata_gcs_uris: tuple[str, ...] = field(default_factory=tuple)
    images_gcs_prefix: str | None = None
    image_extension: str | None = None
    official_split: bool = False


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

        uris_raw = spec.get("metadata_gcs_uris") or []
        if not isinstance(uris_raw, list) or not all(isinstance(u, str) for u in uris_raw):
            raise ValueError(
                f"Dataset '{dataset_name}' 'metadata_gcs_uris' must be a list of strings."
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
            metadata_gcs_uris=tuple(uris_raw),
            images_gcs_prefix=(
                str(spec["images_gcs_prefix"]) if spec.get("images_gcs_prefix") else None
            ),
            image_extension=(
                str(spec["image_extension"]) if spec.get("image_extension") else None
            ),
            official_split=bool(spec.get("official_split", False)),
        )

    return result


def resolve_metadata_local_path(
    repo_root: str | Path, dataset_cfg: DatasetSourceConfig
) -> Path | None:
    """Resolve metadata_local_path against repo root."""
    if not dataset_cfg.metadata_local_path:
        return None
    return (Path(repo_root) / dataset_cfg.metadata_local_path).resolve()


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split a gs://bucket/key URI into (bucket, object_name)."""
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _get_storage_client(project: str | None = None):
    """Return a google.cloud.storage.Client. Imported lazily."""
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-storage is required for GCS access. "
            "Install dependencies from requirements.txt."
        ) from exc
    return storage.Client(project=project)


def download_metadata_csv(
    gcs_uri: str,
    cache_dir: str | Path,
    *,
    client: Any = None,
    project: str | None = None,
) -> Path:
    """Download a single CSV from GCS into cache_dir. Skip-if-cached.

    Cached filename mirrors the GCS object name (slashes -> underscores) so
    multiple datasets don't collide in one cache dir.
    """
    bucket_name, object_name = _parse_gs_uri(gcs_uri)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    local_path = cache_root / object_name.replace("/", "_")

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    if client is None:
        client = _get_storage_client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.download_to_filename(str(local_path))
    return local_path


def _case_folder_from_image_path(image_file_path: str) -> str:
    """Extract the case folder (first path segment) from a CBIS image file path."""
    if not isinstance(image_file_path, str) or not image_file_path:
        return ""
    return image_file_path.split("/", 1)[0]


def _cbis_split_and_type_from_uri(uri: str) -> tuple[str, str]:
    """Infer (split, abnormality_type) from a CBIS case CSV filename.

    Filenames follow {calc|mass}_case_description_{train|test}_set.csv.
    """
    name = uri.rsplit("/", 1)[-1].lower()
    if name.startswith("calc_"):
        abnormality_type = "calc"
    elif name.startswith("mass_"):
        abnormality_type = "mass"
    else:
        raise ValueError(f"Cannot infer abnormality_type from CBIS CSV filename: {name}")
    if "_train_" in name:
        split = "train"
    elif "_test_" in name:
        split = "test"
    else:
        raise ValueError(f"Cannot infer split from CBIS CSV filename: {name}")
    return split, abnormality_type


def load_cbis_case_metadata(
    source_cfg: DatasetSourceConfig,
    cache_dir: str | Path,
    *,
    client: Any = None,
    project: str | None = None,
) -> pd.DataFrame:
    """Pull the four CBIS case description CSVs from GCS and concatenate.

    Adds three derived columns: ``split`` ('train'|'test' from filename),
    ``abnormality_type`` ('calc'|'mass' from filename), and ``case_folder``
    (first path segment of the row's 'image file path' column).
    """
    uris: Iterable[str] = source_cfg.metadata_gcs_uris or (source_cfg.metadata_gcs_uri,)
    if not uris:
        raise ValueError(
            f"Dataset '{source_cfg.name}' has no metadata_gcs_uris configured."
        )

    frames: list[pd.DataFrame] = []
    for uri in uris:
        local_path = download_metadata_csv(uri, cache_dir, client=client, project=project)
        split, abnormality_type = _cbis_split_and_type_from_uri(uri)
        df = pd.read_csv(local_path)
        df["split"] = split
        df["abnormality_type"] = abnormality_type
        if source_cfg.source_ref_column and source_cfg.source_ref_column in df.columns:
            df["case_folder"] = df[source_cfg.source_ref_column].map(_case_folder_from_image_path)
        else:
            df["case_folder"] = ""
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class ImageDownloadResult:
    """Outcome of a single case-folder PNG download attempt."""

    case_folder: str
    local_path: Path | None
    gcs_object: str | None
    status: str  # 'downloaded' | 'cached' | 'missing' | 'multiple' | 'error'
    detail: str = ""


def _list_pngs_in_case_folder(client, bucket_name: str, prefix: str) -> list[str]:
    """Return the .png blob object names under a case-folder prefix."""
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket, prefix=prefix)
    return [b.name for b in blobs if b.name.lower().endswith(".png")]


def _download_one_image(
    client,
    bucket_name: str,
    case_folder: str,
    images_prefix: str,
    cache_dir: Path,
) -> ImageDownloadResult:
    local_path = cache_dir / f"{case_folder}.png"
    if local_path.exists() and local_path.stat().st_size > 0:
        return ImageDownloadResult(
            case_folder=case_folder,
            local_path=local_path,
            gcs_object=None,
            status="cached",
        )

    prefix = f"{images_prefix.rstrip('/')}/{case_folder}/"
    try:
        png_objects = _list_pngs_in_case_folder(client, bucket_name, prefix)
    except Exception as exc:  # network / auth failures
        return ImageDownloadResult(
            case_folder=case_folder,
            local_path=None,
            gcs_object=None,
            status="error",
            detail=f"list failed: {exc}",
        )

    if not png_objects:
        return ImageDownloadResult(
            case_folder=case_folder,
            local_path=None,
            gcs_object=None,
            status="missing",
            detail=f"no .png blobs under {prefix}",
        )
    if len(png_objects) > 1:
        return ImageDownloadResult(
            case_folder=case_folder,
            local_path=None,
            gcs_object=None,
            status="multiple",
            detail=f"{len(png_objects)} .png blobs under {prefix}",
        )

    object_name = png_objects[0]
    bucket = client.bucket(bucket_name)
    try:
        bucket.blob(object_name).download_to_filename(str(local_path))
    except Exception as exc:
        return ImageDownloadResult(
            case_folder=case_folder,
            local_path=None,
            gcs_object=object_name,
            status="error",
            detail=f"download failed: {exc}",
        )

    return ImageDownloadResult(
        case_folder=case_folder,
        local_path=local_path,
        gcs_object=object_name,
        status="downloaded",
    )


def download_images_for_manifest(
    manifest_df: pd.DataFrame,
    source_cfg: DatasetSourceConfig,
    cache_dir: str | Path,
    *,
    case_folder_column: str = "case_folder",
    max_workers: int = DEFAULT_IMAGE_DOWNLOAD_WORKERS,
    client: Any = None,
    project: str | None = None,
) -> list[ImageDownloadResult]:
    """Download one PNG per case folder referenced in ``manifest_df``.

    Each case folder lives at ``images_gcs_prefix/<case_folder>/`` and is
    expected to contain exactly one ``.png``. The blob's UUID-style filename
    is not predictable, so we list the prefix and pick the PNG. Concurrent
    downloads via a bounded thread pool. Skip-if-cached.
    """
    if not source_cfg.images_gcs_prefix:
        raise ValueError(
            f"Dataset '{source_cfg.name}' has no images_gcs_prefix configured."
        )
    if case_folder_column not in manifest_df.columns:
        raise ValueError(
            f"Manifest is missing required column '{case_folder_column}'."
        )

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    case_folders = sorted({c for c in manifest_df[case_folder_column].astype(str) if c})
    if not case_folders:
        return []

    if client is None:
        client = _get_storage_client(project=project)

    results: list[ImageDownloadResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _download_one_image,
                client,
                source_cfg.bucket,
                case_folder,
                source_cfg.images_gcs_prefix,
                cache_root,
            ): case_folder
            for case_folder in case_folders
        }
        for future in as_completed(futures):
            results.append(future.result())

    return sorted(results, key=lambda r: r.case_folder)


def write_download_report(
    results: Iterable[ImageDownloadResult], report_path: str | Path
) -> Path:
    """Write a CSV summarizing image download outcomes (failures + successes)."""
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "case_folder": r.case_folder,
                "status": r.status,
                "gcs_object": r.gcs_object or "",
                "local_path": str(r.local_path) if r.local_path else "",
                "detail": r.detail,
            }
            for r in results
        ]
    )
    df.to_csv(report, index=False)
    return report


def list_gcs_objects(
    bucket_name: str,
    prefix: str = "",
    *,
    project: str | None = None,
    max_results: int = 100,
) -> list[str]:
    """List object names from GCS using google-cloud-storage SDK."""
    client = _get_storage_client(project=project)
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

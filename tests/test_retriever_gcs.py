"""Tests for GCS-backed metadata retrieval (mocked storage client)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from data_pipeline.manifest import build_patient_manifest, load_label_maps_config
from data_pipeline.retriever import (
    download_images_for_manifest,
    download_metadata_csv,
    load_cbis_case_metadata,
    load_sources_config,
    write_download_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


CALC_TRAIN_CSV = (
    "patient_id,pathology,image file path\n"
    "P_00001,MALIGNANT,Calc-Training_P_00001_LEFT_CC/uid-a/uid-b/000000.dcm\n"
)
CALC_TEST_CSV = (
    "patient_id,pathology,image file path\n"
    "P_00141,BENIGN,Calc-Test_P_00141_LEFT_CC/uid-c/uid-d/000000.dcm\n"
)
MASS_TRAIN_CSV = (
    "patient_id,pathology,image file path\n"
    "P_00050,MALIGNANT,Mass-Training_P_00050_RIGHT_MLO/uid-e/uid-f/000000.dcm\n"
)
MASS_TEST_CSV = (
    "patient_id,pathology,image file path\n"
    "P_00200,BENIGN_WITHOUT_CALLBACK,Mass-Test_P_00200_LEFT_MLO/uid-g/uid-h/000000.dcm\n"
)

URI_TO_BODY = {
    "gs://b2-foundation/second-look/DDSM/calc_case_description_train_set.csv": CALC_TRAIN_CSV,
    "gs://b2-foundation/second-look/DDSM/calc_case_description_test_set.csv": CALC_TEST_CSV,
    "gs://b2-foundation/second-look/DDSM/mass_case_description_train_set.csv": MASS_TRAIN_CSV,
    "gs://b2-foundation/second-look/DDSM/mass_case_description_test_set.csv": MASS_TEST_CSV,
}


def _fake_storage_client(
    payloads: dict[str, str], *, png_objects_by_prefix: dict[str, list[str]] | None = None
) -> MagicMock:
    """Build a MagicMock that mimics google.cloud.storage.Client.

    - ``payloads``: maps 'gs://bucket/object' -> file body for downloads.
    - ``png_objects_by_prefix``: maps a prefix (e.g. 'images/Calc-X/') to the
      list of object names returned by list_blobs for that prefix.
    """
    client = MagicMock()
    png_objects_by_prefix = png_objects_by_prefix or {}

    def _bucket(name: str) -> MagicMock:
        bucket = MagicMock()

        def _blob(object_name: str) -> MagicMock:
            blob = MagicMock()
            uri = f"gs://{name}/{object_name}"

            def _download(filename: str) -> None:
                body = payloads.get(uri, f"FAKE-PNG-BYTES:{object_name}")
                Path(filename).write_bytes(
                    body.encode("utf-8") if isinstance(body, str) else body
                )

            blob.download_to_filename.side_effect = _download
            return blob

        bucket.blob.side_effect = _blob
        return bucket

    def _list_blobs(bucket_arg, prefix="", **kwargs):
        # MagicMock(name=...) sets the mock's repr-name, not a .name attribute.
        # Build the mock then assign .name explicitly.
        result = []
        for n in png_objects_by_prefix.get(prefix, []):
            blob = MagicMock()
            blob.name = n
            result.append(blob)
        return result

    client.bucket.side_effect = _bucket
    client.list_blobs.side_effect = _list_blobs
    return client


def test_download_metadata_csv_caches_and_skips_redownload(tmp_path):
    uri = "gs://b2-foundation/second-look/DDSM/calc_case_description_train_set.csv"
    client = _fake_storage_client(URI_TO_BODY)

    first = download_metadata_csv(uri, tmp_path, client=client)
    assert first.exists()
    assert first.read_text(encoding="utf-8") == CALC_TRAIN_CSV
    assert client.bucket.call_count == 1

    second = download_metadata_csv(uri, tmp_path, client=client)
    assert second == first
    assert client.bucket.call_count == 1  # skipped re-download


def test_load_cbis_case_metadata_concatenates_four_csvs(tmp_path):
    sources = load_sources_config(REPO_ROOT / "config" / "sources.yaml")
    client = _fake_storage_client(URI_TO_BODY)

    df = load_cbis_case_metadata(sources["cbis"], tmp_path, client=client)

    assert len(df) == 4
    assert set(df["split"]) == {"train", "test"}
    assert set(df["abnormality_type"]) == {"calc", "mass"}
    assert set(df["case_folder"]) == {
        "Calc-Training_P_00001_LEFT_CC",
        "Calc-Test_P_00141_LEFT_CC",
        "Mass-Training_P_00050_RIGHT_MLO",
        "Mass-Test_P_00200_LEFT_MLO",
    }


def test_build_patient_manifest_use_gcs_routes_cbis_through_loader(tmp_path, monkeypatch):
    sources = load_sources_config(REPO_ROOT / "config" / "sources.yaml")
    label_maps = load_label_maps_config(REPO_ROOT / "config" / "label_maps.yaml")
    client = _fake_storage_client(URI_TO_BODY)

    monkeypatch.setattr(
        "data_pipeline.retriever._get_storage_client", lambda project=None: client
    )

    manifest = build_patient_manifest(
        repo_root=REPO_ROOT,
        selected_datasets=["cbis"],
        sources=sources,
        label_maps=label_maps,
        use_gcs=True,
        cache_dir=tmp_path,
    )

    assert not manifest.empty
    assert set(manifest["patient_id"]) == {"P_00001", "P_00141", "P_00050", "P_00200"}
    assert set(manifest["canonical_label"].unique()).issubset({0, 1})
    p_00200 = manifest[manifest["patient_id"] == "P_00200"].iloc[0]
    assert int(p_00200["canonical_label"]) == 0  # BENIGN_WITHOUT_CALLBACK


def test_download_images_for_manifest_lists_then_downloads_and_caches(tmp_path):
    sources = load_sources_config(REPO_ROOT / "config" / "sources.yaml")
    cbis = sources["cbis"]

    manifest = pd.DataFrame(
        {
            "case_folder": [
                "Calc-Training_P_00001_LEFT_CC",
                "Mass-Test_P_00200_LEFT_MLO",
                "Mass-Test_P_00200_LEFT_MLO",  # duplicate — should dedupe
                "",  # blank — should be skipped
            ]
        }
    )

    prefix = cbis.images_gcs_prefix.rstrip("/")
    png_objects = {
        f"{prefix}/Calc-Training_P_00001_LEFT_CC/": [
            f"{prefix}/Calc-Training_P_00001_LEFT_CC/uid-a/uid-b/abc.png"
        ],
        f"{prefix}/Mass-Test_P_00200_LEFT_MLO/": [
            f"{prefix}/Mass-Test_P_00200_LEFT_MLO/uid-c/uid-d/def.png"
        ],
    }
    client = _fake_storage_client({}, png_objects_by_prefix=png_objects)

    results = download_images_for_manifest(
        manifest, cbis, tmp_path, client=client, max_workers=2
    )
    statuses = {r.case_folder: r.status for r in results}
    assert statuses == {
        "Calc-Training_P_00001_LEFT_CC": "downloaded",
        "Mass-Test_P_00200_LEFT_MLO": "downloaded",
    }
    for r in results:
        assert r.local_path is not None and r.local_path.exists()

    # Second call should hit cache and not re-download (no list_blobs / no blob writes).
    client.reset_mock()
    results2 = download_images_for_manifest(
        manifest, cbis, tmp_path, client=client, max_workers=2
    )
    assert all(r.status == "cached" for r in results2)
    client.list_blobs.assert_not_called()


def test_download_images_for_manifest_reports_missing_and_multiple(tmp_path):
    sources = load_sources_config(REPO_ROOT / "config" / "sources.yaml")
    cbis = sources["cbis"]
    manifest = pd.DataFrame({"case_folder": ["Case-Missing", "Case-Ambiguous"]})

    prefix = cbis.images_gcs_prefix.rstrip("/")
    png_objects = {
        f"{prefix}/Case-Missing/": [],
        f"{prefix}/Case-Ambiguous/": [
            f"{prefix}/Case-Ambiguous/u1/a.png",
            f"{prefix}/Case-Ambiguous/u1/b.png",
        ],
    }
    client = _fake_storage_client({}, png_objects_by_prefix=png_objects)

    results = download_images_for_manifest(manifest, cbis, tmp_path, client=client)
    statuses = {r.case_folder: r.status for r in results}
    assert statuses == {"Case-Missing": "missing", "Case-Ambiguous": "multiple"}

    report = write_download_report(results, tmp_path / "report.csv")
    df = pd.read_csv(report)
    assert set(df["status"]) == {"missing", "multiple"}


def test_build_patient_manifest_use_gcs_requires_cache_dir():
    sources = load_sources_config(REPO_ROOT / "config" / "sources.yaml")
    label_maps = load_label_maps_config(REPO_ROOT / "config" / "label_maps.yaml")

    with pytest.raises(ValueError, match="cache_dir"):
        build_patient_manifest(
            repo_root=REPO_ROOT,
            selected_datasets=["cbis"],
            sources=sources,
            label_maps=label_maps,
            use_gcs=True,
            cache_dir=None,
        )

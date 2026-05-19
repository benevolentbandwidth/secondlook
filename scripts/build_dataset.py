"""Build the Second Look dataset manifest.

Two modes:
- Default (manifest-only, offline): aggregate patient-level canonical labels
  from local metadata fixtures. Produces ``manifest_patients.csv``.
- ``--use-gcs``: pull CBIS-DDSM case description CSVs from GCS, build a
  per-case-folder image manifest, download the corresponding PNGs to a local
  cache, and apply the train/val/test split. Produces
  ``manifest_patients.csv`` (audit), ``manifest.csv`` (training-ready), and
  ``image_download_report.csv`` (download outcomes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.manifest import (
    IMAGE_MANIFEST_COLUMNS,
    MANIFEST_COLUMNS,
    build_cbis_image_manifest,
    build_patient_manifest,
    load_label_maps_config,
)
from data_pipeline.retriever import (
    download_images_for_manifest,
    load_cbis_case_metadata,
    load_sources_config,
    summarize_sources_for_dry_run,
    write_download_report,
)
from data_pipeline.splitter import official_split_train_val, split_dataset


DEFAULT_DATASETS = ["cbis", "rsna", "vindr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible dataset build (manifest + optional GCS image fetch).",
    )
    parser.add_argument("--sources-config", default="config/sources.yaml")
    parser.add_argument("--label-maps-config", default="config/label_maps.yaml")
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        help=f"Datasets to include (default: {' '.join(DEFAULT_DATASETS)}).",
    )
    parser.add_argument(
        "--output-manifest", default="data/manifest_patients.csv",
        help="Patient-level manifest output path.",
    )
    parser.add_argument(
        "--output-image-manifest", default="data/manifest.csv",
        help="Per-image training manifest output path (only written with --use-gcs).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-metadata", action="store_true")
    parser.add_argument(
        "--use-gcs", action="store_true",
        help="Pull CBIS metadata + images from GCS instead of local fixtures.",
    )
    parser.add_argument(
        "--cache-dir", default="data/cache",
        help="Local cache for downloaded CSVs and PNGs (skip-if-exists).",
    )
    parser.add_argument(
        "--skip-image-download", action="store_true",
        help="Build manifest from GCS metadata but don't fetch image PNGs.",
    )
    parser.add_argument(
        "--use-official-split", dest="use_official_split", action="store_true",
        default=True,
        help="Honor CBIS train/test boundary from filename (default).",
    )
    parser.add_argument(
        "--no-official-split", dest="use_official_split", action="store_false",
        help="Override official split; use stratified 70/15/15 instead.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Concurrent image download workers.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="If set, cap the CBIS image manifest to the first N case folders "
             "(after deterministic sort). Useful for smoke tests and small "
             "local runs before committing to the full ~30-70 GB download.",
    )
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def main() -> None:
    args = parse_args()
    sources = load_sources_config(REPO_ROOT / args.sources_config)
    label_maps = load_label_maps_config(REPO_ROOT / args.label_maps_config)
    selected = [d.lower() for d in args.datasets]

    if args.dry_run:
        summary_df = pd.DataFrame(summarize_sources_for_dry_run(REPO_ROOT, sources, selected))
        print("\nDry-run source summary:")
        print(summary_df.to_string(index=False))
        return

    cache_dir = _resolve(args.cache_dir)
    patient_out = _resolve(args.output_manifest)
    image_out = _resolve(args.output_image_manifest)

    patient_manifest = build_patient_manifest(
        repo_root=REPO_ROOT,
        selected_datasets=selected,
        sources=sources,
        label_maps=label_maps,
        allow_missing_metadata=args.allow_missing_metadata,
        use_gcs=args.use_gcs,
        cache_dir=cache_dir if args.use_gcs else None,
    ).reindex(columns=MANIFEST_COLUMNS)

    patient_out.parent.mkdir(parents=True, exist_ok=True)
    patient_manifest.to_csv(patient_out, index=False)
    print(f"Wrote patient manifest: {patient_out}  rows={len(patient_manifest)}")

    if not args.use_gcs:
        print("Manifest-only mode complete. Use --use-gcs to fetch images.")
        return

    if "cbis" not in selected:
        print("--use-gcs currently only fetches CBIS images. Skipping image step.")
        return

    cbis_cfg = sources["cbis"]
    cbis_metadata = load_cbis_case_metadata(cbis_cfg, cache_dir)
    image_manifest = build_cbis_image_manifest(cbis_metadata, cbis_cfg, label_maps)
    print(f"Built CBIS image manifest: {len(image_manifest)} rows")

    if args.limit is not None and args.limit > 0:
        # Sample across both official splits so even tiny --limit runs leave
        # the splitter with non-empty train and test pools. Deterministic via
        # config.constants.SEED.
        from config.constants import SEED

        per_split = max(1, args.limit // 2)
        sampled = []
        for split_value, group in image_manifest.groupby("split"):
            n = min(per_split, len(group))
            sampled.append(group.sample(n=n, random_state=SEED))
        image_manifest = (
            pd.concat(sampled, ignore_index=True)
            .sort_values("case_folder")
            .reset_index(drop=True)
        )
        print(
            f"--limit applied: sampled to {len(image_manifest)} rows "
            f"(train+test, seed={SEED})"
        )

    if not args.skip_image_download:
        image_cache = cache_dir / "images" / "cbis"
        results = download_images_for_manifest(
            image_manifest, cbis_cfg, image_cache, max_workers=args.max_workers
        )
        report_path = image_out.parent / "image_download_report.csv"
        write_download_report(results, report_path)
        status_counts = pd.Series([r.status for r in results]).value_counts()
        print("Image download statuses:")
        print(status_counts.to_string())
        print(f"Wrote download report: {report_path}")

        local_path_by_folder = {
            r.case_folder: str(r.local_path) if r.local_path else "" for r in results
        }
        image_manifest["image_local_path"] = image_manifest["case_folder"].map(
            local_path_by_folder
        ).fillna("")

    if args.use_official_split and cbis_cfg.official_split:
        train_df, val_df, test_df = official_split_train_val(
            image_manifest, label_column="canonical_label", split_column="split"
        )
    else:
        train_df, val_df, test_df = split_dataset(
            image_manifest.drop(columns=["split"]),
            label_column="canonical_label",
        )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    final = pd.concat([train_df, val_df, test_df], ignore_index=True).reindex(
        columns=IMAGE_MANIFEST_COLUMNS
    )

    image_out.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(image_out, index=False)
    print(f"Wrote training manifest: {image_out}  rows={len(final)}")
    print("Split sizes:")
    print(final["split"].value_counts().to_string())
    print("Label distribution by split:")
    print(
        final.groupby("split")["canonical_label"].value_counts().unstack(fill_value=0).to_string()
    )


if __name__ == "__main__":
    main()

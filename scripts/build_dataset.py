"""Build a patient-level canonical manifest from configured dataset metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.manifest import (
    MANIFEST_COLUMNS,
    build_patient_manifest,
    load_label_maps_config,
)
from data_pipeline.retriever import load_sources_config, summarize_sources_for_dry_run


DEFAULT_DATASETS = ["cbis", "rsna", "vindr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manifest-first data builder (patient-level, no image downloads)."
    )
    parser.add_argument(
        "--sources-config",
        default="config/sources.yaml",
        help="Path to sources YAML config.",
    )
    parser.add_argument(
        "--label-maps-config",
        default="config/label_maps.yaml",
        help="Path to label map YAML config.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Datasets to include (default: {' '.join(DEFAULT_DATASETS)}).",
    )
    parser.add_argument(
        "--output-manifest",
        default="data/manifest_patients.csv",
        help="Output CSV path for patient-level manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and print source summary only.",
    )
    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help="Skip datasets with missing metadata_local_path files instead of failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT

    sources = load_sources_config(repo_root / args.sources_config)
    label_maps = load_label_maps_config(repo_root / args.label_maps_config)
    selected = [d.lower() for d in args.datasets]

    if args.dry_run:
        summary_rows = summarize_sources_for_dry_run(repo_root, sources, selected)
        summary_df = pd.DataFrame(summary_rows)
        print("\nDry-run source summary:")
        print(summary_df.to_string(index=False))
        print("\nDry run complete. No manifest written.")
        return

    manifest_df = build_patient_manifest(
        repo_root=repo_root,
        selected_datasets=selected,
        sources=sources,
        label_maps=label_maps,
        allow_missing_metadata=args.allow_missing_metadata,
    )
    manifest_df = manifest_df.reindex(columns=MANIFEST_COLUMNS)

    out_path = (repo_root / args.output_manifest).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(out_path, index=False)

    print(f"Wrote patient manifest: {out_path}")
    print(f"Rows: {len(manifest_df)}")
    if len(manifest_df):
        print("\nLabel distribution:")
        print(manifest_df["canonical_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

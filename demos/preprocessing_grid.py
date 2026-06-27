"""Generate a before/after preprocessing panel for the demo.

Renders each stage of data_pipeline.preprocessor on a real CBIS-DDSM scan:
raw -> grayscale -> CLAHE -> breast mask -> masked -> pectoral-removed ->
orientation-normalized -> final 224x224 model input.

Usage:
    python demos/preprocessing_grid.py [--case CASE_FOLDER] [--out PATH]

Reads from data/manifest.csv (cached images only). Writes a PNG suitable for
a slide.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config.constants import INPUT_SIZE
from data_pipeline import preprocessor as pp


def pick_case(manifest_path: Path, case: str | None) -> pd.Series:
    m = pd.read_csv(manifest_path)
    has_img = ~(m["image_local_path"].isna()
                | (m["image_local_path"].astype(str).str.strip() == ""))
    m = m[has_img].copy()
    if case:
        sel = m[m["case_folder"] == case]
        if sel.empty:
            raise SystemExit(f"Case {case!r} not found among cached images.")
        return sel.iloc[0]
    # Default: a positive MLO case (pectoral triangle visible).
    mlo_pos = m[m["case_folder"].str.contains("MLO", case=False)
                & (m["canonical_label"] == 1)]
    return (mlo_pos if not mlo_pos.empty else m).iloc[0]


def build_stages(path: str):
    raw = pp.load_image(path)
    gray = pp._to_grayscale(raw)
    clahe = pp._apply_clahe(gray)
    mask = pp._breast_mask(clahe)
    masked = cv2.bitwise_and(clahe, clahe, mask=mask)
    no_pec = pp._remove_pectoral(masked, mask)
    oriented = pp._normalize_orientation(no_pec, mask)
    final = cv2.resize(oriented, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    return [
        ("1. Raw scan", raw),
        ("2. Grayscale", gray),
        ("3. CLAHE contrast", clahe),
        ("4. Breast mask", mask),
        ("5. Background removed", masked),
        ("6. Pectoral removed", no_pec),
        ("7. Orientation normalized", oriented),
        (f"8. Model input {INPUT_SIZE[0]}x{INPUT_SIZE[1]}", final),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--manifest", default=str(REPO / "data" / "manifest.csv"))
    ap.add_argument("--out", default=str(REPO / "demos" / "preprocessing_grid.png"))
    args = ap.parse_args()

    row = pick_case(Path(args.manifest), args.case)
    label = "WORTH_SECOND_LOOK" if int(row["canonical_label"]) == 1 else "NOT_WORTH_SECOND_LOOK"
    stages = build_stages(row["image_local_path"])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    fig.suptitle(
        f"Second Look — preprocessing pipeline\n"
        f"{row['case_folder']}   (label: {label})",
        fontsize=15, fontweight="bold",
    )
    for ax, (title, img) in zip(axes.ravel(), stages):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Wrote {args.out}")
    print(f"Case: {row['case_folder']}  label: {label}")


if __name__ == "__main__":
    main()

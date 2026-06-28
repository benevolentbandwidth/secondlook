"""Render an end-to-end architecture diagram of the Second Look pipeline.

This is a factual flow of the system as it exists in the code today:
GCS -> retriever -> manifest -> preprocess -> baseline model -> checkpoint ->
evaluate / tier UX. The on-device TF Lite target is drawn as a future step.

Usage:
    python demos/pipeline_diagram.py [--out PATH]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).resolve().parents[1]

# (title, subtitle, source file) per stage, grouped into lanes.
LANES = [
    ("DATA", "#1565c0", [
        ("Google Cloud Storage", "CBIS-DDSM (RSNA, VinDr wired)", "gs://b2-foundation"),
        ("Retriever", "download CSV + PNG, local cache\n(skip-if-cached)", "retriever.py"),
        ("Manifest builder", "label -> WORTH / NOT WORTH,\npatient-disjoint splits", "manifest.py -> manifest.csv"),
    ]),
    ("PREPROCESS", "#00838f", [
        ("Quality gate", "reject blank / low-contrast /\nlow-resolution inputs", "quality.py"),
        ("Preprocessor", "grayscale -> CLAHE -> breast mask\n-> pectoral removal -> orient -> 224x224", "preprocessor.py"),
    ]),
    ("MODEL", "#6a1b9a", [
        ("Baseline classifier", "1x1 conv -> MobileNetV2 (frozen)\n-> GAP -> dropout -> sigmoid", "baseline_classifier.py"),
        ("Training", "tf.data + class weighting,\nbest checkpoint by val AUC", "train.py -> best.keras"),
    ]),
    ("EVALUATE / UX", "#2e7d32", [
        ("Evaluation", "sensitivity-first; WORTH floor 0.80;\nconfusion matrix", "evaluate.py"),
        ("Result + tiers", "Worth / Not worth a second look;\nLow / Moderate / Elevated", "label_mapper.py"),
        ("On-device (next)", "TF Lite, runs on phone/browser;\nstores & transmits nothing", "future"),
    ]),
]


def _box(ax, x, y, w, h, title, subtitle, src, color):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=color, facecolor="white", zorder=2))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=color, zorder=3)
    ax.text(x + w / 2, y + h - 0.40, subtitle, ha="center", va="top",
            fontsize=8.0, color="#333", zorder=3)
    ax.text(x + w / 2, y + 0.07, src, ha="center", va="bottom",
            fontsize=7.0, style="italic", color="#888", zorder=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "demos" / "pipeline_diagram.png"))
    args = ap.parse_args()

    box_w, box_h, gap_y = 3.4, 1.15, 0.45
    lane_gap = 0.7
    max_rows = max(len(items) for _, _, items in LANES)

    fig_w = len(LANES) * (box_w + lane_gap)
    fig_h = max_rows * (box_h + gap_y) + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.suptitle("Second Look - end-to-end pipeline", fontsize=15, fontweight="bold")

    centers = {}  # (lane_idx, row_idx) -> (cx_top, cx_bottom anchors)
    for li, (lane, color, items) in enumerate(LANES):
        x = li * (box_w + lane_gap) + 0.35
        ax.text(x + box_w / 2, fig_h - 0.55, lane, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)
        for ri, (title, subtitle, src) in enumerate(items):
            y = fig_h - 1.2 - (ri + 1) * (box_h + gap_y) + gap_y
            _box(ax, x, y, box_w, box_h, title, subtitle, src, color)
            centers[(li, ri)] = (x, y, box_w, box_h)
            # Vertical arrow within a lane.
            if ri > 0:
                px, py, pw, ph = centers[(li, ri - 1)]
                ax.add_patch(FancyArrowPatch(
                    (px + pw / 2, py), (x + box_w / 2, y + box_h),
                    arrowstyle="-|>", mutation_scale=14, color="#999", zorder=1))
        # Horizontal arrow to next lane (from last box of this lane to first of next).
        if li < len(LANES) - 1:
            lx, ly, lw, lh = centers[(li, len(items) - 1)]
            nx = (li + 1) * (box_w + lane_gap) + 0.35
            ny0 = fig_h - 1.2 - (box_h + gap_y) + gap_y  # first row y of next lane
            ax.add_patch(FancyArrowPatch(
                (lx + lw, ly + lh / 2), (nx, ny0 + box_h / 2),
                arrowstyle="-|>", mutation_scale=16, color="#555",
                connectionstyle="arc3,rad=0.0", zorder=1))

    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

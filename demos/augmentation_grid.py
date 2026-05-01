"""Render a grid of augmentations for visual inspection.

Extracted from the original Colab notebook. Runs each registered
augmentation once on the input image, plus a few random combos, and
writes an annotated grid PNG to the output directory. Each panel is
color-coded by quality_gate verdict so unrealistic outputs are obvious.

Usage:
    python -m demos.augmentation_grid --input path/to/img.png --out-dir ./
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt

from data_pipeline.augmentation import (
    AUGMENTATIONS,
    apply_random_combo,
    normalize,
)
from data_pipeline.quality import quality_gate


def show_grid(img_path: str, out_dir: str) -> str:
    """Render the augmentation grid for `img_path`, write a PNG, return its path."""
    # Mammograms are single-channel, so grayscale avoids unnecessary processing
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not load: {img_path}")

    h, w = img.shape
    # Resize only for visualization so multiple outputs can be compared clearly
    display_size = (int(w * 250 / h), 250)

    all_images = {"original": cv2.resize(img, display_size)}
    all_labels = {"original": "USABLE"}

    for name, fn in AUGMENTATIONS.items():
        # Skip mask handling to keep outputs simple for visual inspection
        result = fn(img.copy())
        aug = normalize(result if not isinstance(result, tuple) else result[0])
        all_images[name] = cv2.resize(aug, display_size)
        all_labels[name] = quality_gate(aug)

    # Random combos reveal whether stacked augmentations remain realistic
    for i in range(3):
        combo, combo_name = apply_random_combo(img)
        all_images[f"combo_{i+1}"] = cv2.resize(combo, display_size)
        all_labels[f"combo_{i+1}"] = f"{quality_gate(combo)}\n({combo_name})"

    cols, n = 3, len(all_images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(22, rows * 6))
    axes = axes.flatten()

    last_i = 0
    for i, (name, im) in enumerate(all_images.items()):
        label = all_labels[name]
        color = (
            "green" if "USABLE" in label and "BORDERLINE" not in label
            else ("orange" if "BORDERLINE" in label else "red")
        )
        axes[i].imshow(im, cmap="gray")
        axes[i].set_title(f"{name}\n{label}", fontsize=9, color=color, fontweight="bold")
        axes[i].axis("off")
        last_i = i

    for j in range(last_i + 1, len(axes)):
        axes[j].axis("off")

    fname = os.path.basename(img_path)
    plt.suptitle(f"Second Look - Augmentation Grid | {fname}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{fname}_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a mammogram image (PNG/JPG).")
    parser.add_argument("--out-dir", default=".", help="Directory to write the grid PNG into.")
    args = parser.parse_args()

    out_path = show_grid(args.input, args.out_dir)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

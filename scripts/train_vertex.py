"""Vertex AI training entrypoint for the Second Look baseline.

Runs the full loop on a single Vertex worker:

  1. BUILD  — pull dataset metadata + images from GCS into a local cache on the
     VM (reuses ``scripts.build_dataset --use-gcs``), producing a training
     manifest with a ``split`` column.
  2. TRAIN  — train the MobileNetV2 baseline (reuses
     ``modeling.train.train_baseline``) against the local cache.
  3. UPLOAD — copy the best checkpoint to a ``gs://`` prefix.
  4. EVAL   — optionally run ``modeling.evaluate`` on the held-out test split.

Design choice — download vs. stream
------------------------------------
We DOWNLOAD images to the VM's local disk via the existing, tested build path
rather than streaming ``gs://`` URIs through ``tf.data``. In-region (us-east1)
reads are fast and free, the retriever already implements skip-if-cached, and
this reuses the exact code validated locally. Streaming would require rewiring
and re-validating the ``tf.data`` pipeline for marginal benefit at this scale.

Design choice — checkpoint upload
---------------------------------
Training writes checkpoints to a LOCAL directory, then this script uploads the
best ``.keras`` file to the ``gs://`` prefix with ``tf.io.gfile.copy``. This
sidesteps any uncertainty about Keras 3 saving a ``.keras`` archive directly to
a remote filesystem, and it exercises the service account's scoped write to
``gs://b2-foundation/second-look/checkpoints/`` explicitly. On Vertex the
attached service account is the default credential, so no key file is needed.

Auth
----
On Vertex, GCS reads (retriever) and writes (gfile.copy) both authenticate via
the attached service account through Application Default Credentials. Do NOT set
GOOGLE_APPLICATION_CREDENTIALS here; that quirk is local-Windows-only.

Typical usage (run as a module so the sibling packages import cleanly):
    python -m scripts.train_vertex \
        --datasets cbis --limit 40 --max-epochs 1 \
        --checkpoint-dir gs://b2-foundation/second-look/checkpoints/smoke-vertex \
        --run-eval
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.constants import INPUT_SIZE  # noqa: E402

# The training manifest's own column names (see data_pipeline.manifest).
# canonical_label is already the binary 0/1 int; image_local_path is an
# absolute path to the cached PNG on the VM.
IMAGE_COL = "image_local_path"
LABEL_COL = "canonical_label"
SPLIT_COL = "split"

DEFAULT_CHECKPOINT_DIR = "gs://b2-foundation/second-look/checkpoints/baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vertex AI training entrypoint: build -> train -> upload -> eval.",
    )
    # Build phase
    parser.add_argument("--datasets", nargs="+", default=["cbis"],
                        help="Datasets to build/train on (default: cbis).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the image manifest to N case folders (smoke runs).")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Concurrent image download workers.")
    parser.add_argument("--work-dir", default="/tmp/second-look",
                        help="Writable dir on the VM for the cache + manifests.")
    parser.add_argument("--skip-build", action="store_true",
                        help="Reuse an existing manifest under --work-dir "
                             "(skip the GCS download step).")
    # Train phase
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freeze-backbone", dest="freeze_backbone",
                        action="store_true", default=True,
                        help="Keep MobileNetV2 frozen (default; baseline floor).")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone",
                        action="store_false",
                        help="Fine-tune the backbone (use a low LR).")
    # Output
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR,
                        help="gs:// prefix for the uploaded best checkpoint. Must "
                             "be under gs://b2-foundation/second-look/checkpoints/ "
                             "for the training service account to have write access.")
    parser.add_argument("--run-eval", action="store_true",
                        help="Evaluate the best checkpoint on the test split.")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace, work_dir: Path) -> Path:
    """Run scripts.build_dataset --use-gcs; return the training manifest path."""
    manifest_path = work_dir / "manifest.csv"
    if args.skip_build:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"--skip-build set but no manifest at {manifest_path}"
            )
        print(f"[build] skipped; reusing {manifest_path}")
        return manifest_path

    cmd = [
        sys.executable, "-m", "scripts.build_dataset",
        "--use-gcs",
        "--datasets", *args.datasets,
        "--cache-dir", str(work_dir / "cache"),
        "--output-manifest", str(work_dir / "manifest_patients.csv"),
        "--output-image-manifest", str(manifest_path),
        "--max-workers", str(args.max_workers),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]

    print(f"[build] running: {' '.join(cmd)}")
    # Run from REPO_ROOT so `python -m scripts.build_dataset` resolves.
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    if not manifest_path.exists():
        raise FileNotFoundError(f"build did not produce {manifest_path}")
    return manifest_path


def load_splits(manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the manifest and return (train, val, test) with usable images only."""
    df = pd.read_csv(manifest_path)

    # Keep only rows whose image actually downloaded and exists on disk.
    df = df[df[IMAGE_COL].notna()].copy()
    df[IMAGE_COL] = df[IMAGE_COL].astype(str)
    df = df[df[IMAGE_COL].str.len() > 0]
    exists = df[IMAGE_COL].map(lambda p: Path(p).exists())
    dropped = int((~exists).sum())
    if dropped:
        print(f"[data] dropping {dropped} rows with missing image files")
    df = df[exists]

    # canonical_label is already binary; guard against surprises.
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    train_df = df[df[SPLIT_COL] == "train"].reset_index(drop=True)
    val_df = df[df[SPLIT_COL] == "val"].reset_index(drop=True)
    test_df = df[df[SPLIT_COL] == "test"].reset_index(drop=True)

    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        if part.empty:
            raise ValueError(
                f"{name} split is empty after filtering. With a small --limit, "
                f"raise it so every split has images."
            )
        pos = int((part[LABEL_COL] == 1).sum())
        print(f"[data] {name}: {len(part)} images ({pos} positive)")
    return train_df, val_df, test_df


def upload_checkpoint(local_ckpt_dir: Path, gcs_dir: str) -> str | None:
    """Copy the best checkpoint from the local dir to the gs:// prefix."""
    local_best = local_ckpt_dir / "best.keras"
    if not local_best.exists():
        print(f"[upload] no checkpoint at {local_best}; nothing to upload")
        return None
    gcs_best = gcs_dir.rstrip("/") + "/best.keras"
    tf.io.gfile.makedirs(gcs_dir)
    tf.io.gfile.copy(str(local_best), gcs_best, overwrite=True)
    print(f"[upload] checkpoint -> {gcs_best}")
    return gcs_best


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    local_ckpt_dir = work_dir / "checkpoints"

    manifest_path = build_dataset(args, work_dir)
    train_df, val_df, test_df = load_splits(manifest_path)

    # Deferred import: keep TF/Keras model construction out of the arg-parse path.
    from modeling.train import train_baseline

    print(f"[train] epochs={args.max_epochs} batch={args.batch_size} "
          f"freeze_backbone={args.freeze_backbone}")
    train_baseline(
        train_df,
        val_df,
        image_dir="",              # image_local_path is already absolute
        image_col=IMAGE_COL,
        label_col=LABEL_COL,
        input_size=INPUT_SIZE,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        checkpoint_dir=str(local_ckpt_dir),   # train locally, upload after
        freeze_backbone=args.freeze_backbone,
    )

    gcs_best = upload_checkpoint(local_ckpt_dir, args.checkpoint_dir)

    if args.run_eval and gcs_best is not None:
        from modeling.evaluate import evaluate_baseline
        model = tf.keras.models.load_model(str(local_ckpt_dir / "best.keras"))
        print("[eval] evaluating best checkpoint on the test split")
        evaluate_baseline(
            model, test_df, image_dir="",
            image_col=IMAGE_COL, label_col=LABEL_COL,
            input_size=INPUT_SIZE, batch_size=args.batch_size,
        )

    print("[done] training entrypoint complete.")


if __name__ == "__main__":
    main()

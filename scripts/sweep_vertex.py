"""Vertex AI HYPERPARAMETER SWEEP entrypoint for the Second Look baseline.

One Vertex job trains + evaluates MANY hyperparameter combinations in a single
run, so you don't submit (and pay to re-download the dataset for) each combo
one at a time. The expensive dataset build/download happens ONCE, then every
config in ``SWEEP_CONFIGS`` is trained against the same in-memory splits.

For each config it:
  1. trains the MobileNetV2 baseline with that config's hyperparameters,
  2. uploads the best checkpoint to ``<checkpoint-dir>/<config-name>/best.keras``
     (each config keeps its OWN checkpoint — nothing is overwritten),
  3. evaluates on the held-out test split (AUROC, macro-F1, operating point,
     calibration), and
  4. appends a row to ``<checkpoint-dir>/sweep_summary.csv`` so you can rank
     configs at a glance.

A failing config is caught, recorded (with its traceback in the run log), and
the sweep CONTINUES to the next one — a single bad combo never sinks the whole
run. The summary CSV is re-uploaded after every config, so a hard crash still
leaves the results collected so far.

>>> EDIT THE SWEEP HERE <<<
Change ``SWEEP_CONFIGS`` below to whatever grid you want. Each entry is one
model. Keys per config (all optional except ``name``):
    name            : unique label -> checkpoint subdir + summary row id
    mode            : "single" (default) or "two_phase" (see below)
    dropout_rate    : dropout before the head (default 0.3)
    batch_size      : per-config batch size (default: --batch-size CLI value)
    worth_weight_multiplier : extra positive-class weight (default 1.5). Pass 1.0
                      to disable the bias — recommended when fine-tuning, where
                      1.5x can tip the model into an all-positive collapse.

  mode="single" (one-shot) keys:
    freeze_backbone : True keeps MobileNetV2 frozen (head-only); False fine-tunes
    learning_rate   : Adam LR. ~1e-3 for frozen; ~1e-4/1e-5 if fine-tuning
    max_epochs      : per-config epoch cap (default: --max-epochs CLI value);
                      early stopping (val_auc, patience 7) usually halts sooner

  mode="two_phase" (converge head frozen, THEN unfreeze low-LR; BatchNorm kept
  frozen — the stable fine-tuning recipe) keys:
    phase1_lr / phase2_lr        : head LR (~1e-3) then fine-tune LR (~1e-5/1e-4)
    phase1_epochs / phase2_epochs: per-phase caps (early stopping halts sooner)

Run as a module so sibling packages import cleanly:
    python -m scripts.sweep_vertex \
        --datasets cbis rsna vindr \
        --checkpoint-dir gs://b2-foundation/second-look/checkpoints/sweep-01
"""

from __future__ import annotations

# Keep module-level imports to the stdlib only. Heavy imports (tensorflow,
# pandas) happen inside run() so an import-time failure is caught by main()'s
# handler and written to GCS, matching scripts.train_vertex.
import argparse
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the validated single-run building blocks so the sweep and the plain
# baseline share one code path (build, split loading, checkpoint upload, the
# GCS debug-log capture). Only the multi-config loop is new.
from scripts.train_vertex import (  # noqa: E402
    IMAGE_COL,
    LABEL_COL,
    _Tee,
    _gcs_upload_file,
    _write_debug_log,
    build_dataset,
    load_splits,
    upload_checkpoint,
)

DEFAULT_CHECKPOINT_DIR = "gs://b2-foundation/second-look/checkpoints/sweep"


# ---------------------------------------------------------------------------
# >>> EDIT THE SWEEP HERE <<<  (see the module docstring for the key meanings)
# ---------------------------------------------------------------------------
# FINALISTS for the full combined run. The subset sweeps (subset-01/02) narrowed
# the field to these two: the frozen head (stable, strong on small data) vs the
# two-phase fine-tune at p2lr=1e-5 (the winner among fine-tunes: AUROC 0.940,
# macro-F1 0.919, no collapse). On the subset they were within noise; the full
# ~55k-image run is the decider, where fine-tuning has the data to pull ahead.
SWEEP_CONFIGS: list[dict] = [
    # Reference floor: frozen MobileNetV2 head.
    {"name": "frozen_lr1e-3", "mode": "single", "freeze_backbone": True,
     "learning_rate": 1e-3, "dropout_rate": 0.3},

    # Two-phase fine-tune: converge the head (frozen) then unfreeze the backbone
    # at a low LR with BatchNorm kept frozen — the stable recipe validated in
    # subset-02. The one most likely to beat the floor once given full data.
    {"name": "twophase_p2lr1e-5", "mode": "two_phase",
     "phase1_lr": 1e-3, "phase2_lr": 1e-5, "phase1_epochs": 12, "phase2_epochs": 15,
     "dropout_rate": 0.3},
]


# Columns collected per config into sweep_summary.csv (ordered for readability).
SUMMARY_COLUMNS = [
    "name", "status", "mode", "freeze_backbone", "learning_rate", "dropout_rate",
    "worth_weight_multiplier", "batch_size", "max_epochs", "epochs_ran",
    "auroc", "macro_f1", "worth_sensitivity", "passed_floor",
    "op_threshold", "op_sensitivity", "op_specificity",
    "brier", "ece", "checkpoint", "duration_sec", "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vertex AI hyperparameter sweep: build once -> train/eval each config.",
    )
    # Build phase (mirrors scripts.train_vertex so build_dataset/load_splits work).
    parser.add_argument("--datasets", nargs="+", default=["cbis"],
                        help="Datasets to build/train on (e.g. cbis rsna vindr).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the image manifest to N per dataset (smoke/subset sweeps).")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Concurrent image download workers.")
    parser.add_argument("--work-dir", default="/tmp/second-look",
                        help="Writable dir on the VM for the cache + manifests.")
    parser.add_argument("--skip-build", action="store_true",
                        help="Reuse an existing manifest under --work-dir.")
    # Train-phase DEFAULTS (a config may override batch_size / max_epochs).
    parser.add_argument("--max-epochs", type=int, default=20,
                        help="Default per-config epoch cap (early stopping halts sooner).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Default per-config batch size.")
    # Output
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR,
                        help="gs:// prefix. Each config writes <dir>/<name>/best.keras; "
                             "the summary lands at <dir>/sweep_summary.csv. Must be under "
                             "gs://b2-foundation/second-look/checkpoints/ for the training SA.")
    return parser.parse_args()


def _upload_summary(rows: list[dict], checkpoint_dir: str, work_dir: Path) -> None:
    """Write the running summary to CSV and upload it (best effort, incremental)."""
    import pandas as pd
    df = pd.DataFrame(rows).reindex(columns=SUMMARY_COLUMNS)
    local = work_dir / "sweep_summary.csv"
    df.to_csv(local, index=False)
    dest = checkpoint_dir.rstrip("/") + "/sweep_summary.csv"
    try:
        if dest.startswith("gs://"):
            _gcs_upload_file(local, dest)
        else:
            import shutil
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, dest)
        print(f"[sweep] summary -> {dest}")
    except Exception as exc:  # never let summary I/O kill the sweep
        print(f"[sweep] could not upload summary: {exc}")


def _run_one_config(cfg: dict, args, train_df, val_df, test_df, work_dir: Path) -> dict:
    """Train + evaluate a single config. Returns one summary row dict."""
    import tensorflow as tf
    from config.constants import INPUT_SIZE
    from modeling.train import train_baseline, train_baseline_two_phase
    from modeling.evaluate import evaluate_baseline

    name = cfg["name"]
    mode = cfg.get("mode", "single")
    dropout = float(cfg.get("dropout_rate", 0.3))
    batch_size = int(cfg.get("batch_size", args.batch_size))
    worth_mult = cfg.get("worth_weight_multiplier")  # None -> default 1.5

    row = {
        "name": name, "status": "running", "mode": mode,
        "dropout_rate": dropout, "batch_size": batch_size,
        "worth_weight_multiplier": worth_mult,
    }

    local_ckpt_dir = work_dir / "checkpoints" / name
    gcs_config_dir = args.checkpoint_dir.rstrip("/") + "/" + name
    start = time.time()

    print("\n" + "#" * 70)
    if mode == "two_phase":
        p1 = int(cfg.get("phase1_epochs", 10))
        p2 = int(cfg.get("phase2_epochs", 10))
        p1_lr = float(cfg.get("phase1_lr", 1e-3))
        p2_lr = float(cfg.get("phase2_lr", 1e-5))
        row.update({"freeze_backbone": False, "learning_rate": f"{p1_lr}->{p2_lr}",
                    "max_epochs": f"{p1}+{p2}"})
        print(f"[sweep] CONFIG '{name}' [TWO-PHASE]: p1_lr={p1_lr}({p1}ep) -> "
              f"p2_lr={p2_lr}({p2}ep) dropout={dropout} batch={batch_size} "
              f"worth_mult={worth_mult}")
        print("#" * 70)
        history = train_baseline_two_phase(
            train_df, val_df,
            image_dir="", image_col=IMAGE_COL, label_col=LABEL_COL,
            input_size=INPUT_SIZE, batch_size=batch_size,
            phase1_epochs=p1, phase2_epochs=p2, phase1_lr=p1_lr, phase2_lr=p2_lr,
            dropout_rate=dropout, checkpoint_dir=str(local_ckpt_dir),
            worth_weight_multiplier=worth_mult,
        )
    else:
        freeze = bool(cfg.get("freeze_backbone", True))
        lr = float(cfg.get("learning_rate", 1e-3))
        max_epochs = int(cfg.get("max_epochs", args.max_epochs))
        row.update({"freeze_backbone": freeze, "learning_rate": lr,
                    "max_epochs": max_epochs})
        print(f"[sweep] CONFIG '{name}' [SINGLE]: freeze={freeze} lr={lr} "
              f"dropout={dropout} batch={batch_size} max_epochs={max_epochs} "
              f"worth_mult={worth_mult}")
        print("#" * 70)
        history = train_baseline(
            train_df, val_df,
            image_dir="", image_col=IMAGE_COL, label_col=LABEL_COL,
            input_size=INPUT_SIZE, batch_size=batch_size, max_epochs=max_epochs,
            checkpoint_dir=str(local_ckpt_dir),
            freeze_backbone=freeze, learning_rate=lr, dropout_rate=dropout,
            worth_weight_multiplier=worth_mult,
        )
    row["epochs_ran"] = len(history.history.get("loss", []))

    gcs_best = upload_checkpoint(local_ckpt_dir, gcs_config_dir)
    row["checkpoint"] = gcs_best or ""

    if gcs_best is not None:
        model = tf.keras.models.load_model(str(local_ckpt_dir / "best.keras"))
        print(f"[sweep] evaluating '{name}' on the test split")
        res = evaluate_baseline(
            model, test_df, image_dir="", image_col=IMAGE_COL, label_col=LABEL_COL,
            input_size=INPUT_SIZE, batch_size=batch_size, output_dir=gcs_config_dir,
        )
        op = res.get("operating_point") or {}
        row.update({
            "auroc": res.get("auroc"),
            "macro_f1": res.get("macro_f1"),
            "worth_sensitivity": res.get("worth_sensitivity"),
            "passed_floor": res.get("passed_safety_floor"),
            "op_threshold": op.get("threshold"),
            "op_sensitivity": op.get("sensitivity"),
            "op_specificity": op.get("specificity"),
            "brier": res.get("brier_score"),
            "ece": res.get("ece"),
        })
        del model
    else:
        print(f"[sweep] '{name}' produced no checkpoint; skipping eval")

    # Release graph/memory before the next config builds a fresh model.
    tf.keras.backend.clear_session()
    row["duration_sec"] = round(time.time() - start, 1)
    row["status"] = "ok"
    return row


def run(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] {len(SWEEP_CONFIGS)} configs -> {args.checkpoint_dir}")
    print(f"[sweep] configs: {[c['name'] for c in SWEEP_CONFIGS]}")

    # Build the dataset ONCE and reuse the splits across every config.
    manifest_path = build_dataset(args, work_dir)
    train_df, val_df, test_df = load_splits(manifest_path)

    names = [c["name"] for c in SWEEP_CONFIGS]
    if len(names) != len(set(names)):
        raise ValueError(f"SWEEP_CONFIGS names must be unique; got {names}")

    rows: list[dict] = []
    for cfg in SWEEP_CONFIGS:
        try:
            row = _run_one_config(cfg, args, train_df, val_df, test_df, work_dir)
        except Exception:
            tb = traceback.format_exc()
            print(f"[sweep] CONFIG '{cfg.get('name')}' FAILED:\n{tb}")
            row = {
                "name": cfg.get("name"), "status": "failed",
                "freeze_backbone": cfg.get("freeze_backbone"),
                "learning_rate": cfg.get("learning_rate"),
                "dropout_rate": cfg.get("dropout_rate"),
                "error": tb.strip().splitlines()[-1] if tb.strip() else "unknown",
            }
        rows.append(row)
        _upload_summary(rows, args.checkpoint_dir, work_dir)  # incremental durability

    print("\n[sweep] all configs done. Summary:")
    ok = [r for r in rows if r.get("status") == "ok"]
    ranked = sorted(ok, key=lambda r: (r.get("auroc") is None, -(r.get("auroc") or 0)))
    for r in ranked:
        print(f"  {r['name']:22s} AUROC={r.get('auroc')} macroF1={r.get('macro_f1')} "
              f"WORTHsens={r.get('worth_sensitivity')} floor={r.get('passed_floor')}")
    failed = [r["name"] for r in rows if r.get("status") == "failed"]
    if failed:
        print(f"[sweep] FAILED configs: {failed}")
    print("[done] sweep entrypoint complete.")


def main() -> None:
    """Parse args and run the sweep, capturing all output to a GCS debug log."""
    import faulthandler
    import io

    faulthandler.enable()
    args = parse_args()

    _write_debug_log(args.checkpoint_dir, "STARTED: sweep entrypoint reached main()\n")

    buffer = io.StringIO()
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(real_stdout, buffer)
    sys.stderr = _Tee(real_stderr, buffer)
    try:
        run(args)
    except Exception:
        buffer.write("\n" + traceback.format_exc())
        sys.stdout, sys.stderr = real_stdout, real_stderr
        _write_debug_log(args.checkpoint_dir, buffer.getvalue())
        raise
    else:
        sys.stdout, sys.stderr = real_stdout, real_stderr
        _write_debug_log(args.checkpoint_dir, buffer.getvalue())


if __name__ == "__main__":
    main()

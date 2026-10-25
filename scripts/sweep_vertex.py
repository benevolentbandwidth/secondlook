"""Vertex AI HYPERPARAMETER SWEEP entrypoint for the Second Look baseline.

One Vertex job trains + evaluates MANY hyperparameter combinations in a single
run, so you don't submit (and pay to re-download the dataset for) each combo
one at a time. The expensive dataset build/download happens ONCE, then every
config in the selected sweep is trained against the same in-memory splits.

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

>>> EDIT THE SWEEPS HERE <<<
``SWEEPS`` below maps a NAME to a list of configs; pick one at submit time with
``--sweep NAME``. Keeping past sweeps in the dict (with their results in the
comment above each) means one staged package serves every job, and concurrent
jobs cannot race over a single staged tarball. Each entry is one model. Keys
per config (all optional except ``name``):
    name            : unique label -> checkpoint subdir + summary row id
    mode            : "single" (default) or "two_phase" (see below)
    dropout_rate    : dropout before the head (default 0.3)
    batch_size      : per-config batch size (default: --batch-size CLI value)
    worth_weight_multiplier : extra positive-class weight (default 1.5). Pass 1.0
                      to disable the bias — recommended when fine-tuning, where
                      1.5x can tip the model into an all-positive collapse.

  CLASS-IMBALANCE keys (the levers against the low-precision weak point):
    loss            : "bce" (default) or "focal"
    focal_gamma     : focal focusing strength (default 2.0; 0 = no focusing)
    focal_alpha     : focal's INTERNAL positive weight in [0,1] (default None =
                      off). Set it only together with use_class_weights=False.
    use_class_weights : False turns Keras class_weight off entirely (default
                      True). At the real ~6%-positive distribution the balanced
                      weight is ~8x on positives; stacking that under focal
                      alpha double-counts the imbalance and costs precision.
    max_neg_per_pos : per-dataset cap on negatives per positive in the TRAIN
                      split (default None = no rebalancing). Rebalances the
                      dataset MIX (RSNA's 54k mostly-negative images otherwise
                      swamp CBIS+VinDr). Only ever drops negatives; val/test are
                      never touched, so reported metrics stay honest.
    dataset_fractions : {"rsna": 0.3} — keep this share of that dataset's TRAIN
                      negatives. Applied before max_neg_per_pos.
    seed            : random seed for model init + shuffling (default 42).
                      Training is otherwise UNSEEDED-by-default history: config
                      SEED only ever touched data splitting, which is why the
                      same recipe scored AUROC 0.810 and 0.837 on identical
                      splits. Vary it deliberately to measure the noise floor.
    cache           : keep preprocessed images in RAM between epochs (default
                      True). Set False only if the cache would not fit.
    input_size      : (H, W) override, e.g. (320, 320). Default 224x224 from
                      config.constants. NOTE: INPUT_SIZE is a "fixed decision"
                      in CLAUDE.md with TF Lite / on-device implications —
                      changing it is a deliberate act, not a free knob, and the
                      in-memory tf.data cache grows with the square of it.

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


# Each entry is one NAMED sweep. Pick it at submit time with --sweep NAME, so a
# single staged package can serve several jobs and two concurrent jobs can never
# race over one staged tarball.
SWEEPS: dict[str, list[dict]] = {}

# The two-phase recipe every sweep below builds on (full-gpu-01's winner).
_TWO_PHASE = {
    "mode": "two_phase", "phase1_lr": 1e-3, "phase2_lr": 1e-5,
    "phase1_epochs": 12, "phase2_epochs": 15, "dropout_rate": 0.3,
}

# ---------------------------------------------------------------------------
# "imbalance" — DONE (full-gpu-02-imbalance, 2026-08-20). Kept for the record.
# RESULT: every lever LOST to the plain BCE + class-weighting control.
#   tp_bce_cw 0.674 spec / 0.133 prec | bce_nocw 0.630 / 0.119
#   focal_g3 0.612 / 0.114 | focal_g2 0.601 / 0.111 | mix10 0.524 / 0.095
# Loss reweighting moves the boundary but cannot separate classes better, so it
# could not fix the false-alarm problem. Calibration WAS a big win (ECE ~0.002).
# ---------------------------------------------------------------------------
SWEEPS["imbalance"] = [
    {"name": "tp_bce_cw", **_TWO_PHASE, "loss": "bce"},
    {"name": "tp_bce_nocw", **_TWO_PHASE, "loss": "bce", "use_class_weights": False},
    {"name": "tp_focal_g2_a25", **_TWO_PHASE, "loss": "focal", "focal_gamma": 2.0,
     "focal_alpha": 0.25, "use_class_weights": False},
    {"name": "tp_focal_g3_a25", **_TWO_PHASE, "loss": "focal", "focal_gamma": 3.0,
     "focal_alpha": 0.25, "use_class_weights": False},
    {"name": "tp_focal_g2_a25_mix10", **_TWO_PHASE, "loss": "focal", "focal_gamma": 2.0,
     "focal_alpha": 0.25, "use_class_weights": False, "max_neg_per_pos": 10.0},
]

# ---------------------------------------------------------------------------
# "seeds" — MEASURE THE NOISE FLOOR. Three runs of the IDENTICAL champion
# recipe, differing only by random seed.
#
# Why this is worth a whole job: training was never seeded (config.SEED only
# ever touched data splitting), and the same recipe on the SAME splits scored
# AUROC 0.810 in full-gpu-01 and 0.837 in full-gpu-02 — a spread WIDER than the
# gaps between the five imbalance configs. Until we know the noise floor, no
# single-run comparison in this project can be trusted, including that one.
#
# Read the SPREAD, not the winner: the std dev of op_specificity across these
# three is the smallest difference any future sweep may claim as real.
# ---------------------------------------------------------------------------
SWEEPS["seeds"] = [
    {"name": f"tp_bce_cw_seed{seed}", **_TWO_PHASE, "loss": "bce", "seed": seed}
    for seed in (42, 1337, 2024)
]

# ---------------------------------------------------------------------------
# "resolution" — the top remaining hypothesis. The imbalance sweep showed the
# problem is not WHERE the boundary sits but that 224x224 may not carry enough
# signal to separate subtle findings at all (a known ceiling on RSNA).
#
# INPUT_SIZE is a "fixed decision" in CLAUDE.md with TF Lite / on-device
# implications — this run is the evidence for or against changing it, not a
# change to it. The 224 control is included so the comparison is PAIRED
# (same job, same splits, same seed) rather than across runs.
#
# Memory: the in-memory tf.data cache is ~4 bytes * H * W per image over 54k
# train + 11k val images => ~17 GB at 224, ~27 GB at 288, ~34 GB at 320. The
# VM (n1-standard-16) has 60 GB, so 320 is the safe ceiling; 384 (~48 GB) is
# not attempted. `cache: False` is the escape hatch if one still OOMs, at the
# cost of re-running CLAHE every epoch.
# ---------------------------------------------------------------------------
SWEEPS["resolution"] = [
    {"name": "tp_bce_cw_res224", **_TWO_PHASE, "loss": "bce", "input_size": (224, 224)},
    {"name": "tp_bce_cw_res288", **_TWO_PHASE, "loss": "bce", "input_size": (288, 288)},
    {"name": "tp_bce_cw_res320", **_TWO_PHASE, "loss": "bce", "input_size": (320, 320)},
]

DEFAULT_SWEEP = "imbalance"

# Every key a config may set. A sweep is validated against this BEFORE the
# dataset build, because the alternative is discovering a typo ("focal_gama")
# after it has silently done nothing for a 20-hour GPU job.
KNOWN_CONFIG_KEYS = {
    "name", "mode", "dropout_rate", "batch_size", "worth_weight_multiplier",
    "loss", "focal_gamma", "focal_alpha", "use_class_weights",
    "max_neg_per_pos", "dataset_fractions", "input_size", "seed", "cache",
    "freeze_backbone", "learning_rate", "max_epochs",
    "phase1_lr", "phase2_lr", "phase1_epochs", "phase2_epochs",
}


def validate_sweep(configs: list[dict], sweep_name: str = "<sweep>") -> None:
    """Fail fast on a malformed sweep: unknown keys, dupe names, bad modes.

    Raises ValueError rather than warning -- an unrecognized key means the run
    would not test what its author thought it tested, which is worse than not
    running at all.
    """
    if not configs:
        raise ValueError(f"Sweep '{sweep_name}' is empty.")

    names = [c.get("name") for c in configs]
    if any(not n for n in names):
        raise ValueError(f"Sweep '{sweep_name}': every config needs a 'name'.")
    if len(names) != len(set(names)):
        raise ValueError(
            f"Sweep '{sweep_name}': config names must be unique; got {names}."
        )

    for cfg in configs:
        unknown = set(cfg) - KNOWN_CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"Sweep '{sweep_name}', config '{cfg.get('name')}': unknown key(s) "
                f"{sorted(unknown)}. Known keys: {sorted(KNOWN_CONFIG_KEYS)}."
            )
        mode = cfg.get("mode", "single")
        if mode not in ("single", "two_phase"):
            raise ValueError(
                f"Sweep '{sweep_name}', config '{cfg.get('name')}': mode must be "
                f"'single' or 'two_phase'; got '{mode}'."
            )


# Columns collected per config into sweep_summary.csv (ordered for readability:
# identity -> knobs -> headline metrics -> operating point -> calibration).
SUMMARY_COLUMNS = [
    "name", "status", "mode", "freeze_backbone", "learning_rate", "dropout_rate",
    "worth_weight_multiplier", "batch_size", "max_epochs", "epochs_ran",
    # Imbalance knobs under test.
    "loss", "focal_gamma", "focal_alpha", "use_class_weights", "seed", "cache",
    "max_neg_per_pos", "dataset_fractions", "input_size",
    "train_images", "train_positives", "train_pos_rate", "train_mix",
    # Headline metrics.
    "auroc", "macro_f1", "worth_sensitivity", "passed_floor",
    # The operating point at the 0.80 floor — op_precision and op_specificity
    # are THE columns this sweep is trying to move.
    "op_threshold", "op_sensitivity", "op_specificity", "op_precision",
    "op_worth_f1",
    # Calibration, before and after temperature scaling fitted on val.
    "brier", "ece", "temperature", "ece_calibrated",
    "checkpoint", "duration_sec", "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vertex AI hyperparameter sweep: build once -> train/eval each config.",
    )
    # Build phase (mirrors scripts.train_vertex so build_dataset/load_splits work).
    parser.add_argument("--sweep", default=DEFAULT_SWEEP, choices=sorted(SWEEPS),
                        help="Which named sweep in SWEEPS to run.")
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


def _calibrate(
    model, val_df, eval_result: dict, batch_size: int, input_size, gcs_config_dir: str
) -> dict:
    """Fit temperature scaling on VAL, report its effect on the TEST ECE.

    Fitting on val and applying to test is the only honest order — a temperature
    fitted on test would report a calibration the model does not have. Because
    the transform is monotonic, AUROC and the operating point at the sensitivity
    floor are unchanged, so this can never trade away WORTH sensitivity.

    Failures here are caught and recorded: a calibration problem must not throw
    away a config's trained checkpoint and its real metrics.
    """
    import numpy as np  # heavy imports stay inside functions (see module note)

    from modeling.calibrate import (
        apply_temperature,
        expected_calibration_error,
        fit_temperature,
    )
    from modeling.train import _build_dataset

    try:
        val_ds = _build_dataset(
            val_df, "", IMAGE_COL, LABEL_COL, input_size, batch_size, shuffle=False
        )
        val_probs = model.predict(val_ds, verbose=0).ravel()
        val_labels = np.asarray([int(y) for y in val_df[LABEL_COL]])

        temperature = fit_temperature(val_labels, val_probs)

        test_probs = eval_result["probabilities"]
        test_labels = eval_result["true_labels"]
        ece_cal = expected_calibration_error(
            test_labels, apply_temperature(test_probs, temperature)
        )
        print(f"[calibrate] test ECE {eval_result.get('ece'):.4f} -> {ece_cal:.4f} "
              f"(T={temperature:.4f} fitted on val)")
        _write_calibration_json(gcs_config_dir, temperature, eval_result, ece_cal)
        return {"temperature": round(float(temperature), 4),
                "ece_calibrated": round(float(ece_cal), 4)}
    except Exception as exc:
        print(f"[calibrate] SKIPPED (calibration failed): {exc}")
        return {"temperature": None, "ece_calibrated": None}


def _write_calibration_json(
    gcs_config_dir: str, temperature: float, eval_result: dict, ece_cal: float
) -> None:
    """Persist the temperature beside the checkpoint so the UX layer can use it.

    label_mapper.confidence_to_tier's thresholds are placeholders "pending
    calibration"; this file is what unblocks setting them for real.
    """
    import json
    import tempfile

    payload = {
        "temperature": float(temperature),
        "method": "temperature_scaling",
        "fitted_on": "val split",
        "ece_before": float(eval_result.get("ece")),
        "ece_after": float(ece_cal),
        "note": "Apply as sigmoid(logit(p) / T). Monotonic: AUROC and the "
                "operating point at the sensitivity floor are unchanged.",
    }
    try:
        local = Path(tempfile.gettempdir()) / "calibration.json"
        local.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        dest = gcs_config_dir.rstrip("/") + "/calibration.json"
        if dest.startswith("gs://"):
            _gcs_upload_file(local, dest)
        else:
            Path(gcs_config_dir).mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copyfile(local, dest)
        print(f"[calibrate] temperature -> {dest}")
    except Exception as exc:
        print(f"[calibrate] could not write calibration.json: {exc}")


def _run_one_config(cfg: dict, args, train_df, val_df, test_df, work_dir: Path) -> dict:
    """Train + evaluate a single config. Returns one summary row dict."""
    import tensorflow as tf
    from config.constants import INPUT_SIZE, SEED
    from modeling.dataset_mix import rebalance_train_mix, train_mix_stats
    from modeling.train import train_baseline, train_baseline_two_phase
    from modeling.evaluate import evaluate_baseline

    name = cfg["name"]
    mode = cfg.get("mode", "single")
    dropout = float(cfg.get("dropout_rate", 0.3))
    batch_size = int(cfg.get("batch_size", args.batch_size))
    worth_mult = cfg.get("worth_weight_multiplier")  # None -> default 1.5

    # Imbalance knobs.
    loss = str(cfg.get("loss", "bce"))
    focal_gamma = float(cfg.get("focal_gamma", 2.0))
    focal_alpha = cfg.get("focal_alpha")  # None -> focal class balancing off
    use_class_weights = bool(cfg.get("use_class_weights", True))
    max_neg_per_pos = cfg.get("max_neg_per_pos")
    dataset_fractions = cfg.get("dataset_fractions")
    input_size = tuple(cfg.get("input_size", INPUT_SIZE))
    seed = int(cfg.get("seed", SEED))
    cache = bool(cfg.get("cache", True))

    # Seed EVERYTHING (python, numpy, tf) before the model is built, so a config
    # is reproducible and two configs differing only in a knob start from the
    # same initial weights -- a paired comparison. Training was historically
    # unseeded, which is why one recipe scored AUROC 0.810 and 0.837 on the same
    # splits; the "seeds" sweep exists to quantify exactly that.
    tf.keras.utils.set_random_seed(seed)

    row = {
        "name": name, "status": "running", "mode": mode,
        "dropout_rate": dropout, "batch_size": batch_size,
        "worth_weight_multiplier": worth_mult,
        "loss": loss, "focal_gamma": focal_gamma if loss == "focal" else None,
        "focal_alpha": focal_alpha if loss == "focal" else None,
        "use_class_weights": use_class_weights,
        "max_neg_per_pos": max_neg_per_pos,
        "dataset_fractions": str(dataset_fractions) if dataset_fractions else None,
        "input_size": f"{input_size[0]}x{input_size[1]}",
        "seed": seed, "cache": cache,
    }

    # Rebalance the TRAIN mix only. val_df/test_df are deliberately untouched so
    # every reported metric stays on the real screening distribution.
    config_train_df = rebalance_train_mix(
        train_df,
        max_neg_per_pos=max_neg_per_pos,
        dataset_fractions=dataset_fractions,
    )
    row.update(train_mix_stats(config_train_df))

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
              f"worth_mult={worth_mult} seed={seed} "
              f"input={input_size[0]}x{input_size[1]} cache={cache}")
        print("#" * 70)
        history = train_baseline_two_phase(
            config_train_df, val_df,
            image_dir="", image_col=IMAGE_COL, label_col=LABEL_COL,
            input_size=input_size, batch_size=batch_size,
            phase1_epochs=p1, phase2_epochs=p2, phase1_lr=p1_lr, phase2_lr=p2_lr,
            dropout_rate=dropout, checkpoint_dir=str(local_ckpt_dir),
            worth_weight_multiplier=worth_mult,
            loss=loss, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
            use_class_weights=use_class_weights, cache=cache,
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
            config_train_df, val_df,
            image_dir="", image_col=IMAGE_COL, label_col=LABEL_COL,
            input_size=input_size, batch_size=batch_size, max_epochs=max_epochs,
            checkpoint_dir=str(local_ckpt_dir),
            freeze_backbone=freeze, learning_rate=lr, dropout_rate=dropout,
            worth_weight_multiplier=worth_mult,
            loss=loss, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
            use_class_weights=use_class_weights, cache=cache,
        )
    row["epochs_ran"] = len(history.history.get("loss", []))

    gcs_best = upload_checkpoint(local_ckpt_dir, gcs_config_dir)
    row["checkpoint"] = gcs_best or ""

    if gcs_best is not None:
        model = tf.keras.models.load_model(str(local_ckpt_dir / "best.keras"))
        print(f"[sweep] evaluating '{name}' on the test split")
        res = evaluate_baseline(
            model, test_df, image_dir="", image_col=IMAGE_COL, label_col=LABEL_COL,
            input_size=input_size, batch_size=batch_size, output_dir=gcs_config_dir,
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
            "op_precision": op.get("precision"),
            "op_worth_f1": op.get("worth_f1"),
            "brier": res.get("brier_score"),
            "ece": res.get("ece"),
        })
        row.update(
            _calibrate(model, val_df, res, batch_size, input_size, gcs_config_dir)
        )
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

    configs = SWEEPS[args.sweep]
    # Validate BEFORE the ~178 GB build, so a typo costs seconds not hours.
    validate_sweep(configs, args.sweep)
    print(f"[sweep] '{args.sweep}': {len(configs)} configs -> {args.checkpoint_dir}")
    print(f"[sweep] configs: {[c['name'] for c in configs]}")

    # Build the dataset ONCE and reuse the splits across every config.
    manifest_path = build_dataset(args, work_dir)
    train_df, val_df, test_df = load_splits(manifest_path)

    rows: list[dict] = []
    for cfg in configs:
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

    # Ranked by SPECIFICITY AT THE SENSITIVITY FLOOR, not AUROC. The floor is
    # already met by construction at that operating point, so among configs that
    # are equally safe, the better one is the one that raises fewer false
    # alarms. AUROC barely moves under loss reweighting (it reranks nothing), so
    # ranking on it would hide exactly the effect this sweep is measuring.
    def _key(r):
        spec = r.get("op_specificity")
        return (spec is None, -(spec or 0.0))

    print(f"  {'config':24s} {'AUROC':>6s} {'macroF1':>8s} {'spec@floor':>11s} "
          f"{'prec@floor':>11s} {'sens':>6s} {'ECE':>6s} {'ECEcal':>7s}")
    for r in sorted(ok, key=_key):
        def _f(key, width=6, nd=3):
            v = r.get(key)
            return f"{v:>{width}.{nd}f}" if isinstance(v, (int, float)) else f"{'n/a':>{width}s}"
        print(f"  {r['name']:24s} {_f('auroc')} {_f('macro_f1', 8)} "
              f"{_f('op_specificity', 11)} {_f('op_precision', 11)} "
              f"{_f('op_sensitivity')} {_f('ece')} {_f('ece_calibrated', 7)}")
    print("  (ranked by specificity at the 0.80 WORTH-sensitivity floor; "
          "the floor is met by construction at that operating point)")
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

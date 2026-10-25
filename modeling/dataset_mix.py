# Training-mix rebalancing for the combined CBIS + RSNA + VinDr manifest.
#
# The problem this solves: in the full combined train split, RSNA (~54k images,
# ~2% positive, and the hardest of the three) numerically swamps CBIS (~87%
# positive) and VinDr. The model therefore spends almost all of its gradient
# budget on easy RSNA negatives, and the loss compensates via a huge balanced
# class weight -- which is exactly the pressure that produces the observed weak
# point (poor WORTH precision / low specificity at the sensitivity floor).
#
# Rebalancing the MIX is a different lever from reweighting the LOSS: it changes
# which examples the model sees, not how much each one counts.
#
# TWO HARD RULES, both safety-motivated:
#   1. This is a TRAIN-SPLIT-ONLY transform. Val and test must keep the real
#      ~94%-negative screening distribution, or every reported number
#      (specificity, precision, calibration) becomes fiction. Callers must never
#      pass val_df/test_df here.
#   2. It only ever drops NEGATIVES. Dropping WORTH_SECOND_LOOK cases would
#      directly attack the primary metric and the #1 failure mode (false
#      reassurance). Positives are always kept in full.

from __future__ import annotations

import pandas as pd

from config.constants import SEED

# Manifest column names (see data_pipeline.manifest.IMAGE_MANIFEST_COLUMNS).
DATASET_COL = "dataset"
LABEL_COL = "canonical_label"


def rebalance_train_mix(
    train_df: pd.DataFrame,
    max_neg_per_pos: float | None = None,
    dataset_fractions: dict[str, float] | None = None,
    dataset_col: str = DATASET_COL,
    label_col: str = LABEL_COL,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Downsample negatives in the TRAIN split to rebalance the dataset mix.

    Applies, per dataset, in order:
      1. ``dataset_fractions`` -- keep this fraction of that dataset's NEGATIVES
         (e.g. ``{"rsna": 0.3}`` keeps 30% of RSNA negatives). Datasets absent
         from the dict are untouched.
      2. ``max_neg_per_pos`` -- cap that dataset's negatives at
         ``max_neg_per_pos x (its positive count)``.

    Both are applied within each dataset independently, so a dataset with no
    positives (none of ours, but be safe) is governed only by its fraction.

    Args:
        train_df: The TRAIN split only. Passing val/test corrupts evaluation.
        max_neg_per_pos: Cap on negatives per positive, per dataset. None = no cap.
        dataset_fractions: Per-dataset keep-fraction for negatives, in (0, 1].
        dataset_col: Column naming the source dataset.
        label_col: Binary label column (1 = WORTH_SECOND_LOOK).
        seed: Sampling seed, so a rebalanced mix is reproducible.
        verbose: Print the before/after composition (worth having in run logs).

    Returns:
        A new DataFrame (row order shuffled deterministically, index reset).
        Returns the input unchanged when both knobs are None.

    Raises:
        ValueError: on unknown dataset names, out-of-range fractions, a negative
            cap, a missing column, or labels outside {0, 1}.
    """
    if max_neg_per_pos is None and not dataset_fractions:
        return train_df

    for col in (dataset_col, label_col):
        if col not in train_df.columns:
            raise ValueError(
                f"rebalance_train_mix needs column '{col}'; got {list(train_df.columns)}."
            )

    labels = set(pd.unique(train_df[label_col]))
    unknown_labels = labels - {0, 1}
    if unknown_labels:
        raise ValueError(
            f"Unknown label values: {unknown_labels}. Expected binary {{0, 1}}."
        )

    if max_neg_per_pos is not None and float(max_neg_per_pos) < 0:
        raise ValueError(f"max_neg_per_pos must be >= 0; got {max_neg_per_pos}.")

    present = set(train_df[dataset_col].unique())
    fractions = dict(dataset_fractions or {})
    unknown_ds = set(fractions) - present
    if unknown_ds:
        raise ValueError(
            f"dataset_fractions names datasets not in the train split: "
            f"{sorted(unknown_ds)}. Present: {sorted(present)}."
        )
    for name, frac in fractions.items():
        if not 0 < float(frac) <= 1:
            raise ValueError(
                f"dataset_fractions['{name}'] must be in (0, 1]; got {frac}."
            )

    before = _composition(train_df, dataset_col, label_col)

    kept: list[pd.DataFrame] = []
    for ds_name, group in train_df.groupby(dataset_col, sort=True):
        positives = group[group[label_col] == 1]
        negatives = group[group[label_col] == 0]

        frac = fractions.get(ds_name)
        if frac is not None:
            negatives = negatives.sample(frac=float(frac), random_state=seed)

        if max_neg_per_pos is not None:
            cap = int(round(float(max_neg_per_pos) * len(positives)))
            if len(negatives) > cap:
                negatives = negatives.sample(n=cap, random_state=seed)

        kept.append(positives)
        kept.append(negatives)

    out = pd.concat(kept, ignore_index=True)
    # Shuffle so batches are not ordered by dataset then label. tf.data shuffles
    # too, but a deterministic shuffle here keeps any non-shuffled consumer sane.
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if verbose:
        after = _composition(out, dataset_col, label_col)
        print("\n[mix] train-split rebalancing (negatives downsampled; "
              f"max_neg_per_pos={max_neg_per_pos} fractions={dataset_fractions})")
        print(f"[mix]   BEFORE: {_format_composition(before)}")
        print(f"[mix]   AFTER : {_format_composition(after)}")

    return out


def train_mix_stats(
    df: pd.DataFrame,
    dataset_col: str = DATASET_COL,
    label_col: str = LABEL_COL,
) -> dict:
    """Flat stats about a split's composition, for the sweep summary CSV."""
    total = len(df)
    positives = int((df[label_col] == 1).sum())
    return {
        "train_images": total,
        "train_positives": positives,
        "train_pos_rate": round(positives / total, 4) if total else None,
        "train_mix": _format_composition(
            _composition(df, dataset_col, label_col), compact=True
        ),
    }


def _composition(df: pd.DataFrame, dataset_col: str, label_col: str) -> dict:
    comp = {}
    for ds_name, group in df.groupby(dataset_col, sort=True):
        comp[str(ds_name)] = (len(group), int((group[label_col] == 1).sum()))
    comp["TOTAL"] = (len(df), int((df[label_col] == 1).sum()))
    return comp


def _format_composition(comp: dict, compact: bool = False) -> str:
    parts = []
    for name, (n, pos) in comp.items():
        rate = f"{pos / n:.1%}" if n else "n/a"
        parts.append(f"{name}={n}({rate} pos)" if not compact else f"{name}:{n}/{rate}")
    return "  ".join(parts)

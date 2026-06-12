# Reproducible train/validation/test splits.

# Splits are stratified by the binary Second Look label so each partition
# reflects the same positive/negative rate as the full dataset.

# Default ratios: 70% train / 15% val / 15% test.
# The random seed is fixed so splits are reproducible across runs and machines.

# Usage:
#   from data_pipeline.splitter import split_dataset
#   train_df, val_df, test_df = split_dataset(df)            # defaults to "label"
#   train_df, val_df, test_df = split_dataset(df, label_column="my_col")

import pandas as pd
from sklearn.model_selection import train_test_split

from config.constants import SEED, TRAIN_RATIO, VAL_RATIO


def split_dataset(
    df: pd.DataFrame,
    label_column: str = "label",
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
    group_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into stratified train, validation, and test sets.

    Stratification is by the binary label column so that WORTH / NOT_WORTH
    proportions are preserved in every partition.

    When ``group_column`` is provided (e.g. ``"patient_id"``), the split is
    performed at the group level so every row of a given group lands in the
    same partition. Patient-level labels (positive if any of the group's rows
    is positive) are used as the stratification key to keep class balance even.

    Args:
        df: Full dataset DataFrame. Must contain the label_column.
        label_column: Name of the column holding the binary label (int 0/1).
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        seed: Random seed for reproducibility.
        group_column: If set, all rows sharing this column value go in the same
            split. Required for datasets like RSNA where one patient contributes
            multiple image rows and we must avoid patient leakage across splits.

    Returns:
        (train_df, val_df, test_df) — three non-overlapping DataFrames,
        each with a reset index.

    Raises:
        ValueError: If ratios don't sum to <= 1.0, label_column/group_column
            is missing, or any label class has fewer than 3 (groups or rows,
            depending on mode).
    """
    if label_column not in df.columns:
        raise ValueError(
            f"Column '{label_column}' not found. Available columns: {list(df.columns)}"
        )

    test_ratio = round(1.0 - train_ratio - val_ratio, 10)
    if test_ratio <= 0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0"
        )

    if group_column is not None:
        return _grouped_stratified_split(
            df, label_column, group_column, train_ratio, val_ratio, test_ratio, seed
        )

    _check_class_sizes(df, label_column)

    # First cut: split off test set.
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[label_column],
        random_state=seed,
    )

    # Second cut: split train_val into train and val.
    # val_ratio is expressed relative to the full dataset, so adjust for the
    # remaining fraction.
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df[label_column],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _grouped_stratified_split(
    df: pd.DataFrame,
    label_column: str,
    group_column: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Patient-grouped stratified split.

    Aggregates to one row per group with ``any_positive = max(label)``, splits
    those group keys 70/15/15 stratified on the group-level label, then
    propagates the split assignment back to every row.
    """
    if group_column not in df.columns:
        raise ValueError(
            f"Group column '{group_column}' not found. Available columns: {list(df.columns)}"
        )

    group_labels = (
        df.groupby(group_column)[label_column].max().reset_index(name="_group_label")
    )
    _check_class_sizes(group_labels, "_group_label")

    train_val_groups, test_groups = train_test_split(
        group_labels,
        test_size=test_ratio,
        stratify=group_labels["_group_label"],
        random_state=seed,
    )
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=val_ratio_adjusted,
        stratify=train_val_groups["_group_label"],
        random_state=seed,
    )

    def _select(groups: pd.DataFrame) -> pd.DataFrame:
        return df[df[group_column].isin(set(groups[group_column]))].reset_index(drop=True)

    return _select(train_groups), _select(val_groups), _select(test_groups)


def official_split_train_val(
    df: pd.DataFrame,
    label_column: str = "label",
    split_column: str = "split",
    val_fraction: float = 0.15,
    seed: int = SEED,
    group_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Honor a dataset's canonical train/test boundary; carve val out of train.

    Rows where ``split_column`` is 'test' go to the test partition verbatim.
    Rows where ``split_column`` is 'train' are stratified by label into train/val
    using ``val_fraction``. Used for CBIS-DDSM to match published baselines.

    When ``group_column`` is set, the train/val carve happens at the group
    level so a single group (e.g. one VinDr study with four images) cannot
    straddle the train/val boundary. Required for VinDr-Mammo, where the
    official split is at the study level and four images share each study.
    """
    for col in (label_column, split_column):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available columns: {list(df.columns)}"
            )

    test_df = df[df[split_column] == "test"].reset_index(drop=True)
    train_pool = df[df[split_column] == "train"].reset_index(drop=True)

    other = set(df[split_column].unique()) - {"train", "test"}
    if other:
        raise ValueError(
            f"Unexpected values in '{split_column}': {sorted(other)}. "
            "Expected only 'train' and 'test'."
        )

    if group_column is not None:
        if group_column not in train_pool.columns:
            raise ValueError(
                f"Group column '{group_column}' not found. "
                f"Available columns: {list(train_pool.columns)}"
            )
        group_labels = (
            train_pool.groupby(group_column)[label_column]
            .max()
            .reset_index(name="_group_label")
        )
        _check_class_sizes(group_labels, "_group_label")
        train_groups, val_groups = train_test_split(
            group_labels,
            test_size=val_fraction,
            stratify=group_labels["_group_label"],
            random_state=seed,
        )
        train_keys = set(train_groups[group_column])
        val_keys = set(val_groups[group_column])
        return (
            train_pool[train_pool[group_column].isin(train_keys)].reset_index(drop=True),
            train_pool[train_pool[group_column].isin(val_keys)].reset_index(drop=True),
            test_df,
        )

    _check_class_sizes(train_pool, label_column)

    train_df, val_df = train_test_split(
        train_pool,
        test_size=val_fraction,
        stratify=train_pool[label_column],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df,
    )


def summarize_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str = "label",
) -> pd.DataFrame:
    """Return a DataFrame summarizing label distribution across splits.

    Useful for verifying that stratification worked as expected before training.

    Args:
        train_df, val_df, test_df: Split DataFrames from split_dataset().
        label_column: Name of the label column.

    Returns:
        DataFrame with columns [label, train_n, val_n, test_n, train_pct, val_pct, test_pct].
    """
    rows = []
    all_labels = sorted(
        set(train_df[label_column]) | set(val_df[label_column]) | set(test_df[label_column])
    )
    for lbl in all_labels:
        train_n = (train_df[label_column] == lbl).sum()
        val_n = (val_df[label_column] == lbl).sum()
        test_n = (test_df[label_column] == lbl).sum()
        total = train_n + val_n + test_n
        rows.append({
            "label": lbl,
            "train_n": train_n,
            "val_n": val_n,
            "test_n": test_n,
            "train_pct": round(100 * train_n / total, 1) if total else 0,
            "val_pct": round(100 * val_n / total, 1) if total else 0,
            "test_pct": round(100 * test_n / total, 1) if total else 0,
        })
    return pd.DataFrame(rows)


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Write split CSVs to output_dir.

    Files written: train.csv, val.csv, test.csv.
    Directory must already exist.

    Args:
        train_df, val_df, test_df: Split DataFrames from split_dataset().
        output_dir: Path to an existing directory.
    """
    import os
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(output_dir, f"{name}.csv")
        split_df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_class_sizes(df: pd.DataFrame, label_column: str) -> None:
    """Raise if any label class is too small to stratify into three splits."""
    counts = df[label_column].value_counts()
    too_small = counts[counts < 3]
    if not too_small.empty:
        raise ValueError(
            f"The following labels have fewer than 3 samples and cannot be "
            f"stratified: {too_small.to_dict()}. "
            "Collect more data before splitting."
        )

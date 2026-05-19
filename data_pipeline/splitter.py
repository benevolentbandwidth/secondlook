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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into stratified train, validation, and test sets.

    Stratification is by the binary label column so that WORTH / NOT_WORTH
    proportions are preserved in every partition.

    Args:
        df: Full dataset DataFrame. Must contain the label_column.
        label_column: Name of the column holding the binary label (int 0/1).
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df) — three non-overlapping DataFrames,
        each with a reset index.

    Raises:
        ValueError: If ratios don't sum to <= 1.0, or label_column is missing,
                    or any label class has fewer than 3 samples (can't stratify).
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


def official_split_train_val(
    df: pd.DataFrame,
    label_column: str = "label",
    split_column: str = "split",
    val_fraction: float = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Honor a dataset's canonical train/test boundary; carve val out of train.

    Rows where ``split_column`` is 'test' go to the test partition verbatim.
    Rows where ``split_column`` is 'train' are stratified by label into train/val
    using ``val_fraction``. Used for CBIS-DDSM to match published baselines.
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

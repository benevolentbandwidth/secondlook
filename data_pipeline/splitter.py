# Reproducible train/validation/test splits

# Splits are stratified by concern tier (Low / Moderate / Elevated) so each
# partition reflects the same class distribution as the full dataset.

# Default ratios: 70% train / 15% val / 15% test.
# The random seed is fixed so splits are reproducible across runs and machines.

# Usage:
#   from data_pipeline.splitter import split_dataset
#   train_df, val_df, test_df = split_dataset(df, tier_column="concern_tier")

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
# Test ratio is the remainder: 1 - train - val = 0.15


def split_dataset(
    df: pd.DataFrame,
    tier_column: str = "concern_tier",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into stratified train, validation, and test sets.

    Stratification is by concern tier so that Low / Moderate / Elevated
    proportions are preserved in every partition.

    Args:
        df: Full dataset DataFrame. Must contain the tier_column.
        tier_column: Name of the column holding concern tier strings.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df) — three non-overlapping DataFrames,
        each with a reset index.

    Raises:
        ValueError: If ratios don't sum to <= 1.0, or tier_column is missing,
                    or any tier class has fewer than 3 samples (can't stratify).
    """
    if tier_column not in df.columns:
        raise ValueError(
            f"Column '{tier_column}' not found. Available columns: {list(df.columns)}"
        )

    test_ratio = round(1.0 - train_ratio - val_ratio, 10)
    if test_ratio <= 0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0"
        )

    _check_class_sizes(df, tier_column)

    # First cut: split off test set.
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[tier_column],
        random_state=seed,
    )

    # Second cut: split train_val into train and val.
    # val_ratio is expressed relative to the full dataset, so adjust for the
    # remaining fraction.
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df[tier_column],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def summarize_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tier_column: str = "concern_tier",
) -> pd.DataFrame:
    """Return a DataFrame summarizing tier distribution across splits.

    Useful for verifying that stratification worked as expected before training.

    Args:
        train_df, val_df, test_df: Split DataFrames from split_dataset().
        tier_column: Name of the tier column.

    Returns:
        DataFrame with columns [tier, train_n, val_n, test_n, train_pct, val_pct, test_pct].
    """
    rows = []
    for tier in sorted(set(train_df[tier_column]) | set(val_df[tier_column]) | set(test_df[tier_column])):
        train_n = (train_df[tier_column] == tier).sum()
        val_n = (val_df[tier_column] == tier).sum()
        test_n = (test_df[tier_column] == tier).sum()
        total = train_n + val_n + test_n
        rows.append({
            "tier": tier,
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

def _check_class_sizes(df: pd.DataFrame, tier_column: str) -> None:
    """Raise if any tier class is too small to stratify into three splits."""
    counts = df[tier_column].value_counts()
    too_small = counts[counts < 3]
    if not too_small.empty:
        raise ValueError(
            f"The following tiers have fewer than 3 samples and cannot be "
            f"stratified: {too_small.to_dict()}. "
            "Collect more data or merge underrepresented tiers before splitting."
        )

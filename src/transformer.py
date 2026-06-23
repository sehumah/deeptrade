from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_SIZE = 60

TRAIN_END = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2023-01-01")
VAL_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")


def create_sequences(
    df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_col: str = "TargetReturn",
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(df) < window_size:
        raise ValueError(f"Need at least {window_size} rows, got {len(df)}")

    feature_values = df[list(feature_columns)].to_numpy(dtype=np.float32)
    targets = df[target_col].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(df["Date"]).to_numpy()

    sequences = []
    sequence_targets = []
    sequence_dates = []

    for end_idx in range(window_size - 1, len(df)):
        start_idx = end_idx - window_size + 1
        sequences.append(feature_values[start_idx : end_idx + 1])
        sequence_targets.append(targets[end_idx])
        sequence_dates.append(dates[end_idx])

    return (
        np.stack(sequences),
        np.array(sequence_targets, dtype=np.float32),
        np.array(sequence_dates),
    )


def train_val_test_split(
    sequences: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    dates = pd.to_datetime(dates)

    train_mask = dates <= TRAIN_END
    val_mask = (dates >= VAL_START) & (dates <= VAL_END)
    test_mask = dates >= TEST_START

    def _split(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return sequences[mask], targets[mask], dates[mask].to_numpy()

    return {
        "train": _split(train_mask),
        "val": _split(val_mask),
        "test": _split(test_mask),
    }


if __name__ == "__main__":
    from src.features import FEATURE_COLUMNS, build_feature_dataset

    build_feature_dataset("SPY")
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "features" / "SPY_features.csv",
        parse_dates=["Date"],
    )

    X, y, dates = create_sequences(df, FEATURE_COLUMNS)
    splits = train_val_test_split(X, y, dates)

    print(f"X: {X.shape}, y: {y.shape}, dates: {dates.shape}")
    for name, (x_split, y_split, d_split) in splits.items():
        print(f"{name}: X={x_split.shape}, y={y_split.shape}, dates={d_split.shape}")

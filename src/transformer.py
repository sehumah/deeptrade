from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ReturnTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = WINDOW_SIZE,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


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

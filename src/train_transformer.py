from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import FEATURE_COLUMNS, FEATURES_DATA_DIR
from src.transformer import (
    ReturnTransformer,
    count_parameters,
    create_sequences,
    evaluate_forecast,
    train_val_test_split,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHECKPOINT_PATH = RESULTS_DIR / "transformer.pt"
LOG_PATH = RESULTS_DIR / "transformer_training_log.csv"
METRICS_PATH = RESULTS_DIR / "forecast_metrics.json"
PLOT_PATH = RESULTS_DIR / "forecast_eval.png"

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 10


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    _, _, n_features = X_train.shape
    scaler.fit(X_train.reshape(-1, n_features))
    return scaler


def transform_sequences(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n_samples, window, n_features = X.shape
    scaled = scaler.transform(X.reshape(-1, n_features))
    return scaled.reshape(n_samples, window, n_features).astype(np.float32)


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: ReturnTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            preds = model(X_batch)
            loss = criterion(preds, y_batch)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def train_transformer(
    ticker: str = "SPY",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    features_dir: Path = FEATURES_DATA_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
    log_path: Path = LOG_PATH,
) -> dict:
    feature_path = features_dir / f"{ticker}_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature data not found: {feature_path}")

    df = pd.read_csv(feature_path, parse_dates=["Date"])
    X, y, dates = create_sequences(df, FEATURE_COLUMNS)
    splits = train_val_test_split(X, y, dates)

    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]

    scaler = fit_scaler(X_train)
    X_train = transform_sequences(X_train, scaler)
    X_val = transform_sequences(X_val, scaler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReturnTransformer(
        input_dim=len(FEATURE_COLUMNS),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    train_loader = _make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    print(f"Training on {device} | parameters: {count_parameters(model)}")
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, criterion, None, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")

    model.load_state_dict(best_state)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "scaler": scaler,
        "feature_columns": list(FEATURE_COLUMNS),
        "input_dim": len(FEATURE_COLUMNS),
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dropout": dropout,
        "best_val_loss": best_val_loss,
        "ticker": ticker,
    }
    torch.save(checkpoint, checkpoint_path)

    log_df = pd.DataFrame(history)
    log_df.to_csv(log_path, index=False)

    print(f"Saved checkpoint -> {checkpoint_path}")
    print(f"Saved training log -> {log_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return {
        "checkpoint_path": checkpoint_path,
        "log_path": log_path,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def load_model_from_checkpoint(
    checkpoint_path: Path = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> tuple[ReturnTransformer, dict, torch.device]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    model = ReturnTransformer(
        input_dim=checkpoint["input_dim"],
        d_model=checkpoint["d_model"],
        nhead=checkpoint["nhead"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, device


def plot_forecast_evaluation(
    dates: np.ndarray,
    predictions: np.ndarray,
    actuals: np.ndarray,
    output_path: Path = PLOT_PATH,
) -> None:
    dates = pd.to_datetime(dates)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(dates, actuals, label="Actual", linewidth=1.2)
    axes[0].plot(dates, predictions, label="Predicted", linewidth=1.2, alpha=0.8)
    axes[0].set_title("Transformer Forecast: Predicted vs Actual Returns (Test Set)")
    axes[0].set_ylabel("Return")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(actuals, predictions, alpha=0.5, s=12)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    axes[1].set_title("Predicted vs Actual")
    axes[1].set_xlabel("Actual Return")
    axes[1].set_ylabel("Predicted Return")
    axes[1].grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_forecast_evaluation(
    ticker: str | None = None,
    features_dir: Path = FEATURES_DATA_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
    metrics_path: Path = METRICS_PATH,
    plot_path: Path = PLOT_PATH,
) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, checkpoint, device = load_model_from_checkpoint(checkpoint_path, device=None)
    ticker = ticker or checkpoint.get("ticker", "SPY")
    scaler: StandardScaler = checkpoint["scaler"]

    feature_path = features_dir / f"{ticker}_features.csv"
    df = pd.read_csv(feature_path, parse_dates=["Date"])
    X, y, dates = create_sequences(df, FEATURE_COLUMNS)
    splits = train_val_test_split(X, y, dates)

    X_test, y_test, test_dates = splits["test"]
    X_test = transform_sequences(X_test, scaler)

    results = evaluate_forecast(model, X_test, y_test, device=device)
    direction_accuracy = results["direction_accuracy"]
    target_direction_accuracy = 0.55

    metrics = {
        "ticker": ticker,
        "split": "test",
        "test_samples": int(len(y_test)),
        "rmse": results["rmse"],
        "mae": results["mae"],
        "direction_accuracy": direction_accuracy,
        "target_direction_accuracy": target_direction_accuracy,
        "meets_direction_target": direction_accuracy > target_direction_accuracy,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    plot_forecast_evaluation(
        test_dates,
        np.asarray(results["predictions"]),
        np.asarray(results["actuals"]),
        output_path=plot_path,
    )

    print(f"Test samples: {metrics['test_samples']}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"Direction accuracy: {direction_accuracy:.2%}")
    print(
        "Direction target (>55%): "
        f"{'met' if metrics['meets_direction_target'] else 'not met'}"
    )
    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved plot -> {plot_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the return Transformer")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate the saved model on the test split",
    )
    args = parser.parse_args()

    if args.evaluate:
        run_forecast_evaluation()
    else:
        train_transformer()

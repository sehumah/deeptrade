from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import FEATURE_COLUMNS, FEATURES_DATA_DIR
from src.train_transformer import (
    CHECKPOINT_PATH,
    load_model_from_checkpoint,
    transform_sequences,
)
from src.transformer import create_sequences, train_val_test_split

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CLASSIFIER_PATH = RESULTS_DIR / "signal_classifier.pt"
LABEL_DIST_PATH = RESULTS_DIR / "signal_label_distribution.json"
METRICS_PATH = RESULTS_DIR / "signal_metrics.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "signal_confusion_matrix.png"

CLASSIFIER_INDICATOR_COLUMNS = ("RSI", "MACD", "EMA20", "SMA20", "SMA50", "VolumeChange")
CLASS_NAMES = ("Sell", "Hold", "Buy")

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 10
DEFAULT_BUY_THRESH = 0.01
DEFAULT_SELL_THRESH = -0.01


def return_to_signal(
    return_value: float,
    buy_thresh: float = DEFAULT_BUY_THRESH,
    sell_thresh: float = DEFAULT_SELL_THRESH,
) -> int:
    """Map a return value to a signal class: 0=Sell, 1=Hold, 2=Buy."""
    if return_value > buy_thresh:
        return 2
    if return_value < sell_thresh:
        return 0
    return 1


def generate_labels(
    returns: np.ndarray,
    buy_thresh: float = DEFAULT_BUY_THRESH,
    sell_thresh: float = DEFAULT_SELL_THRESH,
) -> np.ndarray:
    vectorized = np.vectorize(
        lambda value: return_to_signal(value, buy_thresh=buy_thresh, sell_thresh=sell_thresh)
    )
    return vectorized(returns).astype(np.int64)


def compute_label_distribution(labels: np.ndarray) -> dict:
    counts = {
        CLASS_NAMES[class_id]: int(np.sum(labels == class_id))
        for class_id in range(len(CLASS_NAMES))
    }
    total = int(len(labels))
    distribution = {
        "counts": counts,
        "percentages": {
            name: round(count / total * 100, 2) if total else 0.0
            for name, count in counts.items()
        },
        "total": total,
    }
    return distribution


def save_label_distribution(
    labels: np.ndarray,
    output_path: Path = LABEL_DIST_PATH,
    split_name: str = "train",
) -> dict:
    distribution = compute_label_distribution(labels)
    payload = {"split": split_name, **distribution}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"Label distribution ({split_name}):")
    for name in CLASS_NAMES:
        count = distribution["counts"][name]
        pct = distribution["percentages"][name]
        print(f"  {name}: {count} ({pct}%)")

    return payload


class SignalClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _predict_transformer_returns(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            preds = model(batch).cpu().numpy().squeeze(-1)
            predictions.append(preds)

    return np.concatenate(predictions).astype(np.float32)


def _extract_indicator_features(df: pd.DataFrame, dates: np.ndarray) -> np.ndarray:
    date_index = pd.to_datetime(df["Date"])
    lookup = df.set_index(date_index)[list(CLASSIFIER_INDICATOR_COLUMNS)]
    aligned = lookup.loc[pd.to_datetime(dates)].to_numpy(dtype=np.float32)
    return aligned


def build_classifier_dataset(
    df: pd.DataFrame,
    transformer: nn.Module,
    feature_columns: list[str] | tuple[str, ...],
    scaler,
    device: torch.device,
    buy_thresh: float = DEFAULT_BUY_THRESH,
    sell_thresh: float = DEFAULT_SELL_THRESH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_seq, y_returns, dates = create_sequences(df, feature_columns)
    X_scaled = transform_sequences(X_seq, scaler)
    transformer_preds = _predict_transformer_returns(transformer, X_scaled, device)
    indicators = _extract_indicator_features(df, dates)

    X = np.column_stack([transformer_preds, indicators]).astype(np.float32)
    y = generate_labels(y_returns, buy_thresh=buy_thresh, sell_thresh=sell_thresh)
    return X, y, dates


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: SignalClassifier,
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
        y_batch = y_batch.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def train_classifier(
    ticker: str = "SPY",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    buy_thresh: float = DEFAULT_BUY_THRESH,
    sell_thresh: float = DEFAULT_SELL_THRESH,
    features_dir: Path = FEATURES_DATA_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
    classifier_path: Path = CLASSIFIER_PATH,
) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Transformer checkpoint not found: {checkpoint_path}")

    transformer, checkpoint, device = load_model_from_checkpoint(checkpoint_path)
    scaler = checkpoint["scaler"]
    ticker = ticker or checkpoint.get("ticker", "SPY")

    feature_path = features_dir / f"{ticker}_features.csv"
    df = pd.read_csv(feature_path, parse_dates=["Date"])

    X, y, dates = build_classifier_dataset(
        df,
        transformer,
        FEATURE_COLUMNS,
        scaler,
        device,
        buy_thresh=buy_thresh,
        sell_thresh=sell_thresh,
    )
    splits = train_val_test_split(
        X,
        y.astype(np.float32),
        dates,
    )

    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)

    save_label_distribution(y_train.astype(np.int64), split_name="train")

    input_dim = X_train.shape[1]
    model = SignalClassifier(input_dim=input_dim).to(device)
    train_loader = _make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print(f"Training signal classifier on {device}")
    print(f"Input dim: {input_dim} | Train: {len(X_train)} | Val: {len(X_val)}")

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, criterion, None, device)
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
        raise RuntimeError("Classifier training did not produce a checkpoint")

    model.load_state_dict(best_state)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": input_dim,
            "indicator_columns": list(CLASSIFIER_INDICATOR_COLUMNS),
            "buy_thresh": buy_thresh,
            "sell_thresh": sell_thresh,
            "best_val_loss": best_val_loss,
            "ticker": ticker,
        },
        classifier_path,
    )

    print(f"Saved classifier -> {classifier_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return {
        "classifier_path": classifier_path,
        "best_val_loss": best_val_loss,
        "input_dim": input_dim,
    }


def load_classifier_from_checkpoint(
    classifier_path: Path = CLASSIFIER_PATH,
    device: torch.device | None = None,
) -> tuple[SignalClassifier, dict, torch.device]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(classifier_path, weights_only=False)

    model = SignalClassifier(input_dim=checkpoint["input_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, device


def evaluate_classifier(
    model: SignalClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device | None = None,
) -> dict:
    device = device or torch.device("cpu")
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()

    y_true = np.asarray(y_test, dtype=np.int64)
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "predictions": y_pred.tolist(),
        "actuals": y_true.tolist(),
    }
    return metrics


def plot_confusion_matrix(
    confusion: np.ndarray,
    output_path: Path = CONFUSION_MATRIX_PATH,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set(
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="Actual",
        xlabel="Predicted",
        title="Signal Classifier Confusion Matrix (Test Set)",
    )

    threshold = confusion.max() / 2.0 if confusion.max() > 0 else 0.0
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            ax.text(
                col,
                row,
                format(confusion[row, col], "d"),
                ha="center",
                va="center",
                color="white" if confusion[row, col] > threshold else "black",
            )

    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_classifier_evaluation(
    ticker: str | None = None,
    features_dir: Path = FEATURES_DATA_DIR,
    transformer_path: Path = CHECKPOINT_PATH,
    classifier_path: Path = CLASSIFIER_PATH,
    metrics_path: Path = METRICS_PATH,
    confusion_matrix_path: Path = CONFUSION_MATRIX_PATH,
) -> dict:
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")

    transformer, transformer_ckpt, device = load_model_from_checkpoint(transformer_path)
    classifier, classifier_ckpt, device = load_classifier_from_checkpoint(classifier_path, device)

    ticker = ticker or classifier_ckpt.get("ticker", "SPY")
    buy_thresh = classifier_ckpt.get("buy_thresh", DEFAULT_BUY_THRESH)
    sell_thresh = classifier_ckpt.get("sell_thresh", DEFAULT_SELL_THRESH)

    feature_path = features_dir / f"{ticker}_features.csv"
    df = pd.read_csv(feature_path, parse_dates=["Date"])

    X, y, dates = build_classifier_dataset(
        df,
        transformer,
        FEATURE_COLUMNS,
        transformer_ckpt["scaler"],
        device,
        buy_thresh=buy_thresh,
        sell_thresh=sell_thresh,
    )
    splits = train_val_test_split(X, y.astype(np.float32), dates)

    X_test, y_test, test_dates = splits["test"]
    y_test = y_test.astype(np.int64)

    raw_metrics = evaluate_classifier(classifier, X_test, y_test, device=device)
    metrics = {
        "ticker": ticker,
        "split": "test",
        "test_samples": int(len(y_test)),
        "accuracy": raw_metrics["accuracy"],
        "precision_macro": raw_metrics["precision_macro"],
        "recall_macro": raw_metrics["recall_macro"],
        "f1_macro": raw_metrics["f1_macro"],
        "per_class": raw_metrics["per_class"],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    plot_confusion_matrix(np.array(raw_metrics["confusion_matrix"]), output_path=confusion_matrix_path)

    print(f"Test samples: {metrics['test_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved confusion matrix -> {confusion_matrix_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the signal classifier")
    parser.add_argument("--train", action="store_true", help="train the signal classifier")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate the saved classifier on the test split",
    )
    args = parser.parse_args()

    if args.evaluate:
        run_classifier_evaluation()
    elif args.train:
        train_classifier()
        run_classifier_evaluation()
    else:
        parser.print_help()

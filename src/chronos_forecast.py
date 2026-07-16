from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.features import FEATURE_COLUMNS, FEATURES_DATA_DIR
from src.transformer import TEST_START, WINDOW_SIZE, create_sequences, train_val_test_split

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PREDICTIONS_PATH = RESULTS_DIR / "chronos_predictions.csv"
METRICS_PATH = RESULTS_DIR / "forecast_metrics.json"
PLOT_PATH = RESULTS_DIR / "forecast_eval.png"

DEFAULT_MODEL_ID = "amazon/chronos-t5-tiny"
DEFAULT_BATCH_SIZE = 32


def resolve_device(device: str | None = None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_chronos_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
):
    from chronos import ChronosPipeline

    device = resolve_device(device)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return ChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=dtype,
    )


def _build_contexts(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> tuple[list[torch.Tensor], np.ndarray]:
    close = df["Close"].to_numpy(dtype=np.float64)
    contexts: list[torch.Tensor] = []
    current_closes: list[float] = []

    for end_idx in range(window_size - 1, len(df)):
        start_idx = end_idx - window_size + 1
        contexts.append(torch.tensor(close[start_idx : end_idx + 1], dtype=torch.float32))
        current_closes.append(close[end_idx])

    return contexts, np.asarray(current_closes, dtype=np.float64)


def _predict_returns_from_contexts(
    contexts: list[torch.Tensor],
    current_closes: np.ndarray,
    pipeline,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    predictions: list[np.ndarray] = []

    for start in range(0, len(contexts), batch_size):
        batch_contexts = contexts[start : start + batch_size]
        batch_current = current_closes[start : start + len(batch_contexts)]
        forecasts = pipeline.predict(batch_contexts, prediction_length=1)
        forecast_array = forecasts.numpy() if hasattr(forecasts, "numpy") else np.asarray(forecasts)
        median_prices = np.median(forecast_array, axis=1).squeeze(-1)
        batch_returns = (median_prices - batch_current) / batch_current
        predictions.append(batch_returns.astype(np.float32))

    return np.concatenate(predictions).astype(np.float32)


def generate_chronos_predictions(
    df: pd.DataFrame,
    *,
    window_size: int = WINDOW_SIZE,
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    pipeline=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate next-day return forecasts aligned with :func:`create_sequences`."""
    _, actual_returns, dates = create_sequences(df, FEATURE_COLUMNS, window_size=window_size)
    contexts, current_closes = _build_contexts(df, window_size=window_size)

    if pipeline is None:
        pipeline = load_chronos_pipeline(model_id=model_id, device=device)

    predicted_returns = _predict_returns_from_contexts(
        contexts,
        current_closes,
        pipeline,
        batch_size=batch_size,
    )
    return predicted_returns, actual_returns, dates


def save_predictions(
    predicted_returns: np.ndarray,
    dates: np.ndarray,
    output_path: Path = PREDICTIONS_PATH,
    model_id: str = DEFAULT_MODEL_ID,
    ticker: str = "SPY",
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "PredictedReturn": predicted_returns,
        }
    )
    frame.to_csv(output_path, index=False)

    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "ticker": ticker,
                "model_id": model_id,
                "window_size": WINDOW_SIZE,
                "samples": int(len(predicted_returns)),
            },
            file,
            indent=2,
        )
    return output_path


def load_cached_predictions(
    df: pd.DataFrame,
    predictions_path: Path = PREDICTIONS_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Chronos predictions not found at {predictions_path}. "
            "Run `python -m src.chronos_forecast` first."
        )

    cached = pd.read_csv(predictions_path, parse_dates=["Date"])
    _, _, dates = create_sequences(df, FEATURE_COLUMNS)
    aligned = cached.set_index("Date").loc[pd.to_datetime(dates), "PredictedReturn"]
    return aligned.to_numpy(dtype=np.float32), dates


def ensure_predictions(
    df: pd.DataFrame,
    *,
    force_regenerate: bool = False,
    predictions_path: Path = PREDICTIONS_PATH,
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    ticker: str = "SPY",
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached Chronos predictions or generate and persist them."""
    if predictions_path.exists() and not force_regenerate:
        return load_cached_predictions(df, predictions_path)

    predicted_returns, _, dates = generate_chronos_predictions(
        df,
        model_id=model_id,
        batch_size=batch_size,
        device=device,
    )
    save_predictions(predicted_returns, dates, output_path=predictions_path, model_id=model_id, ticker=ticker)
    return predicted_returns, dates


def compute_forecast_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float32)
    actuals = np.asarray(actuals, dtype=np.float32)
    errors = predictions - actuals
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "direction_accuracy": float(np.mean(np.sign(predictions) == np.sign(actuals))),
    }


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
    axes[0].set_title("Chronos Forecast: Predicted vs Actual Returns (Test Set)")
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


def run_chronos_forecast(
    ticker: str = "SPY",
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    features_dir: Path = FEATURES_DATA_DIR,
    predictions_path: Path = PREDICTIONS_PATH,
    force_regenerate: bool = False,
) -> dict:
    feature_path = features_dir / f"{ticker}_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature data not found: {feature_path}")

    df = pd.read_csv(feature_path, parse_dates=["Date"])
    predicted_returns, dates = ensure_predictions(
        df,
        force_regenerate=force_regenerate,
        predictions_path=predictions_path,
        model_id=model_id,
        batch_size=batch_size,
        device=device,
        ticker=ticker,
    )

    print(f"Generated Chronos predictions: {len(predicted_returns)} samples")
    print(f"Model: {model_id}")
    print(f"Saved predictions -> {predictions_path}")
    return {
        "ticker": ticker,
        "model_id": model_id,
        "predictions_path": str(predictions_path),
        "samples": int(len(predicted_returns)),
        "dates": dates,
        "predictions": predicted_returns,
    }


def run_forecast_evaluation(
    ticker: str = "SPY",
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    features_dir: Path = FEATURES_DATA_DIR,
    predictions_path: Path = PREDICTIONS_PATH,
    metrics_path: Path = METRICS_PATH,
    plot_path: Path = PLOT_PATH,
    force_regenerate: bool = False,
) -> dict:
    feature_path = features_dir / f"{ticker}_features.csv"
    df = pd.read_csv(feature_path, parse_dates=["Date"])

    predicted_returns, actual_returns, dates = generate_chronos_predictions(
        df,
        model_id=model_id,
        batch_size=batch_size,
        device=device,
    )
    save_predictions(predicted_returns, dates, output_path=predictions_path, model_id=model_id, ticker=ticker)

    splits = train_val_test_split(
        predicted_returns[:, None],
        actual_returns,
        dates,
    )
    _, y_test, test_dates = splits["test"]
    preds_test = splits["test"][0].squeeze(-1)

    raw_metrics = compute_forecast_metrics(preds_test, y_test)
    direction_accuracy = raw_metrics["direction_accuracy"]
    metrics = {
        "ticker": ticker,
        "model_id": model_id,
        "split": "test",
        "test_samples": int(len(y_test)),
        "rmse": raw_metrics["rmse"],
        "mae": raw_metrics["mae"],
        "direction_accuracy": direction_accuracy,
        "target_direction_accuracy": 0.55,
        "meets_direction_target": direction_accuracy > 0.55,
        "forecast_backend": "chronos",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    plot_forecast_evaluation(test_dates, preds_test, y_test, output_path=plot_path)

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
    parser = argparse.ArgumentParser(description="Generate or evaluate Chronos return forecasts")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate Chronos forecasts on the test split",
    )
    parser.add_argument("--ticker", default="SPY", help="ticker symbol")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face Chronos model id")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="inference batch size")
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps")
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="regenerate cached predictions even if they already exist",
    )
    args = parser.parse_args()

    if args.evaluate:
        run_forecast_evaluation(
            ticker=args.ticker,
            model_id=args.model_id,
            batch_size=args.batch_size,
            device=args.device,
            force_regenerate=args.force_regenerate,
        )
    else:
        run_chronos_forecast(
            ticker=args.ticker,
            model_id=args.model_id,
            batch_size=args.batch_size,
            device=args.device,
            force_regenerate=args.force_regenerate,
        )

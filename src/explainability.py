"""Explainability utilities for DeepTrade (Phase 9)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.features import FEATURE_COLUMNS, FEATURES_DATA_DIR
from src.train_transformer import load_model_from_checkpoint, transform_sequences
from src.transformer import WINDOW_SIZE, TEST_START, TRAIN_END, VAL_END, VAL_START, create_sequences, train_val_test_split

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ATTENTION_HEATMAP_PATH = RESULTS_DIR / "attention_heatmap.png"
ATTENTION_SUMMARY_PATH = RESULTS_DIR / "attention_summary.txt"
SHAP_SUMMARY_PATH = RESULTS_DIR / "shap_summary.png"
SHAP_DOC_PATH = RESULTS_DIR / "shap_summary.txt"

DEFAULT_NUM_SAMPLES = 32
DEFAULT_SHAP_BACKGROUND = 80
DEFAULT_SHAP_SAMPLES = 50

CLASSIFIER_FEATURE_NAMES = (
    "TransformerPred",
    "RSI",
    "MACD",
    "EMA20",
    "SMA20",
    "SMA50",
    "VolumeChange",
)
CLASS_NAMES = ("Sell", "Hold", "Buy")


def _day_lag_labels(window_size: int = WINDOW_SIZE) -> list[str]:
    return [f"t-{window_size - 1 - idx}" for idx in range(window_size)]


def aggregate_attention_maps(layer_weights: list[torch.Tensor]) -> np.ndarray:
    """
    Average attention across batch and query positions.

    Returns array with shape (num_layers, num_heads, window_size).
    """
    maps = []
    for weights in layer_weights:
        # (batch, head, query, key) -> (head, key)
        averaged = weights.detach().cpu().numpy().mean(axis=(0, 2))
        maps.append(averaged)
    return np.stack(maps)


def summarize_top_lags(
    attention_map: np.ndarray,
    window_size: int = WINDOW_SIZE,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Return top day lags (0 = oldest day in window) by mean attention."""
    day_scores = attention_map.mean(axis=(0, 1))
    top_indices = np.argsort(day_scores)[::-1][:top_k]
    return [(int(idx), float(day_scores[idx])) for idx in top_indices]


def plot_attention_heatmap(
    attention_map: np.ndarray,
    output_path: Path = ATTENTION_HEATMAP_PATH,
    window_size: int = WINDOW_SIZE,
) -> None:
    num_layers, num_heads, _ = attention_map.shape
    rows = num_layers * num_heads
    heatmap = attention_map.reshape(rows, window_size)
    row_labels = [f"L{layer + 1}H{head + 1}" for layer in range(num_layers) for head in range(num_heads)]
    day_labels = _day_lag_labels(window_size)

    fig, ax = plt.subplots(figsize=(14, 6))
    image = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_title("Transformer Self-Attention (Test Windows)")
    ax.set_xlabel("Historical day in 60-day window")
    ax.set_ylabel("Layer × Head")
    ax.set_xticks(np.arange(0, window_size, 5))
    ax.set_xticklabels([day_labels[i] for i in range(0, window_size, 5)], rotation=45, ha="right")
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels(row_labels)
    fig.colorbar(image, ax=ax, label="Mean attention weight")
    fig.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_attention_summary(
    attention_map: np.ndarray,
    num_samples: int,
    output_path: Path = ATTENTION_SUMMARY_PATH,
    window_size: int = WINDOW_SIZE,
    top_k: int = 5,
) -> str:
    day_scores = attention_map.mean(axis=(0, 1))
    top_indices = np.argsort(day_scores)[::-1][:top_k]
    top_lags = [(int(idx), float(day_scores[idx])) for idx in top_indices]

    lines = [
        "Transformer Attention Summary",
        "=============================",
        f"Samples analyzed: {num_samples} test windows",
        "",
        "Top attended day lags (relative to prediction day t):",
    ]

    for rank, (index, score) in enumerate(top_lags, start=1):
        lag = window_size - 1 - index
        recency = "most recent day (t)" if lag == 0 else f"t-{lag}"
        lines.append(f"  {rank}. {recency} (window index {index}) — mean attention {score:.4f}")

    recent_mass = float(day_scores[-5:].sum() / day_scores.sum())
    monthly_band = day_scores[max(0, window_size - 22) : max(0, window_size - 18)].sum() / day_scores.sum()

    lines.extend(
        [
            "",
            "Interpretation:",
            f"- Recent days (last 5 in window) account for {recent_mass:.1%} of total attention mass.",
            f"- Days around t-20 account for {monthly_band:.1%} of total attention mass.",
        ]
    )

    dominant_lags = [window_size - 1 - idx for idx, _ in top_lags[:3]]
    if max(dominant_lags) <= 5:
        lines.append("- The model focuses most on the very recent trading history.")
    elif any(15 <= lag <= 25 for lag in dominant_lags):
        lines.append("- Attention peaks around t-20 to t-30, suggesting a ~monthly lookback pattern.")
    else:
        lines.append("- Attention is distributed across the window with multiple focal lags.")

    summary = "\n".join(lines)
    output_path.write_text(summary, encoding="utf-8")
    return summary


def run_attention_analysis(
    ticker: str = "SPY",
    num_samples: int = DEFAULT_NUM_SAMPLES,
    features_dir: Path = FEATURES_DATA_DIR,
    heatmap_path: Path = ATTENTION_HEATMAP_PATH,
    summary_path: Path = ATTENTION_SUMMARY_PATH,
    seed: int = 42,
) -> dict:
    model, checkpoint, device = load_model_from_checkpoint()
    scaler = checkpoint["scaler"]

    feature_path = features_dir / f"{ticker}_features.csv"
    df = pd.read_csv(feature_path, parse_dates=["Date"])
    X, _, dates = create_sequences(df, FEATURE_COLUMNS)
    splits = train_val_test_split(X, _, dates)
    X_test, _, _ = splits["test"]
    X_test = transform_sequences(X_test, scaler)

    rng = np.random.default_rng(seed)
    sample_count = min(num_samples, len(X_test))
    sample_idx = rng.choice(len(X_test), size=sample_count, replace=False)
    X_sample = torch.from_numpy(X_test[sample_idx]).to(device)

    with torch.no_grad():
        layer_weights = model.extract_attention_weights(X_sample)

    attention_map = aggregate_attention_maps(layer_weights)
    top_lags = summarize_top_lags(attention_map)

    plot_attention_heatmap(attention_map, output_path=heatmap_path)
    summary = write_attention_summary(attention_map, sample_count, output_path=summary_path)

    print(summary)
    print(f"\nSaved heatmap -> {heatmap_path}")
    print(f"Saved summary -> {summary_path}")

    return {
        "num_samples": sample_count,
        "top_lags": top_lags,
        "heatmap_path": heatmap_path,
        "summary_path": summary_path,
    }


def _build_classifier_splits(
    ticker: str = "SPY",
    features_dir: Path = FEATURES_DATA_DIR,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    from src.signal_generator import build_classifier_dataset, load_classifier_from_checkpoint

    transformer, transformer_ckpt, device = load_model_from_checkpoint()
    _, classifier_ckpt, _ = load_classifier_from_checkpoint(device=device)

    feature_path = features_dir / f"{ticker}_features.csv"
    df = pd.read_csv(feature_path, parse_dates=["Date"])
    X, y, dates = build_classifier_dataset(
    df,
    buy_thresh=classifier_ckpt.get("buy_thresh", 0.01),
    sell_thresh=classifier_ckpt.get("sell_thresh", -0.01),
)

    dates = pd.to_datetime(dates)
    train_mask = dates <= TRAIN_END
    val_mask = (dates >= VAL_START) & (dates <= VAL_END)
    test_mask = dates >= TEST_START

    split_data = {
        "train": (X[train_mask], y[train_mask]),
        "val": (X[val_mask], y[val_mask]),
        "test": (X[test_mask], y[test_mask]),
    }
    return split_data, list(CLASSIFIER_FEATURE_NAMES)


def _classifier_predict_proba(
    model: torch.nn.Module,
    device: torch.device,
) -> callable:
    def predict_proba(X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X.astype(np.float32)).to(device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    return predict_proba


def _top_features_for_class(
    shap_values: np.ndarray,
    class_index: int,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    class_shap = np.abs(shap_values[:, :, class_index]).mean(axis=0)
    top_idx = np.argsort(class_shap)[::-1][:top_k]
    return [(CLASSIFIER_FEATURE_NAMES[i], float(class_shap[i])) for i in top_idx]


def write_shap_summary(
    buy_features: list[tuple[str, float]],
    sell_features: list[tuple[str, float]],
    output_path: Path = SHAP_DOC_PATH,
) -> str:
    lines = [
        "SHAP Feature Importance Summary",
        "===============================",
        "",
        "Top features driving Buy decisions (class=Buy):",
    ]
    for rank, (name, value) in enumerate(buy_features, start=1):
        lines.append(f"  {rank}. {name} — mean |SHAP| {value:.4f}")

    lines.extend(["", "Top features driving Sell decisions (class=Sell):"])
    for rank, (name, value) in enumerate(sell_features, start=1):
        lines.append(f"  {rank}. {name} — mean |SHAP| {value:.4f}")

    lines.extend(
        [
            "",
            "Interpretation:",
            f"- Buy signals are most influenced by {buy_features[0][0]} and {buy_features[1][0]}.",
            f"- Sell signals are most influenced by {sell_features[0][0]} and {sell_features[1][0]}.",
            "- Transformer forecast and momentum indicators (RSI, MACD) typically dominate trading decisions.",
        ]
    )

    summary = "\n".join(lines)
    output_path.write_text(summary, encoding="utf-8")
    return summary


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    output_path: Path = SHAP_SUMMARY_PATH,
) -> None:
    import shap

    plt.close("all")
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=list(CLASSIFIER_FEATURE_NAMES),
        class_names=list(CLASS_NAMES),
        plot_type="bar",
        show=False,
        color_bar=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig.suptitle("SHAP Feature Importance — Signal Classifier (Test Sample)", y=1.02)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_shap_analysis(
    ticker: str = "SPY",
    background_size: int = DEFAULT_SHAP_BACKGROUND,
    num_samples: int = DEFAULT_SHAP_SAMPLES,
    features_dir: Path = FEATURES_DATA_DIR,
    summary_plot_path: Path = SHAP_SUMMARY_PATH,
    summary_doc_path: Path = SHAP_DOC_PATH,
    seed: int = 42,
) -> dict:
    import shap
    from src.signal_generator import load_classifier_from_checkpoint

    splits, _ = _build_classifier_splits(ticker=ticker, features_dir=features_dir)
    X_train, _ = splits["train"]
    X_test, y_test = splits["test"]

    model, _, device = load_classifier_from_checkpoint()
    predict_proba = _classifier_predict_proba(model, device)

    rng = np.random.default_rng(seed)
    background_count = min(background_size, len(X_train))
    sample_count = min(num_samples, len(X_test))
    background_idx = rng.choice(len(X_train), size=background_count, replace=False)
    sample_idx = rng.choice(len(X_test), size=sample_count, replace=False)

    background = X_train[background_idx]
    X_sample = X_test[sample_idx]
    y_sample = y_test[sample_idx]

    explainer = shap.KernelExplainer(predict_proba, background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)

    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=-1)
    else:
        shap_array = shap_values

    buy_features = _top_features_for_class(shap_array, class_index=2)
    sell_features = _top_features_for_class(shap_array, class_index=0)

    plot_shap_summary(shap_array, X_sample, output_path=summary_plot_path)
    summary = write_shap_summary(buy_features, sell_features, output_path=summary_doc_path)

    print(summary)
    print(f"\nSaved SHAP plot -> {summary_plot_path}")
    print(f"Saved SHAP summary -> {summary_doc_path}")
    print(f"Test sample class counts: Sell={np.sum(y_sample == 0)}, "
          f"Hold={np.sum(y_sample == 1)}, Buy={np.sum(y_sample == 2)}")

    return {
        "num_samples": sample_count,
        "buy_features": buy_features,
        "sell_features": sell_features,
        "summary_plot_path": summary_plot_path,
        "summary_doc_path": summary_doc_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepTrade explainability analysis")
    parser.add_argument(
        "--attention",
        action="store_true",
        help="run transformer attention analysis (Phase 9, Step 19)",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="run SHAP analysis for the signal classifier (Phase 9, Step 20)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="number of test windows to sample for attention analysis",
    )
    parser.add_argument(
        "--shap-samples",
        type=int,
        default=DEFAULT_SHAP_SAMPLES,
        help="number of test samples for SHAP analysis",
    )
    args = parser.parse_args()

    if args.attention and args.shap:
        run_attention_analysis(num_samples=args.samples)
        run_shap_analysis(num_samples=args.shap_samples)
    elif args.attention:
        run_attention_analysis(num_samples=args.samples)
    else:
        run_shap_analysis(num_samples=args.shap_samples)

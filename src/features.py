from pathlib import Path

import pandas as pd
import ta

from src.data_loader import DEFAULT_TICKERS, PROCESSED_DATA_DIR

FEATURES_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "features"

BASE_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")
INDICATOR_COLUMNS = ("SMA20", "SMA50", "EMA20", "RSI", "MACD", "VolumeChange")
OUTPUT_COLUMNS = BASE_COLUMNS + INDICATOR_COLUMNS


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    df["SMA20"] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    df["SMA50"] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    df["EMA20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(close=close).rsi()
    df["MACD"] = ta.trend.MACD(close=close).macd()
    df["VolumeChange"] = volume.pct_change()

    return df.dropna().reset_index(drop=True)


def build_feature_dataset(
    ticker: str,
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = FEATURES_DATA_DIR,
) -> pd.DataFrame:
    processed_path = processed_dir / f"{ticker}.csv"
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_path}")

    df = pd.read_csv(processed_path, parse_dates=["Date"])
    df = add_technical_indicators(df)
    df = df[list(OUTPUT_COLUMNS)]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}_features.csv"
    df.to_csv(output_path, index=False)
    return df


def build_all_feature_datasets(
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = FEATURES_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = build_feature_dataset(ticker, processed_dir=processed_dir, output_dir=output_dir)
        datasets[ticker] = df
        print(f"Built features for {ticker}: {len(df)} rows -> {output_dir / f'{ticker}_features.csv'}")
    return datasets


if __name__ == "__main__":
    build_all_feature_datasets()

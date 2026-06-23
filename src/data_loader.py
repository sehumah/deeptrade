from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DEFAULT_TICKERS = ("SPY", "AAPL", "MSFT", "NVDA")
DEFAULT_START = "2015-01-01"
OHLCV_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_data(
    ticker: str,
    start: str = DEFAULT_START,
    end: str | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    df = _flatten_columns(df).reset_index()
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns for {ticker}: {sorted(missing)}")

    df = df[list(OHLCV_COLUMNS)].sort_values("Date").reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def download_all(
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str | None = None,
    output_dir: Path = RAW_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = output_dir / f"{ticker}.csv"
        datasets[ticker] = download_data(ticker, start=start, end=end, output_path=path)
        print(f"Saved {ticker}: {len(datasets[ticker])} rows -> {path}")
    return datasets


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def add_normalized_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Scale OHLC by the first close; raw prices are preserved for P&L."""
    df = df.copy()
    base_close = df["Close"].iloc[0]
    for col in ("Open", "High", "Low", "Close"):
        df[f"{col}Norm"] = df[col] / base_close
    return df


def save_processed(
    ticker: str,
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    raw_path = raw_dir / f"{ticker}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    df = pd.read_csv(raw_path, parse_dates=["Date"])
    df = clean_data(df)
    df = add_normalized_prices(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}.csv"
    df.to_csv(output_path, index=False)
    return df


def process_all(
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = save_processed(ticker, raw_dir=raw_dir, output_dir=output_dir)
        datasets[ticker] = df
        print(f"Processed {ticker}: {len(df)} rows -> {output_dir / f'{ticker}.csv'}")
    return datasets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download or process market data")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("download", "process"),
        default="download",
        help="download raw data (default) or process into cleaned CSVs",
    )
    args = parser.parse_args()

    if args.command == "process":
        process_all()
    else:
        download_all()

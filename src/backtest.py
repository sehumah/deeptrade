from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.trading_env import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    TradingEnv,
    build_env_arrays,
    make_trading_env,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FEATURES_PATH = Path(__file__).resolve().parent.parent / "data" / "features" / "SPY_features.csv"
PPO_MODEL_PATH = RESULTS_DIR / "ppo_agent"
EQUITY_CURVE_PATH = RESULTS_DIR / "backtest_equity_curve.csv"
ACTIONS_PATH = RESULTS_DIR / "backtest_actions.csv"
EQUITY_PLOT_PATH = RESULTS_DIR / "backtest_equity_curve.png"
BASELINE_COMPARISON_PATH = RESULTS_DIR / "baseline_comparison.csv"
BASELINE_COMPARISON_PLOT_PATH = RESULTS_DIR / "baseline_comparison.png"
TRADING_METRICS_PATH = RESULTS_DIR / "trading_metrics.json"

TEST_START = "2024-01-01"
RANDOM_BASELINE_SEED = 42
TRADING_DAYS_PER_YEAR = 252
ACTION_LABELS = {ACTION_HOLD: "hold", ACTION_BUY: "buy", ACTION_SELL: "sell"}
STRATEGY_LABELS = {
    "ppo_agent": "PPO Agent",
    "buy_and_hold": "Buy & Hold",
    "moving_average": "Moving Average",
    "random": "Random",
}


def run_backtest(
    model: PPO,
    env,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roll the trained agent forward day-by-day on the test-period environment."""
    del df_test  # Environment is already built from filtered test data.

    obs, info = env.reset()
    equity_rows: list[dict] = []
    action_rows: list[dict] = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)

        date = info.get("date")
        equity_rows.append(
            {
                "Date": date,
                "portfolio_value": info["portfolio_value"],
                "daily_return": info.get("daily_return", 0.0),
                "reward": float(reward),
                "cash": info.get("cash", 0.0),
                "shares": info.get("shares", 0.0),
                "price": info.get("price", 0.0),
            }
        )
        action_rows.append(
            {
                "Date": date,
                "requested_action": info.get("requested_action", action),
                "action_taken": info.get("action_taken", action),
                "action_label": ACTION_LABELS.get(info.get("action_taken", action), "unknown"),
                "transaction_cost": info.get("transaction_cost", 0.0),
            }
        )

        if terminated or truncated:
            break

    equity_df = pd.DataFrame(equity_rows)
    actions_df = pd.DataFrame(action_rows)
    return equity_df, actions_df


def plot_equity_curve(equity_df: pd.DataFrame, output_path: Path = EQUITY_PLOT_PATH) -> None:
    dates = pd.to_datetime(equity_df["Date"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, equity_df["portfolio_value"], linewidth=1.5, label="PPO Agent")
    ax.set_title("Backtest Equity Curve (Test Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_test_backtest(
    features_path: Path = FEATURES_PATH,
    model_path: Path = PPO_MODEL_PATH,
    test_start: str = TEST_START,
    equity_path: Path = EQUITY_CURVE_PATH,
    actions_path: Path = ACTIONS_PATH,
    plot_path: Path = EQUITY_PLOT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not features_path.exists():
        raise FileNotFoundError(f"Feature data not found: {features_path}")

    model_file = Path(f"{model_path}.zip")
    if not model_file.exists():
        raise FileNotFoundError(f"PPO model not found: {model_file}")

    df = pd.read_csv(features_path, parse_dates=["Date"])
    df_test = df[df["Date"] >= pd.Timestamp(test_start)].copy()
    if df_test.empty:
        raise ValueError(f"No test rows found on or after {test_start}")

    env = make_trading_env(df, start_date=test_start)
    model = PPO.load(str(model_path))

    equity_df, actions_df = run_backtest(model, env, df_test)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    equity_df.to_csv(equity_path, index=False)
    actions_df.to_csv(actions_path, index=False)
    plot_equity_curve(equity_df, output_path=plot_path)

    print(f"Backtest days: {len(equity_df)}")
    print(f"Final portfolio value: ${equity_df['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Saved equity curve -> {equity_path}")
    print(f"Saved actions -> {actions_path}")
    print(f"Saved plot -> {plot_path}")
    return equity_df, actions_df


def load_test_market_arrays(
    df: pd.DataFrame,
    test_start: str = TEST_START,
) -> dict[str, np.ndarray]:
    """Load sequence-aligned test-period prices, indicators, and dates."""
    arrays = build_env_arrays(df, start_date=test_start)
    lookup = df.set_index(pd.to_datetime(df["Date"]))
    dates = pd.to_datetime(arrays["dates"])
    indicators = lookup.loc[dates, ["SMA20", "SMA50"]]
    arrays["sma20"] = indicators["SMA20"].to_numpy(dtype=np.float64)
    arrays["sma50"] = indicators["SMA50"].to_numpy(dtype=np.float64)
    return arrays


def run_strategy_in_env(
    arrays: dict[str, np.ndarray],
    action_selector: Callable[[TradingEnv, int], int],
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a strategy in :class:`TradingEnv` using the same mechanics as the PPO agent."""
    env = TradingEnv(
        prices=arrays["prices"],
        rsi=arrays["rsi"],
        macd=arrays["macd"],
        transformer_predictions=arrays["transformer_predictions"],
        dates=arrays["dates"],
        seed=seed,
    )
    obs, _ = env.reset()
    portfolio_values: list[float] = []
    step_idx = 0

    while True:
        action = action_selector(env, step_idx)
        obs, _, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        step_idx += 1
        if terminated or truncated:
            break

    return np.asarray(portfolio_values, dtype=np.float64)


def run_buy_and_hold_baseline(arrays: dict[str, np.ndarray]) -> np.ndarray:
    def action_selector(env: TradingEnv, step_idx: int) -> int:
        del env
        return ACTION_BUY if step_idx == 0 else ACTION_HOLD

    return run_strategy_in_env(arrays, action_selector)


def run_moving_average_baseline(arrays: dict[str, np.ndarray]) -> np.ndarray:
    sma20 = arrays["sma20"]
    sma50 = arrays["sma50"]

    def action_selector(env: TradingEnv, step_idx: int) -> int:
        del step_idx
        if sma20[env.current_step] > sma50[env.current_step]:
            return ACTION_BUY if env.cash > 0.0 else ACTION_HOLD
        return ACTION_SELL if env.shares > 0.0 else ACTION_HOLD

    return run_strategy_in_env(arrays, action_selector)


def run_random_baseline(
    arrays: dict[str, np.ndarray],
    seed: int = RANDOM_BASELINE_SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    def action_selector(env: TradingEnv, step_idx: int) -> int:
        del env, step_idx
        return int(rng.integers(0, 3))

    return run_strategy_in_env(arrays, action_selector, seed=seed)


def run_ppo_baseline(
    df: pd.DataFrame,
    model_path: Path = PPO_MODEL_PATH,
    test_start: str = TEST_START,
) -> np.ndarray:
    model_file = Path(f"{model_path}.zip")
    if not model_file.exists():
        raise FileNotFoundError(f"PPO model not found: {model_file}")

    env = make_trading_env(df, start_date=test_start)
    model = PPO.load(str(model_path))
    equity_df, _ = run_backtest(model, env, df[df["Date"] >= pd.Timestamp(test_start)])
    return equity_df["portfolio_value"].to_numpy(dtype=np.float64)


def load_ppo_equity_curve(
    features_path: Path = FEATURES_PATH,
    model_path: Path = PPO_MODEL_PATH,
    test_start: str = TEST_START,
    equity_path: Path = EQUITY_CURVE_PATH,
) -> pd.Series:
    if equity_path.exists():
        equity_df = pd.read_csv(equity_path, parse_dates=["Date"])
        return equity_df.set_index("Date")["portfolio_value"]

    equity_df, _ = run_test_backtest(
        features_path=features_path,
        model_path=model_path,
        test_start=test_start,
        equity_path=equity_path,
    )
    return equity_df.set_index("Date")["portfolio_value"]


def run_baseline_comparison(
    features_path: Path = FEATURES_PATH,
    model_path: Path = PPO_MODEL_PATH,
    test_start: str = TEST_START,
    comparison_path: Path = BASELINE_COMPARISON_PATH,
    plot_path: Path = BASELINE_COMPARISON_PLOT_PATH,
    metrics_path: Path = TRADING_METRICS_PATH,
    random_seed: int = RANDOM_BASELINE_SEED,
) -> pd.DataFrame:
    """Run PPO agent and baseline strategies on the same test dates (Step 17)."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature data not found: {features_path}")

    df = pd.read_csv(features_path, parse_dates=["Date"])
    arrays = load_test_market_arrays(df, test_start=test_start)
    dates = pd.to_datetime(arrays["dates"])

    ppo_values = run_ppo_baseline(df, model_path=model_path, test_start=test_start)
    buy_hold_values = run_buy_and_hold_baseline(arrays)
    ma_values = run_moving_average_baseline(arrays)
    random_values = run_random_baseline(arrays, seed=random_seed)

    n_steps = len(ppo_values)
    if n_steps != len(dates):
        dates = dates[:n_steps]
        buy_hold_values = buy_hold_values[:n_steps]
        ma_values = ma_values[:n_steps]
        random_values = random_values[:n_steps]

    comparison_df = pd.DataFrame(
        {
            "Date": dates,
            "ppo_agent": ppo_values,
            "buy_and_hold": buy_hold_values,
            "moving_average": ma_values,
            "random": random_values,
        }
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    plot_baseline_comparison(comparison_df, output_path=plot_path)
    save_trading_metrics(comparison_df, output_path=metrics_path)

    print(f"Saved baseline comparison -> {comparison_path}")
    print(f"Saved comparison plot -> {plot_path}")
    print(f"Saved trading metrics -> {metrics_path}")
    return comparison_df


def plot_baseline_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path = BASELINE_COMPARISON_PLOT_PATH,
) -> None:
    dates = pd.to_datetime(comparison_df["Date"])

    fig, ax = plt.subplots(figsize=(12, 6))
    for column in STRATEGY_LABELS:
        ax.plot(
            dates,
            comparison_df[column],
            linewidth=1.5,
            label=STRATEGY_LABELS[column],
        )

    ax.set_title("Strategy Equity Curves (Test Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_trading_metrics(equity_curve: pd.Series | np.ndarray) -> dict[str, float]:
    """Compute headline trading metrics for a single equity curve (Step 18)."""
    values = np.asarray(equity_curve, dtype=np.float64)
    if len(values) < 2:
        raise ValueError("Equity curve must contain at least two values.")

    daily_returns = np.diff(values) / values[:-1]
    total_return = (values[-1] - values[0]) / values[0]

    n_days = len(values) - 1
    years = n_days / TRADING_DAYS_PER_YEAR
    annual_return = (values[-1] / values[0]) ** (1 / years) - 1 if years > 0 else 0.0

    excess_returns = daily_returns
    return_std = excess_returns.std(ddof=0)
    sharpe_ratio = (
        float(excess_returns.mean() / return_std * np.sqrt(TRADING_DAYS_PER_YEAR))
        if return_std > 0
        else 0.0
    )

    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    max_drawdown = float(drawdowns.min())

    win_rate = float((daily_returns > 0).mean())

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def save_trading_metrics(
    comparison_df: pd.DataFrame,
    output_path: Path = TRADING_METRICS_PATH,
) -> dict[str, dict[str, float]]:
    metrics = {
        strategy: compute_trading_metrics(comparison_df[strategy])
        for strategy in STRATEGY_LABELS
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print_metrics_summary(metrics)
    return metrics


def print_metrics_summary(metrics: dict[str, dict[str, float]]) -> None:
    rows = []
    for strategy, values in metrics.items():
        rows.append(
            {
                "strategy": STRATEGY_LABELS[strategy],
                "total_return": values["total_return"],
                "annual_return": values["annual_return"],
                "sharpe_ratio": values["sharpe_ratio"],
                "max_drawdown": values["max_drawdown"],
                "win_rate": values["win_rate"],
            }
        )

    summary = pd.DataFrame(rows)
    print("\nTrading Metrics Summary (Test Period)")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the trained PPO agent on test data")
    parser.add_argument(
        "--test-start",
        default=TEST_START,
        help="first date of the backtest window (default: 2024-01-01)",
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="compare PPO agent against buy-and-hold, moving-average, and random baselines",
    )
    args = parser.parse_args()

    if args.baselines:
        run_baseline_comparison(test_start=args.test_start)
    else:
        run_test_backtest(test_start=args.test_start)

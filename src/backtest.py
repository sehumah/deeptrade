from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO

from src.trading_env import ACTION_BUY, ACTION_HOLD, ACTION_SELL, make_trading_env

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FEATURES_PATH = Path(__file__).resolve().parent.parent / "data" / "features" / "SPY_features.csv"
PPO_MODEL_PATH = RESULTS_DIR / "ppo_agent"
EQUITY_CURVE_PATH = RESULTS_DIR / "backtest_equity_curve.csv"
ACTIONS_PATH = RESULTS_DIR / "backtest_actions.csv"
EQUITY_PLOT_PATH = RESULTS_DIR / "backtest_equity_curve.png"

TEST_START = "2024-01-01"
ACTION_LABELS = {ACTION_HOLD: "hold", ACTION_BUY: "buy", ACTION_SELL: "sell"}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the trained PPO agent on test data")
    parser.add_argument(
        "--test-start",
        default=TEST_START,
        help="first date of the backtest window (default: 2024-01-01)",
    )
    args = parser.parse_args()
    run_test_backtest(test_start=args.test_start)

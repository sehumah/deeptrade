from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from src.features import FEATURE_COLUMNS
from src.train_transformer import CHECKPOINT_PATH, load_model_from_checkpoint, transform_sequences
from src.transformer import WINDOW_SIZE, create_sequences

# Observation vector: [transformer_pred, RSI, MACD, cash_norm, shares_norm]
OBSERVATION_DIM = 5
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
DEFAULT_INITIAL_CASH = 100_000.0
DEFAULT_TRANSACTION_COST_RATE = 0.001


def _predict_transformer_returns(
    model: torch.nn.Module,
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


def build_env_arrays(
    df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...] = FEATURE_COLUMNS,
    transformer_path: Path | None = CHECKPOINT_PATH,
    transformer_predictions: np.ndarray | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> dict[str, np.ndarray]:
    """Build aligned price, indicator, and transformer arrays for the RL environment."""
    required_columns = {"Date", "Close", "RSI", "MACD"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    if transformer_predictions is None:
        if transformer_path is None or not Path(transformer_path).exists():
            raise FileNotFoundError(
                "Transformer checkpoint not found. Train the model first or pass "
                "transformer_predictions explicitly."
            )

        transformer, checkpoint, device = load_model_from_checkpoint(transformer_path)
        scaler = checkpoint["scaler"]
        X_seq, _, dates = create_sequences(df, feature_columns)
        X_scaled = transform_sequences(X_seq, scaler)
        transformer_predictions = _predict_transformer_returns(transformer, X_scaled, device)
    else:
        X_seq, _, dates = create_sequences(df, feature_columns)
        transformer_predictions = np.asarray(transformer_predictions, dtype=np.float32)
        if len(transformer_predictions) != len(dates):
            raise ValueError(
                "transformer_predictions length must match create_sequences output "
                f"({len(dates)}), got {len(transformer_predictions)}"
            )

    date_index = pd.to_datetime(df["Date"])
    lookup = df.set_index(date_index)
    aligned = lookup.loc[pd.to_datetime(dates), ["Close", "RSI", "MACD"]].to_numpy(dtype=np.float32)

    prices = aligned[:, 0]
    rsi = aligned[:, 1]
    macd = aligned[:, 2]
    dates = pd.to_datetime(dates)

    if start_date is not None:
        start_date = pd.Timestamp(start_date)
        mask = dates >= start_date
        prices = prices[mask]
        rsi = rsi[mask]
        macd = macd[mask]
        transformer_predictions = transformer_predictions[mask]
        dates = dates[mask]

    if end_date is not None:
        end_date = pd.Timestamp(end_date)
        mask = dates <= end_date
        prices = prices[mask]
        rsi = rsi[mask]
        macd = macd[mask]
        transformer_predictions = transformer_predictions[mask]
        dates = dates[mask]

    if len(prices) < 2:
        raise ValueError("Need at least 2 tradable days after filtering for portfolio returns.")

    return {
        "prices": prices,
        "rsi": rsi,
        "macd": macd,
        "transformer_predictions": transformer_predictions,
        "dates": dates.to_numpy(),
    }


class TradingEnv(gym.Env):
    """Gymnasium environment for daily equity trading on historical feature data.

    Observation (shape ``(5,)``):
        0. Transformer predicted return
        1. RSI
        2. MACD
        3. Normalized cash (cash / initial_cash)
        4. Normalized position value (shares * price / initial_cash)

    Actions:
        0 = Hold, 1 = Buy (all-in), 2 = Sell (flatten)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray,
        rsi: np.ndarray,
        macd: np.ndarray,
        transformer_predictions: np.ndarray,
        dates: np.ndarray | None = None,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        transaction_cost_rate: float = DEFAULT_TRANSACTION_COST_RATE,
        seed: int | None = None,
    ):
        super().__init__()

        self.prices = np.asarray(prices, dtype=np.float64)
        self.rsi = np.asarray(rsi, dtype=np.float32)
        self.macd = np.asarray(macd, dtype=np.float32)
        self.transformer_predictions = np.asarray(transformer_predictions, dtype=np.float32)
        self.dates = dates
        self.initial_cash = float(initial_cash)
        self.transaction_cost_rate = float(transaction_cost_rate)

        n = len(self.prices)
        for name, array in (
            ("rsi", self.rsi),
            ("macd", self.macd),
            ("transformer_predictions", self.transformer_predictions),
        ):
            if len(array) != n:
                raise ValueError(f"{name} length ({len(array)}) must match prices length ({n})")

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self._prev_portfolio_value = self.initial_cash

    @property
    def max_step(self) -> int:
        """Last index that still allows a one-day forward return."""
        return len(self.prices) - 2

    def _portfolio_value(self, step: int | None = None) -> float:
        step = self.current_step if step is None else step
        return float(self.cash + self.shares * self.prices[step])

    def _get_obs(self) -> np.ndarray:
        price = self.prices[self.current_step]
        cash_norm = self.cash / self.initial_cash
        shares_norm = (self.shares * price) / self.initial_cash
        obs = np.array(
            [
                self.transformer_predictions[self.current_step],
                self.rsi[self.current_step],
                self.macd[self.current_step],
                cash_norm,
                shares_norm,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}
        start_step = int(options.get("start_step", 0))
        if not 0 <= start_step <= self.max_step:
            raise ValueError(f"start_step must be in [0, {self.max_step}], got {start_step}")

        self.current_step = start_step
        self.cash = self.initial_cash
        self.shares = 0.0
        self._prev_portfolio_value = self._portfolio_value()

        info = {
            "step": self.current_step,
            "portfolio_value": self._prev_portfolio_value,
            "date": self._current_date(),
        }
        return self._get_obs(), info

    def _current_date(self):
        if self.dates is None:
            return None
        return pd.Timestamp(self.dates[self.current_step])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0, 1, or 2.")

        price = self.prices[self.current_step]
        prev_portfolio_value = self._portfolio_value()
        transaction_cost = 0.0
        trade_value = 0.0
        action_taken = ACTION_HOLD

        if action == ACTION_BUY and self.cash > 0.0:
            trade_value = self.cash
            transaction_cost = self.transaction_cost_rate * trade_value
            investable_cash = self.cash - transaction_cost
            self.shares += investable_cash / price
            self.cash = 0.0
            action_taken = ACTION_BUY
        elif action == ACTION_SELL and self.shares > 0.0:
            trade_value = self.shares * price
            transaction_cost = self.transaction_cost_rate * trade_value
            self.cash += trade_value - transaction_cost
            self.shares = 0.0
            action_taken = ACTION_SELL

        self.current_step += 1
        terminated = self.current_step > self.max_step
        truncated = False

        if terminated:
            portfolio_value = self._portfolio_value(self.max_step + 1)
            daily_return = 0.0
            reward = 0.0
        else:
            portfolio_value = self._portfolio_value()
            daily_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
            cost_as_return = transaction_cost / prev_portfolio_value if prev_portfolio_value > 0 else 0.0
            reward = float(daily_return - cost_as_return)

        info = {
            "step": self.current_step,
            "portfolio_value": portfolio_value,
            "daily_return": float(daily_return),
            "transaction_cost": float(transaction_cost),
            "trade_value": float(trade_value),
            "action_taken": action_taken,
            "requested_action": int(action),
            "cash": float(self.cash),
            "shares": float(self.shares),
            "price": float(self.prices[min(self.current_step, len(self.prices) - 1)]),
            "date": self._current_date(),
        }

        obs = self._get_obs() if not terminated else np.zeros(OBSERVATION_DIM, dtype=np.float32)
        return obs, reward, terminated, truncated, info


def make_trading_env(
    df: pd.DataFrame,
    *,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    transaction_cost_rate: float = DEFAULT_TRANSACTION_COST_RATE,
    transformer_path: Path | None = CHECKPOINT_PATH,
    transformer_predictions: np.ndarray | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    feature_columns: list[str] | tuple[str, ...] = FEATURE_COLUMNS,
    seed: int | None = None,
) -> TradingEnv:
    """Create a :class:`TradingEnv` from a feature DataFrame (e.g. ``SPY_features.csv``)."""
    arrays = build_env_arrays(
        df,
        feature_columns=feature_columns,
        transformer_path=transformer_path,
        transformer_predictions=transformer_predictions,
        start_date=start_date,
        end_date=end_date,
    )
    return TradingEnv(
        prices=arrays["prices"],
        rsi=arrays["rsi"],
        macd=arrays["macd"],
        transformer_predictions=arrays["transformer_predictions"],
        dates=arrays["dates"],
        initial_cash=initial_cash,
        transaction_cost_rate=transaction_cost_rate,
        seed=seed,
    )


__all__ = [
    "ACTION_BUY",
    "ACTION_HOLD",
    "ACTION_SELL",
    "OBSERVATION_DIM",
    "TradingEnv",
    "WINDOW_SIZE",
    "build_env_arrays",
    "make_trading_env",
]

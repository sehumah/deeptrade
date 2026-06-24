from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.trading_env import make_trading_env

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FEATURES_PATH = Path(__file__).resolve().parent.parent / "data" / "features" / "SPY_features.csv"
PPO_MODEL_PATH = RESULTS_DIR / "ppo_agent"
PPO_LOG_PATH = RESULTS_DIR / "ppo_training_log.csv"

TRAIN_START = "2015-01-01"
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"
DEFAULT_TIMESTEPS = 100_000


class EpisodicRewardLogger(BaseCallback):
    """Append finished-episode statistics to a CSV log."""

    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self.records: list[dict[str, float | int]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" not in info:
                continue
            episode = info["episode"]
            self.records.append(
                {
                    "timestep": self.num_timesteps,
                    "episode_reward": float(episode["r"]),
                    "episode_length": int(episode["l"]),
                }
            )
        return True

    def _on_training_end(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.records).to_csv(self.log_path, index=False)


def train_ppo(
    total_timesteps: int = DEFAULT_TIMESTEPS,
    eval_freq: int = 10_000,
    features_path: Path = FEATURES_PATH,
    model_path: Path = PPO_MODEL_PATH,
    log_path: Path = PPO_LOG_PATH,
) -> PPO:
    if not features_path.exists():
        raise FileNotFoundError(f"Feature data not found: {features_path}")

    df = pd.read_csv(features_path, parse_dates=["Date"])

    train_env = Monitor(
        make_trading_env(df, start_date=TRAIN_START, end_date=TRAIN_END),
        filename=None,
    )
    val_env = Monitor(
        make_trading_env(df, start_date=VAL_START, end_date=VAL_END),
        filename=None,
    )

    model = PPO("MlpPolicy", train_env, verbose=1)

    callbacks: list[BaseCallback] = [EpisodicRewardLogger(log_path)]
    if eval_freq > 0:
        callbacks.append(
            EvalCallback(
                val_env,
                best_model_save_path=str(RESULTS_DIR / "ppo_agent_best"),
                log_path=str(RESULTS_DIR / "ppo_eval"),
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
            )
        )

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"Saved PPO agent -> {model_path}.zip")
    print(f"Saved training log -> {log_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PPO trading agent on SPY train data")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="total PPO training timesteps",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="validation evaluation frequency in timesteps (0 to disable)",
    )
    args = parser.parse_args()
    train_ppo(total_timesteps=args.timesteps, eval_freq=args.eval_freq)

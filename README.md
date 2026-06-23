# DeepTrade

DeepTrade is an end-to-end deep learning trading system that combines Transformer-based time-series forecasting, signal generation, and reinforcement learning for automated trading decisions. The pipeline learns market patterns from historical OHLCV data and technical indicators, converts forecasts into Buy/Sell/Hold signals, and trains a PPO agent to maximize portfolio value through simulated trading.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

```text
Historical Data
      ↓
Transformer Forecast
      ↓
Buy/Sell/Hold Signal
      ↓
RL Agent Decision
      ↓
Portfolio Performance
```

**Forecast → Signal → Decision → Profit**

## Project Structure

```text
deeptrade/
├── data/                    # Raw and processed market data
├── notebooks/               # Exploratory analysis
├── src/                     # Source modules
├── results/                 # Models, metrics, and plots
├── requirements.txt
├── AGENT.md                 # Step-by-step build instructions
└── README.md
```

## Source Modules

| Module | Purpose |
|--------|---------|
| `data_loader.py` | Download, clean, and split market data |
| `features.py` | Technical indicators and forecast targets |
| `transformer.py` | Transformer model and sequence builder |
| `signal_generator.py` | Signal labels and classifier |
| `trading_env.py` | Gymnasium trading environment |
| `train_transformer.py` | Transformer training script |
| `train_agent.py` | PPO agent training script |
| `backtest.py` | Backtesting and baseline comparison |

# DeepTrade — Agent Build Instructions

This file is the single source of truth for building the DeepTrade project step by step. Each step is self-contained enough for a Cursor agent (or developer) to implement and verify in isolation.

## How to Use This File

Prompt the agent with a phase and step range using **global step numbers** (1–20):

```text
Follow steps 7–9 in Phase 4
```

That means: implement **Step 7**, **Step 8**, and **Step 9** in **Phase 4: Transformer Forecasting**, then run the listed verification commands before marking the work complete.

Other valid prompts:

```text
Follow Step 2 in Phase 2
Follow steps 4–5 in Phase 3
Follow all steps in Phase 6
Follow steps 15–16 in Phase 7
```

### Agent Rules

When executing any step(s):

1. Read `project_description.md` for architecture context.
2. Only implement the requested step(s); do not skip ahead into later phases.
3. Match the repository layout defined in Step 1.
4. Reuse existing modules — do not duplicate logic across files.
5. After implementation, run the **Verification** block for each completed step.
6. Save artifacts (models, plots, CSVs) under `results/` with descriptive names.
7. Do not train on test data (2024–2025). See the data split in Step 16.
8. Do not commit unless explicitly asked.

---

## Team of 5 Developers

Steps are assigned to balance workload and respect dependencies. A developer owns their phases but any step can be run by prompting the agent directly.

| Developer | Phases | Global Steps | Focus |
|-----------|--------|--------------|-------|
| **Dev 1** | 1–2 | 1–3 | Repo setup, data download & cleaning |
| **Dev 2** | 3, 4 (partial) | 4–7 | Feature engineering, sequences, Transformer architecture |
| **Dev 3** | 4 (partial), 5 | 8–12 | Transformer training/eval, signal generation |
| **Dev 4** | 6–7 | 13–16 | RL environment, PPO agent, backtest |
| **Dev 5** | 8–9 | 17–20 | Baselines, metrics, explainability, deliverables |

### Dependency Graph

```text
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8 → Phase 9
  (1)      (2–3)     (4–5)     (6–9)    (10–12)   (13–14)   (15–16)   (17–18)   (19–20)
```

Do not start a phase until its prerequisites are complete.

---

## Data Split (Used From Step 8 Onward)

| Split | Date Range | Purpose |
|-------|------------|---------|
| Train | 2015-01-01 – 2022-12-31 | Model training |
| Validation | 2023-01-01 – 2023-12-31 | Hyperparameter tuning, early stopping |
| Test | 2024-01-01 – present | Final evaluation only — never train on this |

Primary ticker for modeling: **SPY**. Also download AAPL, MSFT, NVDA for multi-asset experiments.

---

## Repository Layout

```text
deeptrade/
├── data/                    # Raw and processed CSVs
├── notebooks/               # Exploratory notebooks (optional)
├── src/
│   ├── data_loader.py       # Download, clean, split data
│   ├── features.py          # Technical indicators & targets
│   ├── transformer.py       # Transformer model & sequence builder
│   ├── signal_generator.py  # Signal labels & classifier
│   ├── trading_env.py       # Gymnasium trading environment
│   ├── train_transformer.py # Transformer training script
│   ├── train_agent.py       # PPO training script
│   └── backtest.py          # Backtesting & baseline comparison
├── results/                 # Models, plots, metrics JSON/CSV
├── requirements.txt
├── README.md
└── AGENT.md
```

---

# Phase 1: Project Setup

**Owner:** Dev 1  
**Prerequisites:** None

---

## Step 1: Create Repository Structure

**Goal:** Scaffold the project so later steps have a consistent home for code and artifacts.

### Tasks

1. Create directories: `data/`, `notebooks/`, `src/`, `results/`.
2. Create empty module files in `src/`:
   - `data_loader.py`
   - `features.py`
   - `transformer.py`
   - `signal_generator.py`
   - `trading_env.py`
   - `train_transformer.py`
   - `train_agent.py`
   - `backtest.py`
3. Ensure `requirements.txt` includes at minimum:

   ```text
   torch
   pandas
   numpy
   matplotlib
   scikit-learn
   yfinance
   ta
   gymnasium
   stable-baselines3
   ```

4. Expand `README.md` with: project title, one-paragraph description, setup instructions (`pip install -r requirements.txt`), and the pipeline overview (Forecast → Signal → Decision → Profit).

### Verification

```bash
pip install -r requirements.txt
python -c "import torch, pandas, yfinance, ta, gymnasium, stable_baselines3; print('OK')"
ls src/data_loader.py src/features.py src/transformer.py
```

### Done When

- All directories and `src/` modules exist.
- Dependencies install without error.
- `README.md` documents setup and pipeline.

---

# Phase 2: Data Collection

**Owner:** Dev 1  
**Prerequisites:** Phase 1 (Step 1)

---

## Step 2: Download Market Data

**Goal:** Fetch historical OHLCV data and persist it as CSV.

### Tasks

1. Implement `download_data(ticker, start, end, output_path)` in `src/data_loader.py` using `yfinance`:

   ```python
   import yfinance as yf
   df = yf.download(ticker, start="2015-01-01")
   ```

2. Download and save to `data/raw/`:
   - `data/raw/SPY.csv`
   - `data/raw/AAPL.csv`
   - `data/raw/MSFT.csv`
   - `data/raw/NVDA.csv`

3. Add a `if __name__ == "__main__"` block (or CLI) to run all downloads.

4. Ensure columns are flat (no MultiIndex): `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

### Verification

```bash
python -m src.data_loader
python -c "
import pandas as pd
df = pd.read_csv('data/raw/SPY.csv', parse_dates=['Date'])
assert len(df) > 1000
assert df['Close'].notna().all()
print(f'SPY rows: {len(df)}, range: {df.Date.min()} – {df.Date.max()}')
"
```

### Done When

- Four CSV files exist under `data/raw/`.
- SPY data spans from 2015 onward with no all-NaN columns.

---

## Step 3: Data Cleaning

**Goal:** Produce analysis-ready datasets with consistent dates and no missing values.

### Tasks

1. Implement `clean_data(df)` in `src/data_loader.py`:
   - Drop rows with missing values: `df = df.dropna()`
   - Sort by date ascending
   - Reset index
   - Ensure `Date` is datetime

2. Implement `save_processed(ticker)` that reads `data/raw/{ticker}.csv`, cleans, and writes `data/processed/{ticker}.csv`.

3. Optionally add price normalization helper (e.g. `Close` divided by first close) — store normalized columns separately so raw prices remain available for P&L calculations.

4. Process all four tickers.

### Verification

```bash
python -c "
from src.data_loader import clean_data
import pandas as pd
df = pd.read_csv('data/processed/SPY.csv', parse_dates=['Date'])
assert df['Date'].is_monotonic_increasing
assert df.isna().sum().sum() == 0
print('Clean SPY:', df.shape)
"
```

### Done When

- `data/processed/` contains cleaned CSVs for all tickers.
- Dates are monotonically increasing with zero NaNs.

---

# Phase 3: Feature Engineering

**Owner:** Dev 2  
**Prerequisites:** Phase 2 (Steps 2–3)

---

## Step 4: Create Technical Indicators

**Goal:** Enrich OHLCV data with technical features used by the Transformer and RL agent.

### Tasks

1. Implement `add_technical_indicators(df)` in `src/features.py` using the `ta` library:

   | Column | Indicator |
   |--------|-----------|
   | `SMA20` | Simple Moving Average (20) |
   | `SMA50` | Simple Moving Average (50) |
   | `EMA20` | Exponential Moving Average (20) |
   | `RSI` | Relative Strength Index |
   | `MACD` | MACD line |
   | `VolumeChange` | Day-over-day volume change |

2. Drop rows with NaN introduced by indicator warm-up periods.

3. Implement `build_feature_dataset(ticker)` that loads processed data, adds indicators, and saves `data/features/{ticker}_features.csv`.

4. Preserve base columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

### Verification

```bash
python -c "
from src.features import build_feature_dataset
build_feature_dataset('SPY')
import pandas as pd
df = pd.read_csv('data/features/SPY_features.csv')
for col in ['RSI','MACD','EMA20','SMA20','SMA50','VolumeChange']:
    assert col in df.columns, f'missing {col}'
print(df[['Date','Close','RSI','MACD']].tail())
"
```

### Done When

- Feature CSVs exist for all tickers.
- All six indicator columns are present and numeric.

---

## Step 5: Create Forecast Target

**Goal:** Add the next-day return target used to train the Transformer.

### Tasks

1. Implement `add_target_return(df)` in `src/features.py`:

   ```python
   df['TargetReturn'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
   ```

2. Drop the final row (no future return available).

3. Update `build_feature_dataset` to include `TargetReturn` in the output.

4. Document feature column list in a module-level constant, e.g. `FEATURE_COLUMNS`, excluding `Date`, `TargetReturn`, and raw OHLCV if you prefer normalized features only.

### Verification

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/features/SPY_features.csv')
assert 'TargetReturn' in df.columns
assert df['TargetReturn'].notna().all()
print('TargetReturn stats:', df['TargetReturn'].describe())
"
```

### Done When

- `TargetReturn` column exists with no NaNs.
- Feature files are regenerated with the target included.

---

# Phase 4: Transformer Forecasting

**Owner:** Dev 2 (Steps 6–7), Dev 3 (Steps 8–9)  
**Prerequisites:** Phase 3 (Steps 4–5)

---

## Step 6: Create Input Sequences

**Goal:** Convert tabular features into sliding-window sequences for the Transformer.

### Tasks

1. In `src/transformer.py`, define:

   ```python
   WINDOW_SIZE = 60  # past 60 trading days
   ```

2. Implement `create_sequences(df, feature_columns, target_col='TargetReturn', window_size=60)`:
   - Input shape: `(samples, 60, num_features)`
   - Target shape: `(samples,)`
   - Each sample uses days `[t-59 … t]` to predict return at `t+1` (already stored as `TargetReturn` at row `t`).

3. Implement `train_val_test_split(sequences, targets, dates)` using the global date split:
   - Train ≤ 2022-12-31
   - Val: 2023
   - Test ≥ 2024-01-01

4. Add a small test/demo in `if __name__ == "__main__"` that prints shapes for SPY.

### Verification

```bash
python -c "
from src.features import build_feature_dataset, FEATURE_COLUMNS
from src.transformer import create_sequences
import pandas as pd
df = pd.read_csv('data/features/SPY_features.csv', parse_dates=['Date'])
X, y, dates = create_sequences(df, FEATURE_COLUMNS)
print('X:', X.shape, 'y:', y.shape)
assert X.shape[1] == 60
"
```

### Done When

- Sequence builder returns correct 3D tensor shape.
- Date-based split function exists and respects train/val/test boundaries.

---

## Step 7: Build Transformer Encoder

**Goal:** Implement the PyTorch Transformer that predicts next-day return from a 60-day window.

### Architecture

```text
Input (batch, 60, features)
  → Linear Embedding
  → Positional Encoding
  → TransformerEncoder (nn.TransformerEncoderLayer × n_layers)
  → Pooling (mean over sequence length)
  → Linear head
  → Predicted return (batch, 1)
```

### Tasks

1. Implement in `src/transformer.py`:
   - `PositionalEncoding(nn.Module)`
   - `ReturnTransformer(nn.Module)` using `nn.TransformerEncoder` and `nn.TransformerEncoderLayer`
   - `forward(x)` returns `(batch, 1)` predicted return

2. Suggested defaults: `d_model=64`, `nhead=4`, `num_layers=2`, `dropout=0.1`.

3. Add `count_parameters(model)` helper for logging.

### Verification

```bash
python -c "
import torch
from src.transformer import ReturnTransformer
model = ReturnTransformer(input_dim=10, d_model=64)
x = torch.randn(8, 60, 10)
out = model(x)
assert out.shape == (8, 1)
print('Parameters:', sum(p.numel() for p in model.parameters()))
"
```

### Done When

- Model instantiates and forward pass works for shape `(batch, 60, features)`.
- Architecture uses `nn.TransformerEncoder`.

---

## Step 8: Train Transformer

**Goal:** Train the Transformer with MSE loss and save the best checkpoint.

### Tasks

1. Implement `src/train_transformer.py`:
   - Load SPY feature data, build sequences, split train/val
   - Loss: `nn.MSELoss()`
   - Optimizer: `torch.optim.Adam`
   - Track training loss and validation loss per epoch
   - Save best model to `results/transformer.pt` based on lowest validation loss
   - Log losses to `results/transformer_training_log.csv` or print/plot with matplotlib

2. Standardize features (fit scaler on **train only**, transform val).

3. Use reasonable defaults: `epochs=50`, `batch_size=32`, `lr=1e-3`, early stopping patience ~10 epochs.

### Verification

```bash
python -m src.train_transformer
python -c "
import torch, os
assert os.path.exists('results/transformer.pt')
ckpt = torch.load('results/transformer.pt', weights_only=True)
print('Checkpoint keys:', ckpt.keys() if isinstance(ckpt, dict) else 'state_dict only')
"
```

### Done When

- `results/transformer.pt` exists.
- Training and validation loss decrease over epochs (no need to be optimal yet).
- Scaler is saved alongside the model for inference.

---

## Step 9: Evaluate Forecasting

**Goal:** Measure forecast quality on the held-out test set.

### Tasks

1. Add `evaluate_forecast(model, X_test, y_test)` in `src/transformer.py` or `train_transformer.py` computing:
   - **RMSE**
   - **MAE**
   - **Direction Accuracy**: `mean(sign(pred) == sign(actual))`

2. Run evaluation on the **test split only** (2024+).

3. Save metrics to `results/forecast_metrics.json` and a pred-vs-actual plot to `results/forecast_eval.png`.

4. Target: direction accuracy **> 55%** (document actual result either way).

### Verification

```bash
python -c "
import json
m = json.load(open('results/forecast_metrics.json'))
for k in ['rmse','mae','direction_accuracy']:
    assert k in m
print(m)
"
```

### Done When

- Metrics file and plot exist.
- Direction accuracy is reported on test data only.

---

# Phase 5: Signal Generation

**Owner:** Dev 3  
**Prerequisites:** Phase 4 (Steps 6–9)

---

## Step 10: Generate Labels

**Goal:** Convert predicted/actual returns into Buy / Hold / Sell class labels.

### Tasks

1. In `src/signal_generator.py`, implement `return_to_signal(return_value, buy_thresh=0.01, sell_thresh=-0.01)`:

   ```python
   # 0 = Sell, 1 = Hold, 2 = Buy
   if return_value > buy_thresh:  return 2
   elif return_value < sell_thresh: return 0
   else: return 1
   ```

2. Implement `generate_labels(returns)` → array of class integers.

3. Create labels from **actual** `TargetReturn` for supervised classifier training.

4. Save label distribution stats to `results/signal_label_distribution.json`.

### Verification

```bash
python -c "
from src.signal_generator import return_to_signal, generate_labels
import numpy as np
assert return_to_signal(0.02) == 2
assert return_to_signal(-0.02) == 0
assert return_to_signal(0.0) == 1
print(generate_labels(np.array([0.02, -0.02, 0.0])))
"
```

### Done When

- Label mapping is implemented and tested.
- Class distribution is logged (watch for severe imbalance).

---

## Step 11: Build Signal Classifier

**Goal:** Train a model that maps Transformer output + indicators to Buy/Sell/Hold.

### Tasks

1. Implement `SignalClassifier` in `src/signal_generator.py` — a small MLP:

   ```text
   Input: [transformer_prediction, RSI, MACD, EMA20, SMA20, ...]
   Hidden: 64 → 32
   Output: 3-class logits
   ```

2. Implement `build_classifier_dataset(df, transformer, feature_columns)`:
   - Run Transformer inference on each window to get predicted return
   - Concatenate with selected technical indicators
   - Attach signal labels from Step 10

3. Train with `CrossEntropyLoss`, save to `results/signal_classifier.pt`.

### Verification

```bash
python -c "
import torch, os
from src.signal_generator import SignalClassifier
model = SignalClassifier(input_dim=8, hidden_dim=64, num_classes=3)
x = torch.randn(16, 8)
assert model(x).shape == (16, 3)
"
# After training script exists:
# python -m src.signal_generator --train
```

### Done When

- Classifier trains without error.
- `results/signal_classifier.pt` is saved.

---

## Step 12: Evaluate Signal Model

**Goal:** Report classification metrics and confusion matrix.

### Tasks

1. Implement `evaluate_classifier(model, X_test, y_test)` returning:
   - Accuracy
   - Precision, Recall, F1 (per class and macro avg via `sklearn.metrics`)

2. Plot confusion matrix → `results/signal_confusion_matrix.png`.

3. Save metrics → `results/signal_metrics.json`.

4. Evaluate on test-period data only.

### Verification

```bash
python -c "
import json
m = json.load(open('results/signal_metrics.json'))
for k in ['accuracy','f1_macro']:
    assert k in m
print(m)
"
```

### Done When

- Metrics JSON and confusion matrix plot exist.
- Evaluation uses test split only.

---

# Phase 6: Trading Environment

**Owner:** Dev 4  
**Prerequisites:** Phase 5 (Steps 10–12), Phase 4 (trained Transformer)

---

## Step 13: Create RL Environment

**Goal:** Build a Gymnasium environment that simulates trading using historical data.

### Tasks

1. Implement `TradingEnv(gymnasium.Env)` in `src/trading_env.py`.

2. **State observation** (1D float vector):
   - Transformer prediction (or last forecast)
   - RSI
   - MACD
   - Current cash (normalized)
   - Current shares (normalized)

3. **Actions** (Discrete):
   - `0` = Hold
   - `1` = Buy
   - `2` = Sell

4. Implement `reset()` and `step(action)`:
   - Advance one trading day per step
   - Update portfolio (cash, shares) on buy/sell
   - Return `(observation, reward, terminated, truncated, info)`

5. Register environment or export factory `make_trading_env(df, ...)`.

### Verification

```bash
python -c "
from src.trading_env import make_trading_env
import pandas as pd
df = pd.read_csv('data/features/SPY_features.csv', parse_dates=['Date'])
env = make_trading_env(df)
obs, info = env.reset()
obs, reward, term, trunc, info = env.step(0)
print('obs shape:', obs.shape, 'reward:', reward)
"
```

### Done When

- Environment passes `reset` / `step` smoke test.
- Observation dimension is fixed and documented.

---

## Step 14: Define Reward Function

**Goal:** Shape agent incentives around portfolio growth minus transaction costs.

### Tasks

1. Implement portfolio value tracking: `portfolio_value = cash + shares * current_price`.

2. **Simple reward** (baseline):
   ```python
   reward = portfolio_value_today - portfolio_value_yesterday
   ```

3. **Preferred reward** (use this by default):
   ```python
   reward = daily_return - transaction_cost
   ```
   where `transaction_cost = 0.001 * trade_value` (0.1% fee per trade).

4. Add `info` dict entries: `portfolio_value`, `daily_return`, `transaction_cost`.

5. Prevent invalid actions (e.g. buy with insufficient cash, sell without shares) by either clipping or applying a penalty.

### Verification

```bash
python -c "
from src.trading_env import make_trading_env
import pandas as pd
df = pd.read_csv('data/features/SPY_features.csv', parse_dates=['Date'])
env = make_trading_env(df)
obs, _ = env.reset()
total_reward = 0
for _ in range(100):
    obs, r, term, trunc, info = env.step(1)
    total_reward += r
    if term or trunc: break
print('100-step reward sum:', total_reward)
assert 'portfolio_value' in info
"
```

### Done When

- Reward uses transaction cost (0.1%).
- `info` exposes portfolio metrics for backtest logging.

---

# Phase 7: Reinforcement Learning Agent

**Owner:** Dev 4  
**Prerequisites:** Phase 6 (Steps 13–14)

---

## Step 15: Train PPO Agent

**Goal:** Train a PPO policy to maximize risk-adjusted portfolio return.

### Tasks

1. Implement `src/train_agent.py`:
   ```python
   from stable_baselines3 import PPO
   model = PPO("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=100_000)  # increase if time allows
   model.save("results/ppo_agent")
   ```

2. Train on **train-period data only** (2015–2022).

3. Optionally evaluate on validation env (2023) during training for monitoring.

4. Log episodic rewards to `results/ppo_training_log.csv`.

### Verification

```bash
python -m src.train_agent
python -c "
import os
assert os.path.exists('results/ppo_agent.zip')
print('PPO model saved')
"
```

### Done When

- `results/ppo_agent.zip` exists.
- Training completes on train split without data leakage.

---

## Step 16: Run Backtest

**Goal:** Simulate the trained agent on unseen test data and record portfolio performance.

### Tasks

1. Implement `run_backtest(model, env, df_test)` in `src/backtest.py`:
   - Roll forward day by day on test period (2024–2025)
   - Record daily portfolio value, actions, rewards

2. Save:
   - `results/backtest_equity_curve.csv`
   - `results/backtest_actions.csv`
   - Plot equity curve → `results/backtest_equity_curve.png`

3. **Never train on test data.**

### Verification

```bash
python -m src.backtest
python -c "
import pandas as pd
df = pd.read_csv('results/backtest_equity_curve.csv', parse_dates=['Date'])
assert len(df) > 50
print(df.tail())
"
```

### Done When

- Backtest runs end-to-end on test data.
- Equity curve CSV and plot are saved.

---

# Phase 8: Performance Analysis

**Owner:** Dev 5  
**Prerequisites:** Phase 7 (Steps 15–16)

---

## Step 17: Compare Against Baselines

**Goal:** Prove the RL agent adds value versus naive strategies.

### Tasks

1. In `src/backtest.py`, implement three baselines on the **same test period**:

   | Baseline | Strategy |
   |----------|----------|
   | Buy & Hold | Purchase on day 1, hold |
   | Moving Average | Buy when SMA20 > SMA50, sell otherwise |
   | Random | Random valid actions seeded for reproducibility |

2. Run all strategies, save comparison → `results/baseline_comparison.csv`.

3. Plot overlaid equity curves → `results/baseline_comparison.png`.

### Verification

```bash
python -m src.backtest --baselines
python -c "
import pandas as pd
df = pd.read_csv('results/baseline_comparison.csv')
for col in ['ppo_agent','buy_and_hold','moving_average','random']:
    assert col in df.columns
print(df.tail())
"
```

### Done When

- All four strategies have equity curves on identical test dates.
- Comparison CSV and plot exist.

---

## Step 18: Calculate Trading Metrics

**Goal:** Compute headline quantitative finance metrics for every strategy.

### Tasks

1. Implement `compute_trading_metrics(equity_curve)` in `src/backtest.py`:

   | Metric | Description |
   |--------|-------------|
   | Total Return | `(final - initial) / initial` |
   | Annual Return | CAGR over test period |
   | Sharpe Ratio | Mean excess daily return / std × √252 |
   | Maximum Drawdown | Largest peak-to-trough decline |
   | Win Rate | Fraction of days with positive return |

2. Save → `results/trading_metrics.json` for PPO and all baselines.

3. Print a summary table to console.

### Verification

```bash
python -c "
import json
m = json.load(open('results/trading_metrics.json'))
for strategy in m:
    for k in ['total_return','sharpe_ratio','max_drawdown']:
        assert k in m[strategy]
print(json.dumps(m, indent=2))
"
```

### Done When

- All five metrics computed for every strategy.
- Results saved and reproducible.

---

# Phase 9: Explainability (Bonus)

**Owner:** Dev 5  
**Prerequisites:** Phase 8 (Steps 17–18), trained Transformer (Step 8)

---

## Step 19: Analyze Attention

**Goal:** Visualize which historical days the Transformer attends to most.

### Tasks

1. Modify or wrap `ReturnTransformer` to expose attention weights from `nn.TransformerEncoder` (hook or `need_weights=True` if using a compatible API).

2. Create `notebooks/attention_analysis.ipynb` or script `src/explainability.py`:
   - Sample N test windows
   - Extract attention maps
   - Plot heatmaps (days × layers/heads) → `results/attention_heatmap.png`

3. Summarize top-attended lags (e.g. "model focuses on last 5 days and t-20").

### Verification

```bash
python -c "
import os
# After attention script runs:
assert os.path.exists('results/attention_heatmap.png')
print('Attention heatmap saved')
"
```

### Done When

- At least one attention heatmap is saved.
- Brief written summary exists in notebook or `results/attention_summary.txt`.

---

## Step 20: SHAP Analysis

**Goal:** Show which input features most influence model decisions.

### Tasks

1. Add `shap` to `requirements.txt`.

2. In `src/explainability.py`:
   - Build a prediction wrapper around the signal classifier (or Transformer + classifier pipeline)
   - Compute SHAP values on a representative test sample
   - Plot summary bar chart → `results/shap_summary.png`

3. Document top features driving Buy vs Sell decisions.

### Verification

```bash
pip install shap
python -m src.explainability
python -c "
import os
assert os.path.exists('results/shap_summary.png')
print('SHAP plot saved')
"
```

### Done When

- SHAP summary plot exists.
- Top influential indicators are documented.

---

# Final Deliverables

After all phases are complete, Dev 5 (or any team member) should ensure:

## GitHub Repository

- [ ] All `src/` modules implemented and importable
- [ ] `README.md` with setup, architecture diagram, and how to reproduce results
- [ ] `results/` contains models, metrics, and key plots
- [ ] `requirements.txt` is complete (including `shap` if Phase 9 done)

## Final Report (Separate Document)

1. Introduction  
2. Literature Review  
3. Dataset  
4. Methodology  
5. Transformer Model  
6. Signal Generation  
7. RL Agent  
8. Results  
9. Limitations  
10. Future Work  

## Demo Narrative

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

Pipeline story: **Forecast → Signal → Decision → Profit**

---

## Quick Reference: Steps by Phase

| Phase | Name | Steps |
|-------|------|-------|
| 1 | Project Setup | 1 |
| 2 | Data Collection | 2–3 |
| 3 | Feature Engineering | 4–5 |
| 4 | Transformer Forecasting | 6–9 |
| 5 | Signal Generation | 10–12 |
| 6 | Trading Environment | 13–14 |
| 7 | Reinforcement Learning Agent | 15–16 |
| 8 | Performance Analysis | 17–18 |
| 9 | Explainability (Bonus) | 19–20 |

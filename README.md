# api-model

ML pipelines + an autonomous paper-trading bot with continuous self-training,
powered by API keys stored as encrypted secrets (never in code):

## 1. `train_model.py` - Iris classifier + LLM analysis

Trains a RandomForest on the Iris dataset (0.90 test accuracy), then sends the
results to an NVIDIA-hosted LLM (`openai/gpt-oss-20b`) for natural-language
analysis. Uses the `API_KEY` (NVIDIA) env var.

## 2. `stock_model.py` - Stock trading model

Fetches daily OHLCV data from Alpha Vantage, engineers technical features
(returns, SMA ratios, volatility, RSI-14, volume), and trains a RandomForest
to predict next-day price direction. Includes a long-only backtest vs
buy-and-hold. Latest local run (IBM): 62.5% direction accuracy vs 56.3%
baseline; backtest +6.20% vs +4.11% buy-and-hold. Not financial advice.

## 3. `cross_check.py` - Data feed cross-validation

Compares the latest close for `SYMBOL` from **Alpha Vantage**, **Polygon.io**,
and **Finnhub**; exits non-zero if any provider diverges beyond
`TOLERANCE_PCT` (default 1%). Latest run: 0.000% deviation.

## 4. `continuous_train.py` - Continuous self-training layer

Runs every CI cycle (daily after close), before the trading bot:

- **Ever-growing data cache:** each ticker's `data/<SYM>.csv` is updated
  incrementally with the newest bars - the training set grows every day.
- **Scheduled retraining:** models are retrained with walk-forward validation
  and persisted to `models/<SYM>.joblib` (fresh for `RETRAIN_DAYS`, default 7).
- **Metrics + drift detection:** every validation is logged to `metrics.json`.
  If a model's accuracy decays more than `DRIFT_TOLERANCE` (10%) below its own
  trailing average, it is flagged as drifted and the bot refuses to use it.

## 5. `paper_trading.py` - Portfolio bot, ROTATING 168-ticker universe - LIVE

Autonomous bot on the Alpaca **paper** account ($100k paper equity):

- **Loads continuously-retrained models** from `continuous_train.py` when
  fresh and drift-free; otherwise trains inline with walk-forward validation
  (tickers below 52% walk-forward accuracy are rejected).
- **Rotating universe:** a pool of **168 tickers** + index/sector ETFs; scans a
  window of 30, shifting every 7 days. Holdings are always scanned - a
  position is only sold when its own signal turns down.
- **Risk layer:** ATR stop-loss + 2:1 take-profit via Alpaca **bracket
  orders**, position sizing risk-based at 1% of equity per trade, 20-day time
  exits, and a **kill switch** that halts all trading if equity drops 3% in a
  day.
- **Track record:** every run appends to `track_record.csv`, committed back by
  CI along with `metrics.json`, `data/`, and `models/`.

## API key setup (important)

Keys are **never** stored in this repo. Locally: `cp .env.example .env` and
fill in keys. For CI, all six are encrypted GitHub Actions secrets
(`API_KEY`, `STOCK_API_KEY`, `POLYGON_API_KEY`, `FINNHUB_API_KEY`,
`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`).

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
python cross_check.py
python stock_model.py
python continuous_train.py
python paper_trading.py
```

## CI

Runs on every push **and** weekdays at 22:00 UTC (after US market close):
Iris training -> cross-check -> stock model -> **continuous training
(cache/retrain/drift)** -> live portfolio rebalance -> commit track record,
metrics, data cache, and models.

# api-model

ML pipelines + an autonomous paper-trading bot, powered by API keys stored as
encrypted secrets (never in code):

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

## 4. `paper_trading.py` - Portfolio bot, ROTATING 168-ticker universe - LIVE

Autonomous bot on the Alpaca **paper** account ($100k paper equity):

- **Rotating universe:** a pool of **168 tickers** across tech, finance,
  health, energy, consumer, industrial, materials, real estate, utilities
  + index/sector ETFs. Each run scans a window of `WINDOW_SIZE` (30);
  the window shifts every `ROTATE_EVERY_DAYS` (7) days, so after each week's
  after-close runs the bot evaluates fresh tickers. Full pool covered over
  ~17 weeks. Runtime stays ~8.5 min regardless of pool size.
- **Holdings are always scanned** - a position is only sold when its own
  signal turns down, never just because it rotated out of view.
- **Market data: Polygon only** in this step (2y daily bars, paced to the
  5 req/min free tier). Alpha Vantage stays in `cross_check.py` /
  `stock_model.py`, so no quota competition.
- **Picks and changes tickers itself:** ranks by P(up), holds the `TOP_N`
  (3) with P(up) > 0.5, equal-weight; sells fallen picks, buys new ones.
- **Track record:** every run appends to `track_record.csv` (date, mode,
  universe, scanned, picks with probabilities, actions, holdings, equity),
  committed back to this repo by CI - a full audit trail of every decision.
- Set `WATCHLIST` to disable rotation and pin a fixed universe.
- Dry-run mode if Alpaca keys unset.

First live runs: bought NVDA + META, then NVDA + AAPL (~$95k deployed).

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
python paper_trading.py
```

## CI

Runs on every push **and** weekdays at 22:00 UTC (after US market close):
Iris training -> cross-check -> stock model -> live portfolio rebalance ->
commit `track_record.csv`.

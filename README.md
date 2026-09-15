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
buy-and-hold and an optional LLM interpretation. Latest local run (IBM):
62.5% direction accuracy vs 56.3% baseline; backtest +6.20% vs +4.11%
buy-and-hold. Small sample - not financial advice.

## 3. `cross_check.py` - Data feed cross-validation

Fetches the latest close for `SYMBOL` from **Alpha Vantage**, **Polygon.io**,
and **Finnhub**, and compares them. Exits non-zero if any provider diverges
beyond `TOLERANCE_PCT` (default 1%) - catches bad data before it poisons
training. Latest run: all three providers agreed to 0.000% deviation.

## 4. `paper_trading.py` - Multi-ticker portfolio bot (Alpaca paper) - LIVE

Autonomous portfolio bot running on the Alpaca **paper** account:

- Scans `WATCHLIST` (default 12 tickers: AAPL MSFT NVDA GOOGL AMZN META TSLA
  AMD JPM V XOM UNH)
- Trains a direction model per ticker on **2 years of Polygon daily bars**
  (Alpha Vantage fallback), respecting the 5 req/min free limit
- **Picks tickers itself:** ranks by P(up), holds the `TOP_N` (default 3)
  with P(up) > 0.5, equal-weight across slots
- **Changes tickers itself:** sells holdings that fall out of the picks,
  buys new picks; goes to cash if all signals are down
- Dry-run mode if Alpaca keys are unset

First live run: picked NVDA (P=0.69) and META (P=0.67), bought 225 NVDA +
71 META (~$95k of $100k paper equity) on the Alpaca paper account.

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
Iris training -> cross-check -> stock model -> live portfolio rebalance.

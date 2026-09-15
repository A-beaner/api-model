# api-model

ML pipelines powered by API keys stored as encrypted secrets (never in code):

## 1. `train_model.py` - Iris classifier + LLM analysis

Trains a RandomForest on the Iris dataset (0.90 test accuracy), then sends the
results to an NVIDIA-hosted LLM (`openai/gpt-oss-20b`) for natural-language
analysis. Uses the `API_KEY` (NVIDIA) env var.

## 2. `stock_model.py` - Stock trading model

Fetches daily OHLCV data from Alpha Vantage, engineers technical features
(returns, SMA ratios, volatility, RSI-14, volume), and trains a RandomForest
to predict next-day price direction. Includes a long-only backtest vs
buy-and-hold and an optional LLM interpretation. Uses `STOCK_API_KEY`
(Alpha Vantage) and `API_KEY` (NVIDIA) env vars. Set `SYMBOL` to change the
ticker (default: IBM).

Latest local run (IBM): 62.5% direction accuracy vs 56.3% baseline;
backtest +6.20% vs +4.11% buy-and-hold. Small sample (100-day free feed) -
not financial advice.

## 3. `cross_check.py` - Data feed cross-validation

Fetches the latest close for `SYMBOL` from **Alpha Vantage**, **Polygon.io**,
and **Finnhub**, and compares them. Exits non-zero if any provider diverges
beyond `TOLERANCE_PCT` (default 1%) - catches bad data before it poisons
training. Includes an LLM trustworthiness verdict. Latest run: all three
providers agreed to 0.000% deviation on IBM (249.09).

## 4. `paper_trading.py` - Paper trading (Alpaca) - LIVE

Generates a daily signal from the trained model and acts on it via Alpaca's
**paper** API (`https://paper-api.alpaca.markets` - fake money, real market):

- Signal UP -> hold `POSITION_QTY` shares (buy if flat)
- Signal DOWN -> go flat (sell any position)

Status: **LIVE on a paper account** ($100k paper equity). First live signal
(IBM, 2026-09-14): FLAT (P(up)=0.47) -> HOLD, no order placed. Without Alpaca
keys the script runs in dry-run mode (prints intended order, submits nothing).

## API key setup (important)

Keys are **never** stored in this repo. Locally:

```bash
cp .env.example .env   # paste your keys into .env (gitignored)
```

For CI, keys are stored as encrypted GitHub Actions secrets (`API_KEY`,
`STOCK_API_KEY`, `POLYGON_API_KEY`, `FINNHUB_API_KEY`, `ALPACA_API_KEY`,
`ALPACA_SECRET_KEY`) - the workflow injects them automatically.

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
Iris training -> cross-check -> stock model -> live paper trade signal.

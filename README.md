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

Latest local run (IBM): 58.8% direction accuracy vs 52.9% baseline;
backtest +8.73% vs +6.59% buy-and-hold. Small sample (100-day free feed) -
not financial advice.

## 3. `cross_check.py` - Data feed cross-validation

Fetches the latest close for `SYMBOL` from **Alpha Vantage**, **Polygon.io**,
and **Finnhub**, and compares them. Exits non-zero if any provider diverges
beyond `TOLERANCE_PCT` (default 1%) - catches bad data before it poisons
training. Includes an LLM trustworthiness verdict. Latest run: all three
providers agreed to 0.000% deviation on IBM (249.09).

## API key setup (important)

Keys are **never** stored in this repo. Locally:

```bash
cp .env.example .env   # paste your keys into .env (gitignored)
# or:
export API_KEY="nvapi-..."
export STOCK_API_KEY="your-alpha-vantage-key"
export POLYGON_API_KEY="your-polygon-key"
export FINNHUB_API_KEY="your-finnhub-key"
```

For CI, all four are stored as encrypted GitHub Actions secrets (`API_KEY`,
`STOCK_API_KEY`, `POLYGON_API_KEY`, `FINNHUB_API_KEY`) - the workflow
injects them automatically.

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
python cross_check.py
python stock_model.py
```

## CI

Every push triggers `.github/workflows/run.yml`, which runs all three
pipelines using the repo secrets.

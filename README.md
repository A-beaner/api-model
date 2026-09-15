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

## API key setup (important)

Keys are **never** stored in this repo. Locally:

```bash
cp .env.example .env   # paste your keys into .env (gitignored)
# or:
export API_KEY="nvapi-..."
export STOCK_API_KEY="your-alpha-vantage-key"
```

For CI, both are stored as encrypted GitHub Actions secrets (`API_KEY`,
`STOCK_API_KEY`) - the workflow injects them automatically.

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
python stock_model.py
```

## CI

Every push triggers `.github/workflows/run.yml`, which runs both pipelines
using the repo secrets.

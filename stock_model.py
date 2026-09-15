"""
stock_model.py - Stock trading model.

Fetches daily OHLCV data from Alpha Vantage (STOCK_API_KEY env var), engineers
technical features, and trains a RandomForest to predict next-day price direction
(up/down). Includes a simple long-only backtest vs buy-and-hold on the test window,
plus an optional LLM analysis via the NVIDIA API (API_KEY env var).

Keys come from environment variables - NEVER hardcode secrets.
"""

import json
import os
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-20b"
SYMBOL = os.environ.get("SYMBOL", "IBM")


def fetch_daily(symbol, api_key):
    url = (f"{ALPHA_VANTAGE_URL}?function=TIME_SERIES_DAILY&symbol={symbol}"
           f"&outputsize=compact&apikey={api_key}")
    with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as r:
        data = json.loads(r.read())
    ts = data.get("Time Series (Daily)")
    if not ts:
        raise RuntimeError(f"Alpha Vantage returned no data: {str(data)[:200]}")
    df = pd.DataFrame.from_dict(ts, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = ["open", "high", "low", "close", "volume"]
    return df


def engineer_features(df):
    out = df.copy()
    out["return_1d"] = out["close"].pct_change()
    out["return_5d"] = out["close"].pct_change(5)
    out["sma_ratio_5_20"] = out["close"].rolling(5).mean() / out["close"].rolling(20).mean()
    out["volatility_10"] = out["return_1d"].rolling(10).std()
    out["volume_change"] = out["volume"].pct_change()
    # RSI-14
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = 100 - (100 / (1 + gain / loss))
    # Target: 1 if tomorrow closes higher than today
    out["target"] = (out["close"].shift(-1) > out["close"]).astype(int)
    return out.dropna()


def llm_analysis(summary, api_key):
    prompt = (
        "A RandomForest model predicts next-day stock direction. Results:\n"
        f"{summary}\n\nIn 3-4 sentences, interpret this for a trader: "
        "is the edge meaningful, what are the risks, and one improvement."
    )
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(NVIDIA_API_URL, data=payload, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main():
    stock_key = os.environ.get("STOCK_API_KEY")
    if not stock_key:
        raise SystemExit("STOCK_API_KEY env var is required (Alpha Vantage key).")
    nvidia_key = os.environ.get("API_KEY")
    print(f"Keys loaded: STOCK_API_KEY ***{stock_key[-4:]}"
          + (f", API_KEY ***{nvidia_key[-4:]}" if nvidia_key else ""))

    print(f"Fetching daily data for {SYMBOL} from Alpha Vantage...")
    df = fetch_daily(SYMBOL, stock_key)
    print(f"{len(df)} trading days: {df.index.min().date()} -> {df.index.max().date()}")

    data = engineer_features(df)
    features = ["return_1d", "return_5d", "sma_ratio_5_20", "volatility_10", "volume_change", "rsi_14"]
    X, y = data[features], data["target"]

    # Time-based split - no shuffling for time series
    split = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"Train: {len(X_train)} days | Test: {len(X_test)} days")

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"\nDirection accuracy: {acc:.4f} (majority-class baseline: {baseline:.4f})")
    print(classification_report(y_test, preds, target_names=["down", "up"], zero_division=0))

    # Simple long-only backtest on the test window
    test_returns = data["return_1d"].iloc[split:].shift(-1).dropna()  # next-day returns
    strat = (preds[:len(test_returns)] == 1) * test_returns.values
    strat_cum = (1 + pd.Series(strat)).prod() - 1
    bh_cum = (1 + test_returns).prod() - 1
    print(f"Backtest (test window): strategy {strat_cum:+.2%} vs buy-and-hold {bh_cum:+.2%}")

    joblib.dump(model, "stock_model.joblib")
    print("Model saved to stock_model.joblib")

    if nvidia_key:
        summary = (f"Symbol {SYMBOL}, {len(data)} usable days. Accuracy {acc:.2%} vs baseline "
                   f"{baseline:.2%}. Strategy backtest {strat_cum:+.2%} vs buy-hold {bh_cum:+.2%}.")
        print(f"\n--- LLM analysis ({LLM_MODEL} via NVIDIA API) ---")
        try:
            print(llm_analysis(summary, nvidia_key))
        except Exception as e:
            print(f"LLM analysis failed: {e}")


if __name__ == "__main__":
    main()

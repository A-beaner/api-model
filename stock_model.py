"""
stock_model.py - Stock trading model.

Daily OHLCV candles come from Alpaca market data when keys are set (same
source as the charts and the orders), with Polygon and Alpha Vantage as
fallbacks. Engineers technical + candlestick features and trains a
RandomForest to predict next-day price direction (up/down). Includes a
simple long-only backtest vs buy-and-hold on the test window, plus an
optional LLM analysis via the NVIDIA API (API_KEY env var).

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
FEATURES = ["return_1d", "return_5d", "sma_ratio_5_20", "volatility_10", "volume_change", "rsi_14",
            "body_pct", "upper_wick", "lower_wick", "hl_range", "gap_pct", "up_streak"]

ALPACA_DATA_URL = "https://data.alpaca.markets"


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


def alpaca_daily(symbol, days=730, key=None, secret=None):
    """Fetch daily bars (candles) from Alpaca market data - free with the
    paper-trading keys. Used by the bot so charts and orders come from one source."""
    key = key or os.environ.get("ALPACA_API_KEY")
    secret = secret or os.environ.get("ALPACA_SECRET_KEY")
    if not (key and secret):
        raise RuntimeError("Alpaca keys not set")
    from datetime import datetime, timedelta
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    url = (f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars?timeframe=1Day"
           f"&start={start}&end={end}&limit=10000&adjustment=all&feed=iex")
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    bars = data.get("bars") or []
    if not bars:
        raise RuntimeError(f"Alpaca bars: {str(data)[:150]}")
    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["t"]).dt.tz_localize(None)
    df = df.set_index("date").rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def get_daily(symbol, polygon_key=None, days=730):
    """Unified daily-candle loader: Alpaca first (same source as the charts and
    orders), Polygon as fallback."""
    try:
        return alpaca_daily(symbol, days=days)
    except Exception as e:
        print(f"  {symbol}: Alpaca bars failed ({e}), falling back to Polygon")
    if not polygon_key:
        raise
    end_d = pd.Timestamp.utcnow().date()
    start_d = end_d - pd.Timedelta(days=days)
    url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_d}/{end_d}"
           f"?adjusted=true&sort=asc&limit=50000&apiKey={polygon_key}")
    with urllib.request.urlopen(urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
        data = json.loads(r.read())
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"Polygon: {str(data)[:150]}")
    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.set_index("date").rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def engineer_features(df):
    """Compute technical + candlestick features. The final row has target=NaN
    (tomorrow unknown) and is kept for live prediction; callers training a
    model must dropna on target."""
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
    # --- Candlestick features: read the shape of each bar ---
    out["body_pct"] = (out["close"] - out["open"]) / out["open"]           # candle body (direction + size)
    out["upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["open"]   # rejection above
    out["lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["open"]    # rejection below
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]            # intraday volatility
    out["gap_pct"] = out["open"] / out["close"].shift() - 1                # overnight gap
    # consecutive up-candles (down-candles count negative)
    up = (out["close"] > out["open"]).astype(int)
    down = (out["close"] < out["open"]).astype(int)
    streak = up - down
    out["up_streak"] = streak.groupby((streak != streak.shift()).cumsum()).cumsum()

    # Target: 1 if tomorrow closes higher than today (NaN for the latest day)
    next_close = out["close"].shift(-1)
    out["target"] = (next_close > out["close"]).where(next_close.notna()).astype("float")
    return out.dropna(subset=FEATURES)


def train_model(data):
    train = data.dropna(subset=["target"])
    X, y = train[FEATURES], train["target"]
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, train


def llm_analysis(summary, api_key):
    prompt = (
        "A RandomForest model predicts next-day stock direction. Results:\n"
        f"{summary}\n\nIn 3-4 sentences, interpret this for a trader: "
        "is the edge meaningful, what are the risks, and one improvement."
    )
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(NVIDIA_API_URL, data=payload, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    return (msg.get("content") or "").strip() or "(model returned empty content)"


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
    train = data.dropna(subset=["target"])
    X, y = train[FEATURES], train["target"]

    # Time-based split - no shuffling for time series
    split = int(len(train) * 0.8)
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
    test_returns = train["return_1d"].iloc[split:].shift(-1).dropna()
    strat = (preds[:len(test_returns)] == 1) * test_returns.values
    strat_cum = (1 + pd.Series(strat)).prod() - 1
    bh_cum = (1 + test_returns).prod() - 1
    print(f"Backtest (test window): strategy {strat_cum:+.2%} vs buy-and-hold {bh_cum:+.2%}")

    joblib.dump(model, "stock_model.joblib")
    print("Model saved to stock_model.joblib")

    if nvidia_key:
        summary = (f"Symbol {SYMBOL}, {len(train)} usable days. Accuracy {acc:.2%} vs baseline "
                   f"{baseline:.2%}. Strategy backtest {strat_cum:+.2%} vs buy-hold {bh_cum:+.2%}.")
        print(f"\n--- LLM analysis ({LLM_MODEL} via NVIDIA API) ---")
        try:
            print(llm_analysis(summary, nvidia_key))
        except Exception as e:
            print(f"LLM analysis failed: {e}")


if __name__ == "__main__":
    main()

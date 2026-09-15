"""
cross_check.py - Cross-validate stock prices across multiple data providers.

Fetches the latest daily close for SYMBOL from Alpha Vantage, Polygon.io, and
Finnhub, then compares them. Flags discrepancies beyond TOLERANCE_PCT (default 1%).
Exits non-zero if any provider pair diverges beyond tolerance - catches bad data
feeds before they poison trading models.

Keys come from environment variables - NEVER hardcode secrets:
  STOCK_API_KEY    (Alpha Vantage)
  POLYGON_API_KEY  (Polygon.io)
  FINNHUB_API_KEY  (Finnhub)
A provider is skipped if its key is not set.
"""

import json
import os
import sys
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SYMBOL = os.environ.get("SYMBOL", "IBM")
TOLERANCE_PCT = float(os.environ.get("TOLERANCE_PCT", "1.0"))
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-20b"


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def alpha_vantage_close(symbol, key):
    data = get_json(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
                    f"&symbol={symbol}&outputsize=compact&apikey={key}")
    ts = data.get("Time Series (Daily)")
    if not ts:
        raise RuntimeError(f"Alpha Vantage: {str(data)[:150]}")
    latest = sorted(ts.keys())[-1]
    return float(ts[latest]["4. close"]), latest


def polygon_close(symbol, key):
    data = get_json(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={key}")
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"Polygon: {str(data)[:150]}")
    return float(results[0]["c"]), "prev trading day"


def finnhub_close(symbol, key):
    data = get_json(f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={key}")
    if not data.get("c"):
        raise RuntimeError(f"Finnhub: {str(data)[:150]}")
    return float(data["c"]), "latest quote"


def llm_verdict(summary, api_key):
    prompt = (
        "Stock price cross-check across data providers:\n" + summary +
        "\n\nIn 2-3 sentences: is this data trustworthy for a trading model, "
        "and what would you watch out for?"
    )
    payload = json.dumps({"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 600, "temperature": 0.3}).encode()
    req = urllib.request.Request(NVIDIA_API_URL, data=payload, headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
        return (msg.get("content") or "").strip() or "(model returned empty content)"


def main():
    providers = [
        ("Alpha Vantage", "STOCK_API_KEY", alpha_vantage_close),
        ("Polygon.io", "POLYGON_API_KEY", polygon_close),
        ("Finnhub", "FINNHUB_API_KEY", finnhub_close),
    ]
    prices = {}
    for name, env_var, fn in providers:
        key = os.environ.get(env_var)
        if not key:
            print(f"- {name}: skipped ({env_var} not set)")
            continue
        try:
            price, asof = fn(SYMBOL, key)
            prices[name] = price
            print(f"+ {name:15s} {SYMBOL} close: {price:.2f}  ({asof})  [key ***{key[-4:]}]")
        except Exception as e:
            print(f"! {name}: failed - {e}")

    if len(prices) < 2:
        sys.exit("Need at least 2 providers to cross-check.")

    values = list(prices.values())
    mean_price = sum(values) / len(values)
    max_dev = max(abs(v - mean_price) / mean_price * 100 for v in values)
    print(f"\nProviders compared: {len(prices)} | mean close: {mean_price:.2f} | "
          f"max deviation: {max_dev:.3f}% (tolerance {TOLERANCE_PCT}%)")

    summary = (f"{SYMBOL}: " + ", ".join(f"{k}={v:.2f}" for k, v in prices.items()) +
               f". Max deviation {max_dev:.3f}% vs tolerance {TOLERANCE_PCT}%.")
    nvidia_key = os.environ.get("API_KEY")
    if nvidia_key:
        print(f"\n--- LLM verdict ({LLM_MODEL} via NVIDIA API) ---")
        try:
            print(llm_verdict(summary, nvidia_key))
        except Exception as e:
            print(f"LLM verdict failed: {e}")

    if max_dev > TOLERANCE_PCT:
        print(f"\nFAIL: price feeds diverge by {max_dev:.3f}% (> {TOLERANCE_PCT}%)")
        sys.exit(1)
    print("\nPASS: all provider prices agree within tolerance.")


if __name__ == "__main__":
    main()

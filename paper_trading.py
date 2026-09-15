"""
paper_trading.py - Multi-ticker paper trading bot (Alpaca paper API).

Scans a WATCHLIST of tickers, trains a direction model per ticker on Polygon
daily bars (2y history; Alpha Vantage fallback), then picks the TOP_N tickers
with the strongest up-signal (P(up) > 0.5) and rebalances the Alpaca paper
account to hold them equal-weight: sells positions that dropped out of the
picks, buys new picks. Can pick and change tickers autonomously each run.

Keys come from environment variables - NEVER hardcode secrets:
  POLYGON_API_KEY     (Polygon.io - primary market data, 2y daily bars)
  STOCK_API_KEY       (Alpha Vantage - fallback market data)
  ALPACA_API_KEY      (Alpaca paper key id)      -> dry-run if unset
  ALPACA_SECRET_KEY   (Alpaca paper secret)      -> dry-run if unset
  API_KEY             (NVIDIA - optional AI commentary)
Config env vars: WATCHLIST (csv), TOP_N (default 3).
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from stock_model import engineer_features, FEATURES

ALPACA_BASE = "https://paper-api.alpaca.markets"  # paper environment only
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-20b"

WATCHLIST = [t.strip().upper() for t in os.environ.get(
    "WATCHLIST", "AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,AMD,JPM,V,XOM,UNH").split(",") if t.strip()]
TOP_N = int(os.environ.get("TOP_N", "3"))
POLYGON_MIN_INTERVAL = 12.2  # free tier: 5 requests/min
_last_polygon_call = [0.0]


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def polygon_daily(symbol, key):
    wait = POLYGON_MIN_INTERVAL - (time.time() - _last_polygon_call[0])
    if wait > 0:
        time.sleep(wait)
    end = datetime.utcnow().date()
    start = end - timedelta(days=730)
    url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
           f"?adjusted=true&sort=asc&limit=50000&apiKey={key}")
    _last_polygon_call[0] = time.time()
    data = get_json(url)
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"Polygon: {str(data)[:150]}")
    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.set_index("date").rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_history(symbol, poly_key, av_key):
    if poly_key:
        try:
            return polygon_daily(symbol, poly_key), "polygon"
        except Exception as e:
            print(f"  {symbol}: Polygon failed ({e}); trying Alpha Vantage")
    if av_key:
        from stock_model import fetch_daily
        return fetch_daily(symbol, av_key), "alphavantage"
    raise RuntimeError("no market data key available")


def scan(symbols, poly_key, av_key):
    rows = []
    for sym in symbols:
        try:
            df, src = fetch_history(sym, poly_key, av_key)
            data = engineer_features(df)
            train = data.dropna(subset=["target"])
            X, y = train[FEATURES], train["target"]
            model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
            model.fit(X, y)
            latest = data[FEATURES].iloc[-1:]
            proba = model.predict_proba(latest)[0]
            p_up = float(proba[1]) if len(proba) > 1 else 0.0
            price = float(df["close"].iloc[-1])
            rows.append({"symbol": sym, "p_up": p_up, "price": price,
                         "days": len(train), "src": src})
            print(f"  {sym:6s} P(up)={p_up:.2f} close=${price:,.2f} ({len(train)} days, {src})")
        except Exception as e:
            print(f"  {sym}: skipped - {e}")
    return rows


def alpaca(method, path, key, secret, body=None):
    req = urllib.request.Request(
        ALPACA_BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                 "Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


def llm_commentary(context, api_key):
    prompt = (
        "A multi-ticker paper-trading bot rebalanced its portfolio. Context:\n" + context +
        "\n\nIn 2-3 sentences, comment as a risk manager: is the selection logic "
        "reasonable and what should be watched?"
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
    poly_key = os.environ.get("POLYGON_API_KEY")
    av_key = os.environ.get("STOCK_API_KEY")
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")
    nvidia_key = os.environ.get("API_KEY")
    dry_run = not (alpaca_key and alpaca_secret)

    print(f"Mode: {'DRY-RUN (no Alpaca keys)' if dry_run else 'LIVE PAPER TRADING'}")
    print(f"Watchlist: {', '.join(WATCHLIST)} | Picks: top {TOP_N} with P(up)>0.5\n")

    rows = scan(WATCHLIST, poly_key, av_key)
    if not rows:
        raise SystemExit("No ticker data available - check market data keys.")

    ranked = sorted(rows, key=lambda r: r["p_up"], reverse=True)
    picks = [r for r in ranked if r["p_up"] > 0.5][:TOP_N]
    print(f"\nPicks: {[p['symbol'] for p in picks] if picks else 'NONE - all signals down, going to cash'}")

    target = {p["symbol"] for p in picks}
    actions = []

    if dry_run:
        print("DRY-RUN: would sell positions not in picks, buy picks not held "
              f"(~equal weight across {TOP_N} slots)")
    else:
        acct = alpaca("GET", "/v2/account", alpaca_key, alpaca_secret)
        equity = float(acct["equity"])
        positions = alpaca("GET", "/v2/positions", alpaca_key, alpaca_secret)
        held = {p["symbol"]: int(float(p["qty"])) for p in positions}
        print(f"Paper account: equity=${equity:,.2f} | holding: {held or 'nothing'}")

        # 1) Sell positions that are no longer picks
        for sym, qty in held.items():
            if sym not in target:
                try:
                    o = alpaca("POST", "/v2/orders", alpaca_key, alpaca_secret,
                               {"symbol": sym, "qty": str(qty), "side": "sell",
                                "type": "market", "time_in_force": "day"})
                    actions.append(f"SELL {qty} {sym} -> {o.get('status')}")
                except Exception as e:
                    actions.append(f"SELL {qty} {sym} FAILED: {e}")

        # 2) Buy new picks, equal weight across TOP_N slots
        budget = equity * 0.95 / TOP_N
        for pick in picks:
            sym = pick["symbol"]
            if sym in held:
                continue
            qty = int(budget // pick["price"])
            if qty < 1:
                actions.append(f"BUY {sym} skipped (price ${pick['price']:,.2f} > slot budget)")
                continue
            try:
                o = alpaca("POST", "/v2/orders", alpaca_key, alpaca_secret,
                           {"symbol": sym, "qty": str(qty), "side": "buy",
                            "type": "market", "time_in_force": "day"})
                actions.append(f"BUY {qty} {sym} (~${qty * pick['price']:,.0f}) -> {o.get('status')}")
            except Exception as e:
                actions.append(f"BUY {qty} {sym} FAILED: {e}")

        print("\nRebalance actions:")
        for a in actions:
            print(f"  {a}")
        if not actions:
            print("  none - portfolio already matches picks")

    if nvidia_key:
        context = (f"Watchlist {len(WATCHLIST)} tickers scanned. Signals: "
                   + ", ".join(f"{r['symbol']} P(up)={r['p_up']:.2f}" for r in ranked)
                   + f". Picks: {[p['symbol'] for p in picks]}. Actions: {actions or 'none (dry-run)'}.")
        print(f"\n--- AI portfolio commentary ({LLM_MODEL} via NVIDIA API) ---")
        try:
            print(llm_commentary(context, nvidia_key))
        except Exception as e:
            print(f"LLM commentary failed: {e}")


if __name__ == "__main__":
    main()

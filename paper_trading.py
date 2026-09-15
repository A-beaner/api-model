"""
paper_trading.py - Multi-ticker paper trading bot (Alpaca paper API) with a
ROTATING ticker universe and a persistent track record.

Each run picks a window of WINDOW_SIZE tickers from TICKER_POOL (default 168
liquid US names + ETFs). The window rotates every ROTATE_EVERY_DAYS days
(default 7), so the bot regularly evaluates fresh tickers after market close
and covers the whole pool over ~17 weeks. Currently-held positions are ALWAYS
scanned too, so a holding is only sold when its own signal turns down - never
just because it rotated out of view.

Per run: train a direction model per ticker (Polygon 2y daily bars), rank by
P(up), hold the TOP_N with P(up) > 0.5, equal-weight. Sells holdings that
fall out of the picks, buys new picks. Appends every run to track_record.csv
(committed back to the repo by CI) for a full audit trail.

Market data: Polygon only (this step no longer touches Alpha Vantage).
Keys come from environment variables - NEVER hardcode secrets:
  POLYGON_API_KEY     (Polygon.io - market data, 2y daily bars)
  ALPACA_API_KEY      (Alpaca paper key id)      -> dry-run if unset
  ALPACA_SECRET_KEY   (Alpaca paper secret)      -> dry-run if unset
  API_KEY             (NVIDIA - optional AI commentary)
Config env vars:
  TICKER_POOL         (csv, default 168 tickers) - the rotating universe pool
  WINDOW_SIZE         (default 30)               - tickers scanned per run
  ROTATE_EVERY_DAYS   (default 7)                - how often the window shifts
  TOP_N               (default 3)                - max simultaneous positions
  WATCHLIST           (optional csv)             - if set, disables rotation
  TRACK_RECORD_FILE   (default track_record.csv)
  DAILY_TRADE_LIMIT_USD (default 50000)           - max new $ bought per run/day
"""

import csv
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

DEFAULT_POOL = (
    "AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,AMD,AVGO,NFLX,CRM,ORCL,IBM,INTC,QCOM,"
    "TXN,MU,ADBE,PYPL,PLTR,UBER,ABNB,SHOP,CRWD,COIN,NOW,SNPS,CDNS,PANW,DDOG,NET,"
    "JPM,V,MA,BAC,MS,GS,C,WFC,AXP,BLK,SCHW,COF,USB,PNC,TFC,BK,AIG,MET,PRU,CME,"
    "UNH,JNJ,LLY,PFE,MRK,ABBV,GILD,AMGN,TMO,ABT,DHR,BMY,ISRG,MDT,SYK,BDX,REGN,VRTX,ZTS,CI,"
    "XOM,CVX,COP,SLB,EOG,MPC,PSX,VLO,OXY,HAL,KMI,WMB,"
    "WMT,COST,HD,MCD,KO,NKE,PG,PEP,PM,MO,SBUX,TGT,LOW,TJX,DG,DLTR,ROST,LULU,YUM,CMG,EL,CL,"
    "CAT,GE,LMT,HON,UPS,RTX,DE,BA,MMM,GD,NOC,FDX,CSX,UNP,ETN,EMR,PH,"
    "T,VZ,CMCSA,CHTR,TMUS,DIS,WBD,EA,TTWO,"
    "LIN,APD,SHW,ECL,FCX,NEM,DOW,DD,NUE,MLM,"
    "AMT,PLD,EQIX,SPG,O,WELL,DLR,AVB,PSA,CCI,"
    "NEE,DUK,SO,AEP,D,"
    "SPY,QQQ,DIA,IWM,VOO,XLF,XLE,XLV,XLK,XLI,XLP,XLU"
)
TICKER_POOL = [t.strip().upper() for t in os.environ.get(
    "TICKER_POOL", DEFAULT_POOL).split(",") if t.strip()]
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "30"))
ROTATE_EVERY_DAYS = int(os.environ.get("ROTATE_EVERY_DAYS", "7"))
TOP_N = int(os.environ.get("TOP_N", "3"))
DAILY_TRADE_LIMIT_USD = float(os.environ.get("DAILY_TRADE_LIMIT_USD", "50000"))  # max $ of new buys per run/day
FIXED_WATCHLIST = [t.strip().upper() for t in os.environ.get(
    "WATCHLIST", "").split(",") if t.strip()]
TRACK_RECORD_FILE = os.environ.get("TRACK_RECORD_FILE", "track_record.csv")
POLYGON_MIN_INTERVAL = 12.2  # free tier: 5 requests/min
_last_polygon_call = [0.0]


def rotating_window(pool, window_size, period, step=10):
    """Deterministic rotating slice of the pool. `period` increments every
    ROTATE_EVERY_DAYS days, so the universe shifts on that cadence and cycles
    through the whole pool over time."""
    if window_size >= len(pool):
        return list(pool)
    offset = (period * step) % len(pool)
    doubled = pool + pool
    return doubled[offset:offset + window_size]


def current_universe(now=None):
    if FIXED_WATCHLIST:
        return FIXED_WATCHLIST, "fixed WATCHLIST (rotation disabled)"
    now = now or datetime.utcnow()
    period = int(now.timestamp() // 86400) // ROTATE_EVERY_DAYS
    window = rotating_window(TICKER_POOL, WINDOW_SIZE, period)
    return window, f"rotating window (pool={len(TICKER_POOL)}, period={period})"


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


def scan(symbols, poly_key):
    rows = []
    for sym in symbols:
        try:
            df = polygon_daily(sym, poly_key)
            data = engineer_features(df)
            train = data.dropna(subset=["target"])
            X, y = train[FEATURES], train["target"]
            model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
            model.fit(X, y)
            latest = data[FEATURES].iloc[-1:]
            proba = model.predict_proba(latest)[0]
            p_up = float(proba[1]) if len(proba) > 1 else 0.0
            price = float(df["close"].iloc[-1])
            rows.append({"symbol": sym, "p_up": p_up, "price": price, "days": len(train)})
            print(f"  {sym:6s} P(up)={p_up:.2f} close=${price:,.2f} ({len(train)} days)")
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


def log_track_record(universe_desc, scanned, picks, actions, held, equity, dry_run):
    row = {
        "date_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "mode": "dry-run" if dry_run else "paper",
        "universe": universe_desc,
        "scanned": scanned,
        "picks": "|".join(f"{p['symbol']}:{p['p_up']:.2f}" for p in picks),
        "actions": "|".join(actions),
        "holdings": "|".join(f"{s}:{q}" for s, q in held.items()),
        "equity": f"{equity:.2f}",
    }
    exists = os.path.exists(TRACK_RECORD_FILE)
    with open(TRACK_RECORD_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"\nTrack record appended to {TRACK_RECORD_FILE}")


def main():
    poly_key = os.environ.get("POLYGON_API_KEY")
    if not poly_key:
        raise SystemExit("POLYGON_API_KEY env var is required.")
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")
    nvidia_key = os.environ.get("API_KEY")
    dry_run = not (alpaca_key and alpaca_secret)

    universe, universe_desc = current_universe()
    print(f"Mode: {'DRY-RUN (no Alpaca keys)' if dry_run else 'LIVE PAPER TRADING'}")
    print(f"Universe: {universe_desc} | scanning {len(universe)} tickers | top {TOP_N}")

    held = {}
    equity = 0.0
    if not dry_run:
        acct = alpaca("GET", "/v2/account", alpaca_key, alpaca_secret)
        equity = float(acct["equity"])
        positions = alpaca("GET", "/v2/positions", alpaca_key, alpaca_secret)
        held = {p["symbol"]: int(float(p["qty"])) for p in positions}
        extras = [s for s in held if s not in universe]
        if extras:
            universe = universe + extras
            print(f"Also scanning held positions outside this window: {extras}")
        print(f"Paper account: equity=${equity:,.2f} | holding: {held or 'nothing'}\n")

    rows = scan(universe, poly_key)
    if not rows:
        raise SystemExit("No ticker data available - check POLYGON_API_KEY.")

    ranked = sorted(rows, key=lambda r: r["p_up"], reverse=True)
    picks = [r for r in ranked if r["p_up"] > 0.5][:TOP_N]
    print(f"\nPicks: {[p['symbol'] for p in picks] if picks else 'NONE - all signals down, going to cash'}")

    target = {p["symbol"] for p in picks}
    actions = []

    if dry_run:
        actions.append(f"DRY-RUN: would hold top {TOP_N} of {len(ranked)} scanned")
        print(actions[0])
    else:
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
        daily_budget = min(equity * 0.95, DAILY_TRADE_LIMIT_USD)
        budget = daily_budget / TOP_N
        print(f"Daily trade limit: ${DAILY_TRADE_LIMIT_USD:,.0f} -> slot budget ${budget:,.2f}")
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

    log_track_record(universe_desc, len(rows), picks, actions, held, equity, dry_run)

    if nvidia_key:
        context = (f"Universe: {universe_desc}, {len(rows)} tickers scanned. Signals: "
                   + ", ".join(f"{r['symbol']} P(up)={r['p_up']:.2f}" for r in ranked)
                   + f". Picks: {[p['symbol'] for p in picks]}. Actions: {actions}.")
        print(f"\n--- AI portfolio commentary ({LLM_MODEL} via NVIDIA API) ---")
        try:
            print(llm_commentary(context, nvidia_key))
        except Exception as e:
            print(f"LLM commentary failed: {e}")


if __name__ == "__main__":
    main()

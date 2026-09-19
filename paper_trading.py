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
  MAX_PORTFOLIO_VALUE (default 100000)         - hard cap: total capital deployed, USD
  MAX_TRADE_VALUE     (default 50000)          - max notional per single order, USD
  MIN_CASH_PCT        (default 0.05)           - cash buffer never traded, fraction
  STOP_LOSS_DROP      (default 3.0)            - sell if a position's total value drops
                                               this many USD below its peak while held
  STATE_FILE          (default stop_state.json) - peak prices per holding (persisted)
  MODEL_MEMORY_FILE   (default model_memory.csv) - realized outcomes of past trades
  PERFORMANCE_WEIGHT  (default 1.0)            - how strongly past returns move rankings
  MEMORY_WINDOW       (default 10)             - trades per ticker used in the feedback score
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
FIXED_WATCHLIST = [t.strip().upper() for t in os.environ.get(
    "WATCHLIST", "").split(",") if t.strip()]
TRACK_RECORD_FILE = os.environ.get("TRACK_RECORD_FILE", "track_record.csv")

# ---- Trading limits (plan to trade with less than $100,000) ----
MAX_PORTFOLIO_VALUE = float(os.environ.get("MAX_PORTFOLIO_VALUE", "100000"))
MAX_TRADE_VALUE = float(os.environ.get("MAX_TRADE_VALUE", "50000"))
MIN_CASH_PCT = float(os.environ.get("MIN_CASH_PCT", "0.05"))

# ---- Trailing stop-loss ----
# Sell a holding if its TOTAL POSITION VALUE falls STOP_LOSS_DROP dollars or more
# below the peak value it reached while we owned it, but ONLY while the price is
# still at or above the entry price (never sell at a loss).
STOP_LOSS_DROP = float(os.environ.get("STOP_LOSS_DROP", "3.0"))  # USD, per position
STATE_FILE = os.environ.get("STATE_FILE", "stop_state.json")

# ---- Outcome feedback (learn from past mistakes) ----
# Every sell records its realized return to MODEL_MEMORY_FILE. Future runs adjust
# each tickers ranking by PERFORMANCE_WEIGHT * mean return of its last
# MEMORY_WINDOW trades, so tickers the model repeatedly loses on are down-ranked
# and consistent winners are up-ranked.
MODEL_MEMORY_FILE = os.environ.get("MODEL_MEMORY_FILE", "model_memory.csv")
PERFORMANCE_WEIGHT = float(os.environ.get("PERFORMANCE_WEIGHT", "1.0"))
MEMORY_WINDOW = int(os.environ.get("MEMORY_WINDOW", "10"))

# ---- Trading algorithm tuning ----
# BUY: composite score = 0.55*P(up) + 0.20*trend + 0.15*momentum + 0.10*rsi,
# plus the outcome-memory adjustment. Only buy if the total exceeds MIN_SCORE.
# Sizing is proportional to conviction (score-weighted), capped by MAX_TRADE_VALUE.
# SELL: (1) take-profit at +TAKE_PROFIT_PCT from entry, (2) $ trailing stop on
# total position value, (3) signal/trend breakdown (falls out of picks).
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.55"))
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "0.15"))
MAX_DAILY_VOL = float(os.environ.get("MAX_DAILY_VOL", "0.04"))  # skip wilder tickers
POLYGON_MIN_INTERVAL = 12.2  # free tier: 5 requests/min
_last_polygon_call = [0.0]


def load_peak_state():
    """Persisted per-symbol peak POSITION VALUE (qty*price) since purchase."""
    try:
        with open(STATE_FILE) as f:
            return {k: float(v) for k, v in json.load(f).items()}
    except Exception:
        return {}


def save_peak_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_memory():
    """All recorded trade outcomes (realized returns) from MODEL_MEMORY_FILE."""
    try:
        with open(MODEL_MEMORY_FILE, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def record_outcome(symbol, entry, exit_price, reason):
    """Append one closed-trade outcome so future runs can learn from it."""
    ret = (exit_price - entry) / entry
    row = {"date_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
           "symbol": symbol, "entry_price": f"{entry:.2f}",
           "exit_price": f"{exit_price:.2f}", "return_pct": f"{ret:.4f}",
           "reason": reason}
    exists = os.path.exists(MODEL_MEMORY_FILE)
    with open(MODEL_MEMORY_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    return ret


def memory_scores(memory):
    """Per-symbol mean realized return over its last MEMORY_WINDOW trades."""
    by_sym = {}
    for row in memory:
        try:
            by_sym.setdefault(row["symbol"], []).append(float(row["return_pct"]))
        except (KeyError, ValueError):
            continue
    return {s: sum(rets[-MEMORY_WINDOW:]) / len(rets[-MEMORY_WINDOW:])
            for s, rets in by_sym.items() if rets}


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


def indicators(df):
    """Technical indicators for the composite score."""
    close = df["close"]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    price = float(close.iloc[-1])
    # trend: 1.0 strong up, 0.5 mild up, 0.0 down
    trend = 1.0 if price > sma20 > sma50 else (0.5 if price > sma20 else 0.0)
    # momentum: 20-day return clipped to [-1, 1] then shifted to [0, 1]
    mom20 = float(close.pct_change(20).iloc[-1])
    momentum = min(max(mom20 / 0.10, -1.0), 1.0) / 2 + 0.5  # +/-10% maps to 0..1
    # RSI(14): reward healthy 40-65, penalize overbought >75 and weak <35
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] else 100.0
    rsi = 100 - 100 / (1 + rs)
    if 40 <= rsi <= 65:
        rsi_score = 1.0
    elif rsi > 65:
        rsi_score = max(0.0, 1 - (rsi - 65) / 20)
    else:
        rsi_score = max(0.0, rsi / 40)
    vol20 = float(close.pct_change().rolling(20).std().iloc[-1])
    return {"price": price, "trend": trend, "momentum": momentum,
            "rsi": float(rsi), "rsi_score": rsi_score, "vol20": vol20,
            "sma20": float(sma20)}


def composite_score(p_up, ind):
    """Multi-factor buy score in [0,1]-ish: model conviction + trend + momentum + RSI."""
    return (0.55 * p_up + 0.20 * ind["trend"]
            + 0.15 * ind["momentum"] + 0.10 * ind["rsi_score"])


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
            ind = indicators(df)
            price = ind["price"]
            algo = composite_score(p_up, ind)
            rows.append({"symbol": sym, "p_up": p_up, "price": price,
                         "days": len(train), "ind": ind, "algo": algo})
            print(f"  {sym:6s} P(up)={p_up:.2f} trend={ind['trend']:.1f} "
                  f"mom={ind['momentum']:.2f} RSI={ind['rsi']:.0f} vol={ind['vol20']:.3f} "
                  f"-> algo={algo:.2f} close=${price:,.2f}")
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
    print(f"Limits: max portfolio ${MAX_PORTFOLIO_VALUE:,.0f} | "
          f"max per trade ${MAX_TRADE_VALUE:,.0f} | cash buffer {MIN_CASH_PCT:.0%} | "
          f"stop-loss -${STOP_LOSS_DROP:,.0f} from position peak value (never below entry)")
    print(f"Algorithm: min score {MIN_SCORE} | take-profit +{TAKE_PROFIT_PCT:.0%} | "
          f"max daily vol {MAX_DAILY_VOL:.0%} | score-weighted sizing")

    held = {}
    entries = {}
    equity = 0.0
    if not dry_run:
        acct = alpaca("GET", "/v2/account", alpaca_key, alpaca_secret)
        equity = float(acct["equity"])
        positions = alpaca("GET", "/v2/positions", alpaca_key, alpaca_secret)
        held = {p["symbol"]: int(float(p["qty"])) for p in positions}
        entries = {p["symbol"]: float(p["avg_entry_price"]) for p in positions}
        extras = [s for s in held if s not in universe]
        if extras:
            universe = universe + extras
            print(f"Also scanning held positions outside this window: {extras}")
        print(f"Paper account: equity=${equity:,.2f} | holding: {held or 'nothing'}\n")

    rows = scan(universe, poly_key)
    if not rows:
        raise SystemExit("No ticker data available - check POLYGON_API_KEY.")

    # Learn from past outcomes: shift each ticker's score by the average realized
    # return of its previous trades (winners up-ranked, repeat losers down-ranked).
    scores = memory_scores(load_memory())
    for r in rows:
        mem = scores.get(r["symbol"])
        r["memory_adj"] = PERFORMANCE_WEIGHT * mem if mem is not None else 0.0
        r["score"] = r["algo"] + r["memory_adj"]
        if mem is not None:
            print(f"  {r['symbol']:6s} memory: avg return {mem:+.2%} over past trades "
                  f"-> score {r['algo']:.2f} {r['memory_adj']:+.2f} = {r['score']:.2f}")
        if r["ind"]["vol20"] > MAX_DAILY_VOL:
            r["score"] = -1  # too volatile - excluded from buys
            print(f"  {r['symbol']:6s} excluded: daily vol {r['ind']['vol20']:.3f} > {MAX_DAILY_VOL}")
        if r["ind"]["rsi"] > 75:
            r["score"] = -1  # overbought - don't chase
            print(f"  {r['symbol']:6s} excluded: RSI {r['ind']['rsi']:.0f} overbought")

    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
    picks = [r for r in ranked if r["score"] > MIN_SCORE][:TOP_N]
    print(f"\nPicks: {[p['symbol'] for p in picks] if picks else 'NONE - all signals down, going to cash'}")

    target = {p["symbol"] for p in picks}
    price_map = {r["symbol"]: r["price"] for r in rows}
    actions = []

    if dry_run:
        actions.append(f"DRY-RUN: would hold top {TOP_N} of {len(ranked)} scanned")
        print(actions[0])
    else:
        # -1) Take-profit: lock in gains at +TAKE_PROFIT_PCT from entry
        for sym, qty in list(held.items()):
            price = price_map.get(sym)
            entry = entries.get(sym)
            if price is None or entry is None:
                continue
            if price >= entry * (1 + TAKE_PROFIT_PCT):
                try:
                    o = alpaca("POST", "/v2/orders", alpaca_key, alpaca_secret,
                               {"symbol": sym, "qty": str(qty), "side": "sell",
                                "type": "market", "time_in_force": "day"})
                    actions.append(f"TAKE-PROFIT SELL {qty} {sym} (entry ${entry:,.2f} "
                                   f"-> ${price:,.2f}, +{(price / entry - 1) * 100:.1f}%) "
                                   f"-> {o.get('status')}")
                    ret = record_outcome(sym, entry, price, "take-profit")
                    actions.append(f"LEARN: {sym} closed at {ret:+.2%} (take-profit)")
                    peaks.pop(sym, None)
                    del held[sym]
                except Exception as e:
                    actions.append(f"TAKE-PROFIT SELL {qty} {sym} FAILED: {e}")

        # 0) Trailing stop-loss: sell if the position's TOTAL VALUE dropped
        # STOP_LOSS_DROP dollars or more from its peak value while owned -
        # but never sell below the entry (buy) price.
        peaks = load_peak_state()
        for sym, qty in list(held.items()):
            price = price_map.get(sym)
            entry = entries.get(sym)
            if price is None or entry is None:
                continue
            value = qty * price
            peaks[sym] = max(peaks.get(sym, value), value)
            drop = peaks[sym] - value
            if drop >= STOP_LOSS_DROP and price >= entry:
                try:
                    o = alpaca("POST", "/v2/orders", alpaca_key, alpaca_secret,
                               {"symbol": sym, "qty": str(qty), "side": "sell",
                                "type": "market", "time_in_force": "day"})
                    actions.append(f"STOP-LOSS SELL {qty} {sym} (value ${peaks[sym]:,.2f} "
                                   f"-> ${value:,.2f}, -${drop:,.2f}, "
                                   f"entry ${entry:,.2f}) -> {o.get('status')}")
                    ret = record_outcome(sym, entry, price, "stop-loss")
                    actions.append(f"LEARN: {sym} closed at {ret:+.2%} (stop-loss)")
                    del peaks[sym]
                    del held[sym]
                except Exception as e:
                    actions.append(f"STOP-LOSS SELL {qty} {sym} FAILED: {e}")
            elif drop >= STOP_LOSS_DROP:
                actions.append(f"STOP-LOSS HOLD {sym} (value down ${drop:,.2f} "
                               f"from peak but below entry ${entry:,.2f} - not selling at a loss)")

        # 1) Sell positions that are no longer picks
        for sym, qty in held.items():
            if sym not in target:
                try:
                    o = alpaca("POST", "/v2/orders", alpaca_key, alpaca_secret,
                               {"symbol": sym, "qty": str(qty), "side": "sell",
                                "type": "market", "time_in_force": "day"})
                    actions.append(f"SELL {qty} {sym} -> {o.get('status')}")
                    peaks.pop(sym, None)
                    entry = entries.get(sym)
                    exit_px = price_map.get(sym)
                    if entry and exit_px:
                        ret = record_outcome(sym, entry, exit_px, "signal")
                        actions.append(f"LEARN: {sym} closed at {ret:+.2%} (signal)")
                except Exception as e:
                    actions.append(f"SELL {qty} {sym} FAILED: {e}")

        # 2) Buy new picks, equal weight across TOP_N slots
        # Trading plan limits: never deploy more than MAX_PORTFOLIO_VALUE,
        # always keep MIN_CASH_PCT in cash, cap each order at MAX_TRADE_VALUE.
        deployable = min(equity, MAX_PORTFOLIO_VALUE)
        if equity > MAX_PORTFOLIO_VALUE:
            actions.append(f"LIMIT: equity ${equity:,.2f} exceeds cap "
                           f"${MAX_PORTFOLIO_VALUE:,.2f} - deploying "
                           f"${deployable:,.2f} only")
        # Score-weighted sizing: stronger picks get a bigger share of the budget
        total_score = sum(max(p["score"], 0.01) for p in picks) or 1.0
        pool = min(deployable * (1 - MIN_CASH_PCT), MAX_TRADE_VALUE * len(picks))
        for pick in picks:
            sym = pick["symbol"]
            if sym in held:
                continue
            budget = min(pool * max(pick["score"], 0.01) / total_score, MAX_TRADE_VALUE)
            qty = int(budget // pick["price"])
            if qty < 1:
                actions.append(f"BUY {sym} skipped (price ${pick['price']:,.2f} > slot budget)")
                continue
            try:
                o = alpaca("POST", "/v2/orders", alpaca_key, alpaca_secret,
                           {"symbol": sym, "qty": str(qty), "side": "buy",
                            "type": "market", "time_in_force": "day"})
                actions.append(f"BUY {qty} {sym} (~${qty * pick['price']:,.0f}) -> {o.get('status')}")
                peaks[sym] = qty * pick["price"]  # start trailing stop tracking (position value)
            except Exception as e:
                actions.append(f"BUY {qty} {sym} FAILED: {e}")

        save_peak_state(peaks)

        print("\nrebalance actions:")
        for a in actions:
            print(f"  {a}")
        if not actions:
            print("  none - portfolio already matches picks")

    log_track_record(universe_desc, len(rows), picks, actions, held, equity, dry_run)

    if nvidia_key:
        context = (f"Universe: {universe_desc}, {len(rows)} tickers scanned. Signals: "
                   + ", ".join(f"{r['symbol']} P(up)={r['p_up']:.2f} mem={r['memory_adj']:+.2f}"
                               for r in ranked)
                   + f". Picks: {[p['symbol'] for p in picks]}. Actions: {actions}.")
        print(f"\n--- AI portfolio commentary ({LLM_MODEL} via NVIDIA API) ---")
        try:
            print(llm_commentary(context, nvidia_key))
        except Exception as e:
            print(f"LLM commentary failed: {e}")


if __name__ == "__main__":
    main()

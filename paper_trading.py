"""
paper_trading.py - Multi-ticker paper trading bot (Alpaca paper API) with a
ROTATING ticker universe, a persistent track record, and a full risk layer.

Each run picks a window of WINDOW_SIZE tickers from TICKER_POOL. The window
rotates every ROTATE_EVERY_DAYS days. Currently-held positions are ALWAYS
scanned too, so a holding is only sold when its own signal turns down - never
just because it rotated out of view.

Market data: ALPACA stock candles first (same source as the charts and the
orders), Polygon as fallback. Models read candlestick features (body, wicks,
range, gap, streak) plus classic technicals.

Per run: load the persisted, continuously-retrained model per ticker when one
is fresh and drift-free (see continuous_train.py); otherwise train inline and
validate with WALK-FORWARD cross-validation (tickers that fail are skipped),
rank by P(up), hold the TOP_N with P(up) > 0.5. Position size is RISK-BASED:
each new position risks at most RISK_PER_TRADE of equity, sized from its ATR
stop distance (capped by an equal-weight slot budget and the daily trade
limit). New buys are placed as Alpaca BRACKET orders with an ATR stop-loss
and a reward:risk take-profit. Positions older than MAX_HOLD_DAYS are closed
(time exit). A TRAILING-DIP PROFIT LOCK sells any holding that is >=
DIP_EXIT_PCT off today's high while still above its entry price. A KILL
SWITCH halts all new trading if equity is down more than KILL_SWITCH_DD vs
the previous day's equity. Orders are only placed while the market is OPEN
(Alpaca clock check) - off-hours orders are rejected (403).

Optional strategy conditions (regime filter, weekend rule, exposure cap,
confidence sizing, cooldown) are toggled by strategy_config.json, written by
strategy_lab.py after it measures each condition on out-of-sample data -
the model chooses which conditions are active, and its decisions are archived
in LEARNINGS.md.

Keys from environment - NEVER hardcode secrets:
  POLYGON_API_KEY     (Polygon.io - fallback market data)
  ALPACA_API_KEY      (Alpaca paper key id)      -> dry-run if unset
  ALPACA_SECRET_KEY   (Alpaca paper secret)      -> dry-run if unset
  API_KEY             (NVIDIA - optional AI commentary)
Config env vars:
  TICKER_POOL           (csv, default 168 tickers)
  WINDOW_SIZE           (default 30)
  ROTATE_EVERY_DAYS     (default 7)
  TOP_N                 (default 3)
  WATCHLIST             (optional csv - disables rotation)
  TRACK_RECORD_FILE     (default track_record.csv)
  DAILY_TRADE_LIMIT_USD (default 50000)  - max new $ bought per run/day
  RISK_PER_TRADE        (default 0.01)   - fraction of equity risked per position
  STOP_ATR_MULT         (default 2.0)    - stop distance in ATR-14 units
  TP_RR                 (default 2.0)    - take-profit reward:risk ratio
  MAX_HOLD_DAYS         (default 20)     - time exit: close positions older than this
  KILL_SWITCH_DD        (default 0.03)   - halt trading if daily drawdown exceeds this
  MIN_WF_ACCURACY       (default 0.54)   - min walk-forward accuracy to trade a ticker
  P_UP_MIN              (default 0.55)   - min P(up) for a new buy
  MIN_HOLD_DAYS         (default 3)      - don't rotation-sell positions younger than this
  DIP_EXIT_PCT          (default 0.0125) - sell if this far off day high and above entry
  STRATEGY_CONFIG_FILE  (default strategy_config.json) - model-chosen conditions
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
from sklearn.metrics import accuracy_score

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
DAILY_TRADE_LIMIT_USD = float(os.environ.get("DAILY_TRADE_LIMIT_USD", "50000"))
FIXED_WATCHLIST = [t.strip().upper() for t in os.environ.get(
    "WATCHLIST", "").split(",") if t.strip()]
TRACK_RECORD_FILE = os.environ.get("TRACK_RECORD_FILE", "track_record.csv")

# --- Risk layer config ---
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.01"))
STOP_ATR_MULT = float(os.environ.get("STOP_ATR_MULT", "2.0"))
TP_RR = float(os.environ.get("TP_RR", "2.0"))
MAX_HOLD_DAYS = int(os.environ.get("MAX_HOLD_DAYS", "20"))
KILL_SWITCH_DD = float(os.environ.get("KILL_SWITCH_DD", "0.03"))
MIN_WF_ACCURACY = float(os.environ.get("MIN_WF_ACCURACY", "0.54"))
DIP_EXIT_PCT = float(os.environ.get("DIP_EXIT_PCT", "0.0125"))  # sell if 1.25% off day high and above entry
P_UP_MIN = float(os.environ.get("P_UP_MIN", "0.55"))  # min P(up) to buy - coin-flip signals cost money
MIN_HOLD_DAYS = int(os.environ.get("MIN_HOLD_DAYS", "3"))  # don't churn positions younger than this
WF_FOLDS = 3  # walk-forward validation folds

STRATEGY_CONFIG_FILE = os.environ.get("STRATEGY_CONFIG_FILE", "strategy_config.json")


def load_strategy_config():
    """Model-chosen condition toggles written by strategy_lab.py.
    Missing file -> conservative defaults (all optional conditions off)."""
    defaults = {"conditions": {}, "params": {}}
    if os.path.exists(STRATEGY_CONFIG_FILE):
        try:
            with open(STRATEGY_CONFIG_FILE) as f:
                cfg = json.load(f)
            print(f"Strategy config loaded ({STRATEGY_CONFIG_FILE}, "
                  f"generated {cfg.get('generated_at', '?')}): {cfg.get('conditions')}")
            return cfg
        except Exception as e:
            print(f"strategy_config.json unreadable ({e}); using defaults")
    return defaults


STRATEGY_CFG = load_strategy_config()
COND = STRATEGY_CFG.get("conditions", {})
PARAMS = STRATEGY_CFG.get("params", {})

POLYGON_MIN_INTERVAL = 12.2  # free tier: 5 requests/min
_last_polygon_call = [0.0]


def rotating_window(pool, window_size, period, step=10):
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
    """Load daily candles: Alpaca stock bars first (same source as the charts
    and the orders), Polygon as fallback (paced for the free tier)."""
    from stock_model import get_daily
    try:
        return get_daily(symbol, polygon_key=None)  # Alpaca, no pacing needed
    except Exception as e:
        print(f"  {symbol}: Alpaca candles unavailable ({e}); using Polygon")
    wait = POLYGON_MIN_INTERVAL - (time.time() - _last_polygon_call[0])
    if wait > 0:
        time.sleep(wait)
    df = get_daily(symbol, polygon_key=key)
    _last_polygon_call[0] = time.time()
    return df


def compute_atr(df, period=14):
    """Average True Range - used for stop-loss distance."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def walk_forward_accuracy(data, folds=WF_FOLDS):
    """Walk-forward (expanding-window) validation on the time series.
    Returns mean out-of-fold accuracy, or 0.0 if not enough data.
    Never trains on the future - no look-ahead bias."""
    train = data.dropna(subset=["target"])
    n = len(train)
    if n < 120:  # need a decent sample to validate
        return 0.0
    fold_size = n // (folds + 1)
    accs = []
    for k in range(1, folds + 1):
        cut = fold_size * k
        X_tr, y_tr = train[FEATURES].iloc[:cut], train["target"].iloc[:cut]
        X_te, y_te = train[FEATURES].iloc[cut:cut + fold_size], train["target"].iloc[cut:cut + fold_size]
        if len(X_te) == 0 or y_tr.nunique() < 2:
            continue
        m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        m.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_te, m.predict(X_te)))
    return sum(accs) / len(accs) if accs else 0.0


def scan(symbols, poly_key):
    rows = []
    try:
        from continuous_train import load_model
    except ImportError:
        load_model = None
    for sym in symbols:
        try:
            df = polygon_daily(sym, poly_key)
            data = engineer_features(df)
            cached = load_model(sym) if load_model else None
            if cached:
                # Continuous-training layer: fresh, drift-free persisted model
                model, wf_acc = cached
                print(f"  {sym:6s} using persisted model (wf_acc={wf_acc:.2f})")
            else:
                wf_acc = walk_forward_accuracy(data)
                train = data.dropna(subset=["target"])
                X, y = train[FEATURES], train["target"]
                model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
                model.fit(X, y)
            latest = data[FEATURES].iloc[-1:]
            proba = model.predict_proba(latest)[0]
            p_up = float(proba[1]) if len(proba) > 1 else 0.0
            price = float(df["close"].iloc[-1])
            atr = compute_atr(df)
            rows.append({"symbol": sym, "p_up": p_up, "price": price,
                         "atr": atr, "wf_acc": wf_acc, "days": len(data)})
            print(f"  {sym:6s} P(up)={p_up:.2f} wf_acc={wf_acc:.2f} "
                  f"close=${price:,.2f} atr=${atr:.2f} ({len(data)} days)")
        except Exception as e:
            print(f"  {sym}: skipped - {e}")
    return rows


def risk_based_qty(equity, price, atr, slot_budget):
    """Size a position so the ATR stop risks at most RISK_PER_TRADE of equity,
    capped by the equal-weight slot budget."""
    stop_dist = STOP_ATR_MULT * atr
    if stop_dist <= 0:
        return 0
    risk_qty = int((equity * RISK_PER_TRADE) / stop_dist)
    budget_qty = int(slot_budget // price)
    return min(risk_qty, budget_qty)


def alpaca(method, path, key, secret, body=None, retries=3):
    """Alpaca API call with retry/backoff on transient 403/5xx failures.
    A failed SELL leaves a phantom position, so retry before giving up."""
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(
            ALPACA_BASE + path,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                     "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (403, 429) or e.code >= 500:
                if attempt < retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"  {method} {path} -> HTTP {e.code}, retrying in {wait}s "
                          f"(attempt {attempt + 1}/{retries})")
                    time.sleep(wait)
                    continue
            raise
    raise last_err



def sell_with_pnl(sym, qty, key, secret, entry_price, label="SELL"):
    """Market-sell `qty` of `sym`, wait briefly for the fill, and return an
    action string that includes the fill price and realized P&L vs entry.
    Makes 'sold at a loss?' answerable directly from the track record."""
    o = alpaca("POST", "/v2/orders", key, secret,
               {"symbol": sym, "qty": str(qty), "side": "sell",
                "type": "market", "time_in_force": "day"})
    fill_px = None
    oid = o.get("id")
    if oid:
        for _ in range(6):  # poll up to ~30s for the fill
            time.sleep(5)
            try:
                st = alpaca("GET", f"/v2/orders/{oid}", key, secret)
            except Exception:
                break
            if st.get("status") == "filled":
                fill_px = float(st.get("filled_avg_price") or 0) or None
                break
            if st.get("status") in ("canceled", "expired", "rejected"):
                return f"{label} {qty} {sym} -> {st.get('status')} (no fill)"
    if fill_px is not None and entry_price:
        pnl = (fill_px - entry_price) * qty
        pct = (fill_px / entry_price - 1) if entry_price else 0.0
        return (f"{label} {qty} {sym} @ ${fill_px:,.2f} (entry ${entry_price:,.2f}, "
                f"P&L {'+' if pnl >= 0 else ''}${pnl:,.2f} / {pct:+.2%}) -> filled")
    if fill_px is not None:
        return f"{label} {qty} {sym} @ ${fill_px:,.2f} (entry unknown) -> filled"
    return f"{label} {qty} {sym} -> {o.get('status')} (fill price pending)"

def regime_ok(poly_key):
    """Regime filter (if enabled): only buy when SPY trades above its SMA-50."""
    if not COND.get("regime_filter"):
        return True
    try:
        from stock_model import get_daily
        spy = get_daily("SPY", polygon_key=poly_key, days=120)
        ok = float(spy["close"].iloc[-1]) > float(spy["close"].rolling(50).mean().iloc[-1])
        print(f"Regime filter: SPY {'>' if ok else '<='} SMA-50 -> buys {'allowed' if ok else 'BLOCKED'}")
        return ok
    except Exception as e:
        print(f"  regime check failed ({e}); allowing buys")
        return True


def weekend_ok():
    """Weekend rule (if enabled): no new buys on Fridays."""
    if not COND.get("weekend_rule"):
        return True
    ok = datetime.utcnow().weekday() != 4
    if not ok:
        print("Weekend rule: Friday -> new buys BLOCKED")
    return ok


def cooldown_blocked(symbol):
    """Cooldown (if enabled): skip re-buying a symbol sold within the last
    cooldown_days according to the track record."""
    if not COND.get("cooldown"):
        return False
    days = int(PARAMS.get("cooldown_days", 3))
    try:
        if not os.path.exists(TRACK_RECORD_FILE):
            return False
        with open(TRACK_RECORD_FILE) as f:
            rows = list(csv.DictReader(f))[-30:]
        for r in reversed(rows):
            if symbol in (r.get("actions") or "") and "SELL" in r["actions"]:
                sold_at = datetime.strptime(r["date_utc"], "%Y-%m-%d %H:%M")
                if (datetime.utcnow() - sold_at).days < days:
                    print(f"  {symbol}: cooldown active (sold {r['date_utc']}) - buy blocked")
                    return True
                return False
    except Exception:
        pass
    return False


def dip_exit_triggered(symbol, entry_price, poly_key, threshold=DIP_EXIT_PCT):
    """Trailing-dip profit lock: True if today's price is `threshold`+ below
    today's high (peak) AND still above the entry price. Uses Alpaca candles."""
    try:
        from stock_model import get_daily
        df = get_daily(symbol, polygon_key=poly_key, days=5)
        day_high = float(df["high"].iloc[-1])
        price = float(df["close"].iloc[-1])
        dip = (day_high - price) / day_high if day_high else 0.0
        return dip >= threshold and price > entry_price, dip, price, day_high
    except Exception as e:
        print(f"  dip check {symbol} failed: {e}")
        return False, 0.0, 0.0, 0.0


def market_is_open(key, secret):
    """Check Alpaca's market clock - returns (is_open, next_open, next_close)."""
    try:
        c = alpaca("GET", "/v2/clock", key, secret)
        return bool(c.get("is_open")), c.get("next_open"), c.get("next_close")
    except Exception as e:
        print(f"  market clock check failed: {e}")
        return False, None, None


def holding_days(symbol, key, secret):
    """Days since the most recent filled BUY for `symbol` (for time exits).
    Returns None if unknown."""
    try:
        orders = alpaca("GET", "/v2/orders?status=closed&limit=200&direction=desc",
                        key, secret)
        for o in orders:
            if o.get("symbol") == symbol and o.get("side") == "buy" and o.get("filled_at"):
                filled = datetime.fromisoformat(o["filled_at"].replace("Z", "+00:00"))
                return (datetime.now(filled.tzinfo) - filled).days
    except Exception as e:
        print(f"  holding_days({symbol}) lookup failed: {e}")
    return None


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
    print(f"Risk: {RISK_PER_TRADE:.1%}/trade, stop={STOP_ATR_MULT}xATR, TP={TP_RR}R, "
          f"max hold {MAX_HOLD_DAYS}d, kill switch -{KILL_SWITCH_DD:.0%}/day, "
          f"min walk-forward acc {MIN_WF_ACCURACY:.2f}, dip exit {DIP_EXIT_PCT:.2%}")

    held = {}
    entry_px = {}
    equity = 0.0
    halted = False
    if not dry_run:
        acct = alpaca("GET", "/v2/account", alpaca_key, alpaca_secret)
        equity = float(acct["equity"])
        last_equity = float(acct.get("last_equity") or equity)
        positions = alpaca("GET", "/v2/positions", alpaca_key, alpaca_secret)
        held = {p["symbol"]: int(float(p["qty"])) for p in positions}
        entry_px = {p["symbol"]: float(p["avg_entry_price"]) for p in positions}
        extras = [s for s in held if s not in universe]
        if extras:
            universe = universe + extras
            print(f"Also scanning held positions outside this window: {extras}")
        print(f"Paper account: equity=${equity:,.2f} | holding: {held or 'nothing'}\n")

        # --- KILL SWITCH: halt all trading if daily drawdown is too large ---
        day_dd = equity / last_equity - 1 if last_equity else 0.0
        if day_dd <= -KILL_SWITCH_DD:
            halted = True
            print(f"KILL SWITCH TRIGGERED: equity {day_dd:+.2%} vs yesterday "
                  f"(limit -{KILL_SWITCH_DD:.0%}). No orders will be placed.")

    rows = scan(universe, poly_key)
    if not rows:
        raise SystemExit("No ticker data available - check POLYGON_API_KEY.")

    # Walk-forward gate: only trade tickers whose model validated out-of-sample
    valid = [r for r in rows if r["wf_acc"] >= MIN_WF_ACCURACY]
    rejected = [r for r in rows if r["wf_acc"] < MIN_WF_ACCURACY]
    if rejected:
        print(f"Walk-forward rejected: "
              f"{[(r['symbol'], round(r['wf_acc'], 2)) for r in rejected]}")

    ranked = sorted(valid, key=lambda r: r["p_up"], reverse=True)
    picks = [r for r in ranked if r["p_up"] >= P_UP_MIN][:TOP_N]
    print(f"\nPicks: {[p['symbol'] for p in picks] if picks else 'NONE - all signals down, going to cash'}")

    target = {p["symbol"] for p in picks}
    actions = []

    if dry_run:
        actions.append(f"DRY-RUN: would hold top {TOP_N} of {len(ranked)} validated tickers")
        print(actions[0])
    elif halted:
        actions.append("HALTED: kill switch active, no orders placed")
    else:
        open_now, nxt_open, nxt_close = market_is_open(alpaca_key, alpaca_secret)
        if not open_now:
            # Orders placed while the market is closed are rejected (HTTP 403).
            # CI targets 19:30 UTC (3:30pm ET) so this should rarely trigger.
            actions.append(f"SKIPPED: market closed (next open {nxt_open})")
            print(actions[-1])
    if actions and actions[-1].startswith(("HALTED", "SKIPPED")):
        pass  # trading blocked this run; skip to logging
    elif not dry_run:
        # 0) TRAILING-DIP PROFIT LOCK: sell if >=DIP_EXIT_PCT off today's high
        #    and still above entry - locks gains before they round-trip
        for sym, qty in held.items():
            triggered, dip, price, peak = dip_exit_triggered(
                sym, entry_px.get(sym, float("inf")), poly_key)
            if triggered:
                try:
                    actions.append(sell_with_pnl(sym, qty, alpaca_key, alpaca_secret,
                                                 entry_px.get(sym), label="DIP-EXIT SELL")
                                   + f" (peak ${peak:,.2f}, -{dip:.1%} off high)")
                except Exception as e:
                    actions.append(f"DIP-EXIT SELL {qty} {sym} FAILED: {e}")

        # 1) TIME EXITS: close positions held longer than MAX_HOLD_DAYS
        for sym, qty in held.items():
            days = holding_days(sym, alpaca_key, alpaca_secret)
            if days is not None and days >= MAX_HOLD_DAYS:
                try:
                    actions.append(sell_with_pnl(sym, qty, alpaca_key, alpaca_secret,
                                                 entry_px.get(sym), label="TIME-EXIT SELL")
                                   + f" (held {days}d)")
                except Exception as e:
                    actions.append(f"TIME-EXIT SELL {qty} {sym} FAILED: {e}")

        # 2) Sell positions that are no longer picks (and not just exited)
        exited = set()
        for a in actions:
            if "FAILED" in a or "no fill" in a:
                continue
            parts = a.split()
            if a.startswith("SELL"):
                exited.add(parts[2])
            elif a.startswith(("TIME-EXIT SELL", "DIP-EXIT SELL")):
                exited.add(parts[3])
        for sym, qty in held.items():
            if sym not in target and sym not in exited:
                days_held = holding_days(sym, alpaca_key, alpaca_secret)
                if days_held is not None and days_held < MIN_HOLD_DAYS:
                    actions.append(f"HOLD {qty} {sym} (held {days_held}d < min {MIN_HOLD_DAYS}d - avoiding churn)")
                    continue
                try:
                    actions.append(sell_with_pnl(sym, qty, alpaca_key, alpaca_secret,
                                                 entry_px.get(sym)))
                except Exception as e:
                    actions.append(f"SELL {qty} {sym} FAILED: {e}")

        # 3) Buy new picks: risk-based size, BRACKET order with ATR stop + TP
        failed_sells = [a for a in actions if "SELL" in a and "FAILED" in a]
        buys_allowed = regime_ok(poly_key) and weekend_ok()
        if failed_sells:
            buys_allowed = False
            actions.append(f"BUYS BLOCKED: {len(failed_sells)} SELL(s) failed - "
                           f"positions may still be open; refusing to add risk")
        if not buys_allowed:
            actions.append("BUYS BLOCKED by strategy config (regime/weekend)")
        invested = sum(int(q) * float(entry_px.get(s, 0)) for s, q in held.items())
        max_deploy = float(PARAMS.get("max_deploy", 0.95)) if COND.get("exposure_cap") else 0.95
        daily_budget = min(equity * max_deploy - invested, equity * 0.95, DAILY_TRADE_LIMIT_USD)
        daily_budget = max(daily_budget, 0.0)
        slot_budget = daily_budget / TOP_N
        print(f"Daily trade limit: ${DAILY_TRADE_LIMIT_USD:,.0f} -> slot budget ${slot_budget:,.2f}")
        for pick in picks if buys_allowed else []:
            sym = pick["symbol"]
            if sym in held:
                continue
            if cooldown_blocked(sym):
                actions.append(f"BUY {sym} skipped (cooldown)")
                continue
            qty = risk_based_quy(equity, pick["price"], pick["atr"], slot_budget)
            if COND.get("confidence_sizing"):
                lo = float(PARAMS.get("confidence_size_min", 0.5))
                hi = float(PARAMS.get("confidence_size_max", 1.5))
                mult = min(max((pick["p_up"] - 0.5) * 2, lo), hi)
                qty = int(qty * mult)
            if qty < 1:
                actions.append(f"BUY {sym} skipped (risk/size constraints)")
                continue
            stop = round(pick["price"] - STOP_ATR_MULT * pick["atr"], 2)
            target_px = round(pick["price"] + TP_RR * STOP_ATR_MULT * pick["atr"], 2)
            order = {
                "symbol": sym, "qty": str(qty), "side": "buy",
                "type": "market", "time_in_force": "day",
                "order_class": "bracket",
                "take_profit": {"limit_price": f"{target_px:.2f}"},
                "stop_loss": {"stop_price": f"{stop:.2f}"},
            }
            try:
                o = alpaca("POST", "/v2/orders", alpaca_key, alpaca_secret, order)
                actions.append(
                    f"BUY {qty} {sym} (~${qty * pick['price']:,.0f}, "
                    f"stop=${stop}, tp=${target_px}, risks ${qty * STOP_ATR_MULT * pick['atr']:,.0f} "
                    f"= {qty * STOP_ATR_MULT * pick['atr'] / equity:.2%} of equity) "
                    f"-> {o.get('status')}")
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

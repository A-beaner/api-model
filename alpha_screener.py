"""
Alpha Vantage momentum screener (days-to-weeks horizon)

Funnel design, so scarce API calls are spent only where they matter:

  Stage 0  MARKET_STATUS / regime    1 call    Is the market tape supportive?
  Stage 1  TOP_GAINERS_LOSERS        1 call    Universe of candidates
  Stage 2  Hard filters (free)       0 calls   Drop warrants, penny stocks, illiquid
  Stage 3  Daily OHLCV + local calc  1/ticker  Compute RSI, MACD, SMA, ATR, RVOL locally
  Stage 4  Score + rank              0 calls   Composite score with risk-based sizing
  Stage 5  NEWS_SENTIMENT (top N)    1/ticker  Sanity check the finalists only

Key idea: fetch raw prices once and compute all indicators locally with pandas.
Calling RSI, MACD, SMA, ATR endpoints separately costs 4+ calls per ticker for
data you can derive from a single TIME_SERIES_DAILY call.

Setup:
    pip install requests pandas numpy
    export ALPHAVANTAGE_API_KEY=your_key      # free key: alphavantage.co

Free tier is ~25 calls/day, so keep MAX_CANDIDATES small. Premium keys allow
much larger scans; set CALLS_PER_MINUTE to match your plan.

NOT financial advice. This ranks candidates; it does not predict returns.
Backtest before risking money.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://www.alphavantage.co/query"

# ----------------------------- configuration -----------------------------
API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
CALLS_PER_MINUTE = 5          # free tier ~5/min; raise on premium (e.g. 75, 150)
MAX_CANDIDATES = 8            # how many tickers get the expensive daily-data call
TOP_N_NEWS = 3                # finalists that get a news sentiment check

MIN_PRICE = 5.00              # no penny stocks
MIN_DOLLAR_VOLUME = 5_000_000 # price * volume, filters illiquid names
MAX_ONE_DAY_MOVE_PCT = 60.0   # >60% single-day pops are usually news/pump spikes
ACCOUNT_RISK_PCT = 0.01       # risk 1% of account per trade
ATR_STOP_MULT = 2.0           # stop = entry - 2*ATR


# ------------------------------ API client -------------------------------
class AlphaVantage:
    """Thin client with rate limiting and error handling."""

    def __init__(self, api_key: str, calls_per_minute: int):
        if not api_key:
            sys.exit("Set ALPHAVANTAGE_API_KEY first.")
        self.key = api_key
        self.min_interval = 60.0 / calls_per_minute
        self._last_call = 0.0
        self.calls_made = 0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def get(self, **params) -> dict:
        self._throttle()
        params["apikey"] = self.key
        for attempt in range(3):
            try:
                r = requests.get(BASE_URL, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
            except (requests.RequestException, ValueError) as e:
                print(f"  ! network/parse error ({e}); retry {attempt + 1}/3")
                time.sleep(2 ** attempt)
                continue
            self.calls_made += 1
            # Alpha Vantage reports limits/errors inside a 200 response.
            note = data.get("Note") or data.get("Information")
            if note:
                print(f"  ! API notice: {note[:120]}...")
                return {}
            if "Error Message" in data:
                print(f"  ! API error: {data['Error Message'][:120]}")
                return {}
            return data
        return {}

    # --- endpoint wrappers ---
    def top_movers(self) -> dict:
        return self.get(function="TOP_GAINERS_LOSERS")

    def daily(self, symbol: str) -> pd.DataFrame:
        data = self.get(function="TIME_SERIES_DAILY", symbol=symbol, outputsize="compact")
        ts = data.get("Time Series (Daily)")
        if not ts:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(ts, orient="index").astype(float)
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def news(self, symbol: str, limit: int = 20) -> dict:
        return self.get(function="NEWS_SENTIMENT", tickers=symbol, limit=limit)


# ---------------------------- local indicators ---------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ------------------------------- data model ------------------------------
@dataclass
class Candidate:
    ticker: str
    price: float
    day_change_pct: float
    volume: float
    metrics: dict = field(default_factory=dict)
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    sentiment: float | None = None


# ------------------------------ stage 2: filter --------------------------
_DERIVATIVE_SUFFIX = re.compile(r"(W|WS|R|RT|U)$")   # warrants, rights, units


def parse_pct(s: str) -> float:
    return float(str(s).replace("%", ""))


def hard_filter(rows: list[dict]) -> list[Candidate]:
    """Zero-cost filtering of the movers list using only data already fetched."""
    out = []
    for r in rows:
        t = r["ticker"]
        price, vol = float(r["price"]), float(r["volume"])
        chg = parse_pct(r["change_percentage"])

        if not t.isalpha():                       # BCAT^#, COPL+ etc.
            continue
        if len(t) >= 5 and _DERIVATIVE_SUFFIX.search(t):
            continue                              # warrants / rights / units
        if price < MIN_PRICE:
            continue
        if price * vol < MIN_DOLLAR_VOLUME:
            continue
        if chg > MAX_ONE_DAY_MOVE_PCT:
            continue                              # likely a one-day spike; poor entry
        out.append(Candidate(t, price, chg, vol))
    return out


# --------------------------- stage 3/4: analyze + score ------------------
def analyze(c: Candidate, df: pd.DataFrame) -> bool:
    """Compute indicators locally and store on the candidate. False if unusable."""
    if len(df) < 60:
        return False

    close = df["close"]
    r = rsi(close)
    macd_line, macd_sig, hist = macd(close)
    a = atr(df)
    sma20, sma50 = close.rolling(20).mean(), close.rolling(50).mean()
    avg_vol20 = df["volume"].rolling(20).mean()

    last = df.index[-1]
    m = {
        "close": close.iloc[-1],
        "rsi": r.iloc[-1],
        "macd_hist": hist.iloc[-1],
        "macd_hist_prev": hist.iloc[-2],
        "sma20": sma20.iloc[-1],
        "sma50": sma50.iloc[-1],
        "atr": a.iloc[-1],
        "rvol": df["volume"].iloc[-1] / avg_vol20.iloc[-1],
        "ret_5d": close.iloc[-1] / close.iloc[-6] - 1,
        "ret_20d": close.iloc[-1] / close.iloc[-21] - 1,
        "high_20d": df["high"].iloc[-21:-1].max(),
        "data_date": last,
    }
    if any(pd.isna(v) for k, v in m.items() if k != "data_date"):
        return False
    c.metrics = m
    return True


def score(c: Candidate) -> None:
    """
    Composite 0-100 score. Rewards trend + confirmed momentum + volume,
    penalizes overextension. Weights are starting points; tune via backtest.
    """
    m = c.metrics
    s = 0.0

    # Trend structure (30): price above rising averages
    if m["close"] > m["sma20"]:
        s += 10; c.reasons.append("above 20d SMA")
    if m["sma20"] > m["sma50"]:
        s += 10; c.reasons.append("20d SMA above 50d (uptrend)")
    if m["close"] > m["sma50"]:
        s += 10

    # Momentum quality (30): MACD positive and improving, RSI in the sweet spot
    if m["macd_hist"] > 0:
        s += 10; c.reasons.append("MACD histogram positive")
    if m["macd_hist"] > m["macd_hist_prev"]:
        s += 10; c.reasons.append("MACD accelerating")
    if 55 <= m["rsi"] <= 72:
        s += 10; c.reasons.append(f"RSI {m['rsi']:.0f} (strong, not exhausted)")

    # Participation (20): unusual volume confirms the move
    if m["rvol"] >= 1.5:
        s += 12; c.reasons.append(f"volume {m['rvol']:.1f}x average")
    elif m["rvol"] >= 1.0:
        s += 6
    if m["close"] > m["high_20d"]:
        s += 8; c.reasons.append("breaking 20d high")

    # Relative strength (20): sustained, not just one-day, gains
    if m["ret_20d"] > 0.05:
        s += 10
    if m["ret_5d"] > 0.02:
        s += 10

    # Penalties: chasing extended names is how momentum trades lose money
    if m["rsi"] > 80:
        s -= 20; c.flags.append("RSI > 80: overbought")
    ext = (m["close"] - m["sma20"]) / m["sma20"]
    if ext > 0.15:
        s -= 15; c.flags.append(f"{ext:.0%} above 20d SMA: extended")
    if m["atr"] / m["close"] > 0.08:
        s -= 10; c.flags.append("very high volatility (ATR > 8% of price)")

    c.score = max(0.0, min(100.0, s))


def position_plan(c: Candidate, account_size: float) -> dict:
    """ATR-based stop and volatility-adjusted position size."""
    m = c.metrics
    entry = m["close"]
    stop = entry - ATR_STOP_MULT * m["atr"]
    risk_per_share = entry - stop
    shares = int((account_size * ACCOUNT_RISK_PCT) / risk_per_share) if risk_per_share > 0 else 0
    shares = min(shares, int(account_size * 0.25 / entry))   # cap at 25% of account
    return {
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target_2R": round(entry + 2 * risk_per_share, 2),
        "shares": shares,
        "capital": round(shares * entry, 2),
    }


# ------------------------------ stage 5: news ----------------------------
def attach_sentiment(av: AlphaVantage, c: Candidate) -> None:
    data = av.news(c.ticker)
    scores = []
    for item in data.get("feed", []):
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker") == c.ticker:
                rel = float(ts.get("relevance_score", 0))
                scores.append((float(ts["ticker_sentiment_score"]), rel))
    if scores:
        w = sum(r for _, r in scores)
        c.sentiment = sum(s * r for s, r in scores) / w if w else None
        if c.sentiment is not None and c.sentiment < -0.15:
            c.flags.append(f"negative news sentiment ({c.sentiment:+.2f})")


# ---------------------------------- main ---------------------------------
def run(account_size: float = 10_000.0) -> list[Candidate]:
    av = AlphaVantage(API_KEY, CALLS_PER_MINUTE)

    print("Stage 1: fetching market movers...")
    movers = av.top_movers()
    if not movers:
        sys.exit("No mover data returned.")

    # Use gainers AND most-active: active large caps often hold better setups
    pool = movers.get("top_gainers", []) + movers.get("most_actively_traded", [])
    seen, rows = set(), []
    for r in pool:
        if r["ticker"] not in seen:
            seen.add(r["ticker"]); rows.append(r)

    print(f"Stage 2: hard filters on {len(rows)} tickers...")
    cands = hard_filter(rows)
    # Prioritize by dollar volume so limited calls go to the most tradeable names
    cands.sort(key=lambda c: c.price * c.volume, reverse=True)
    cands = cands[:MAX_CANDIDATES]
    print(f"  {len(cands)} candidates survive: {[c.ticker for c in cands]}")

    print("Stage 3/4: fetching daily bars, computing indicators locally...")
    scored = []
    for c in cands:
        df = av.daily(c.ticker)
        if df.empty or not analyze(c, df):
            print(f"  - {c.ticker}: insufficient data, skipped")
            continue
        score(c)
        scored.append(c)
    scored.sort(key=lambda c: c.score, reverse=True)

    print(f"Stage 5: news sentiment for top {TOP_N_NEWS}...")
    for c in scored[:TOP_N_NEWS]:
        attach_sentiment(av, c)

    print(f"\nAPI calls used: {av.calls_made}\n" + "=" * 64)
    for i, c in enumerate(scored, 1):
        plan = position_plan(c, account_size)
        print(f"#{i} {c.ticker:<6} score {c.score:>5.1f}  price ${c.price:.2f}  "
              f"RSI {c.metrics['rsi']:.0f}  RVOL {c.metrics['rvol']:.1f}x")
        print(f"    why:   {'; '.join(c.reasons) or 'n/a'}")
        if c.flags:
            print(f"    flags: {'; '.join(c.flags)}")
        if c.sentiment is not None:
            print(f"    news sentiment: {c.sentiment:+.2f}")
        print(f"    plan:  entry ${plan['entry']}  stop ${plan['stop']}  "
              f"2R target ${plan['target_2R']}  size {plan['shares']} sh "
              f"(${plan['capital']:,.0f})")
    print("=" * 64 + "\nRanking tool only, not advice. Verify catalysts and earnings dates.")
    return scored


if __name__ == "__main__":
    run(account_size=float(sys.argv[1]) if len(sys.argv) > 1 else 10_000.0)

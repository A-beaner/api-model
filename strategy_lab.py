"""
strategy_lab.py - The model's self-improvement lab.

Evaluates every optional trading condition against the pooled cached data
(data/*.csv) with OUT-OF-SAMPLE evidence, decides which conditions actually
help, writes the winning configuration to strategy_config.json (which
paper_trading.py obeys), and archives a human-readable record of everything
it learns in LEARNINGS.md - the gitdoc of the model's accumulated knowledge.

Runs weekly in CI (and on push). Conditions it can measure it measures;
conditions needing data it doesn't have (e.g. earnings calendar) are marked
"undetermined" and left off until measurable.

Env: DATA_DIR (default data), CONFIG_FILE (strategy_config.json),
     LEARNINGS_FILE (LEARNINGS.md), MIN_TRADES_EVAL (default 30).
"""

import json
import os
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from stock_model import engineer_features, FEATURES

DATA_DIR = os.environ.get("DATA_DIR", "data")
CONFIG_FILE = os.environ.get("CONFIG_FILE", "strategy_config.json")
LEARNINGS_FILE = os.environ.get("LEARNINGS_FILE", "LEARNINGS.md")
MIN_TRADES_EVAL = int(os.environ.get("MIN_TRADES_EVAL", "30"))

CONDITIONS = [
    "regime_filter",        # only buy when SPY close > SMA-50
    "weekend_rule",         # no new buys on Fridays
    "exposure_cap",         # cap total invested at MAX_DEPLOY of equity
    "confidence_sizing",    # scale position size with P(up)
    "cooldown",             # no re-buy within N days of selling
    "correlation_guard",    # max 1 holding per sector / high pairwise corr
    "trailing_stop",        # broker-side trailing stop instead of fixed stop
    "breakeven_stop",       # move stop to entry at +1R
    "partial_take_profit",  # sell half at +1R
    "earnings_blackout",    # skip buys near earnings (needs earnings data)
]


def load_panels():
    panels = {}
    for f in sorted(os.listdir(DATA_DIR)) if os.path.isdir(DATA_DIR) else []:
        if f.endswith(".csv"):
            sym = f[:-4]
            try:
                df = pd.read_csv(os.path.join(DATA_DIR, f), index_col=0, parse_dates=True)
                if len(df) >= 150:
                    panels[sym] = df
            except Exception:
                pass
    return panels


def oos_signals(df):
    """Out-of-sample P(up) and forward returns for the last 30% of history."""
    data = engineer_features(df)
    train = data.dropna(subset=["target"])
    split = int(len(train) * 0.7)
    if len(train) - split < 40:
        return None
    m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    m.fit(train[FEATURES].iloc[:split], train["target"].iloc[:split])
    te = train.iloc[split:].copy()
    te["p_up"] = m.predict_proba(te[FEATURES])[:, 1]
    te["fwd_ret"] = te["close"].shift(-1) / te["close"] - 1
    te["sma50"] = te["close"].rolling(50).mean()
    te["weekday"] = te.index.dayofweek
    te["down_day"] = te["return_1d"] < 0
    return te.dropna(subset=["fwd_ret", "sma50"])


def evaluate(panels):
    """Pool out-of-sample signal rows across tickers; measure each condition."""
    rows = []
    for sym, df in panels.items():
        te = oos_signals(df)
        if te is not None:
            rows.append(te[["p_up", "fwd_ret", "sma50", "close", "weekday", "down_day"]])
    if not rows:
        return None
    pool = pd.concat(rows)
    sig = pool[pool["p_up"] > 0.5]
    if len(sig) < MIN_TRADES_EVAL:
        return None

    ev = {"n_signals": int(len(sig)), "base_mean_fwd": float(sig["fwd_ret"].mean())}
    decisions = {}

    # regime_filter: signals above vs below the 50-day average
    above = sig[sig["close"] > sig["sma50"]]["fwd_ret"].mean()
    below = sig[sig["close"] <= sig["sma50"]]["fwd_ret"].mean()
    decisions["regime_filter"] = bool(above > below)
    ev["regime_filter"] = {"fwd_above_sma50": round(float(above), 5),
                           "fwd_below_sma50": round(float(below), 5)}

    # weekend_rule: Friday signals vs rest
    fri = sig[sig["weekday"] == 4]["fwd_ret"].mean()
    nonfri = sig[sig["weekday"] != 4]["fwd_ret"].mean()
    decisions["weekend_rule"] = bool(pd.notna(fri) and fri < nonfri)
    ev["weekend_rule"] = {"fwd_friday": round(float(fri), 5) if pd.notna(fri) else None,
                          "fwd_non_friday": round(float(nonfri), 5)}

    # confidence_sizing: does higher P(up) actually mean higher forward return?
    corr = sig["p_up"].corr(sig["fwd_ret"])
    decisions["confidence_sizing"] = bool(pd.notna(corr) and corr > 0.02)
    ev["confidence_sizing"] = {"p_up_vs_fwd_corr": round(float(corr), 4) if pd.notna(corr) else None}

    # cooldown: signals taken the day after a down day (post-loss re-entry proxy)
    after_down = sig[sig["down_day"]]["fwd_ret"].mean()
    after_up = sig[~sig["down_day"]]["fwd_ret"].mean()
    decisions["cooldown"] = bool(after_down < after_up)
    ev["cooldown"] = {"fwd_after_down_day": round(float(after_down), 5),
                      "fwd_after_up_day": round(float(after_up), 5)}

    # exposure_cap: enable if signal edge is thin (protect capital when edge is weak)
    decisions["exposure_cap"] = bool(ev["base_mean_fwd"] < 0.001)
    ev["exposure_cap"] = {"reason": "edge thin (<0.1%/day) - cap deployment"
                          if decisions["exposure_cap"] else "edge sufficient"}

    # correlation_guard: cheap insurance, always on
    decisions["correlation_guard"] = True
    ev["correlation_guard"] = {"reason": "always-on structural guard"}

    # Exit-side conditions: keep trailing stop on; A/B the rest at >100 trades
    for c in ["trailing_stop", "breakeven_stop", "partial_take_profit"]:
        decisions[c] = c == "trailing_stop"
        ev[c] = {"reason": "exit-side condition; set by risk policy, "
                           "will A/B when trade count > 100"}

    # earnings_blackout: no earnings calendar data available
    decisions["earnings_blackout"] = False
    ev["earnings_blackout"] = {"reason": "undetermined - needs earnings calendar feed"}

    return decisions, ev, pool


def write_config(decisions):
    cfg = {
        "generated_at": datetime.utcnow().isoformat(),
        "conditions": decisions,
        "params": {
            "max_deploy": 0.60,        # exposure cap: max 60% invested
            "cooldown_days": 3,
            "confidence_size_min": 0.5,
            "confidence_size_max": 1.5,
            "trail_percent": 1.5,      # trailing stop %
        },
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def append_learnings(decisions, ev, panels):
    date = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"\n## {date}\n",
             f"Evaluated {len(panels)} tickers, {ev['n_signals']} out-of-sample signals.\n"]
    for c in CONDITIONS:
        d = decisions[c]
        detail = json.dumps(ev.get(c, {}))
        lines.append(f"- **{c}**: {'ENABLED' if d else 'OFF'} - `{detail}`")
    lines.append("")
    with open(LEARNINGS_FILE, "a") as f:
        f.write("\n".join(lines))
    print(f"Learnings appended to {LEARNINGS_FILE}")


def main():
    panels = load_panels()
    print(f"Loaded {len(panels)} cached tickers")
    if len(panels) < 5:
        print("Not enough cached data yet - lab will run once the cache grows.")
        return
    result = evaluate(panels)
    if not result:
        print("Not enough out-of-sample signals to decide - keeping defaults.")
        return
    decisions, ev, pool = result
    cfg = write_config(decisions)
    append_learnings(decisions, ev, panels)
    print(json.dumps(cfg["conditions"], indent=2))


if __name__ == "__main__":
    main()

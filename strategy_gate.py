"""
strategy_gate.py - The model chooses its own trading conditions.

For each ticker, this module BACKTESTS candidate strategy conditions on the
cached price history and keeps only the ones that improved out-of-sample
returns. Decisions are saved to strategy_state.json, which paper_trading.py
reads and obeys at trade time. This is the meta-learning layer: the strategy
adapts itself as new data arrives.

Candidate conditions:
  regime_filter     - only go long when SPY is above its 50-day SMA
  dip_exit          - trailing-dip profit lock (sell 1.25% off day high if above entry)
  cooldown          - after an exit, skip re-entry for 5 days
  confidence_sizing - scale position size by signal confidence (P(up) - 0.5)

Env: STATE_FILE (default strategy_state.json), MIN_GAIN_PCT (default 0.5 -
a condition must beat baseline by this much to be enabled).
"""

import json
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from stock_model import engineer_features, FEATURES

STATE_FILE = os.environ.get("STATE_FILE", "strategy_state.json")
MIN_GAIN_PCT = float(os.environ.get("MIN_GAIN_PCT", "0.5"))  # min % improvement to enable
DIP_PCT = 0.0125
COOLDOWN_DAYS = 5

CONDITIONS = ["regime_filter", "dip_exit", "cooldown", "confidence_sizing"]


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def wf_signals(df, folds=3):
    """Walk-forward P(up) series over the tail of the data - the raw strategy
    signal before any conditions are applied."""
    data = engineer_features(df).dropna(subset=["target"])
    n = len(data)
    if n < 120:
        return None
    fold_size = n // (folds + 1)
    sig = pd.Series(index=data.index, dtype=float)
    for k in range(1, folds + 1):
        cut = fold_size * k
        m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        m.fit(data[FEATURES].iloc[:cut], data["target"].iloc[:cut])
        test = data.iloc[cut:cut + fold_size]
        if len(test) == 0:
            continue
        sig.iloc[cut:cut + fold_size] = m.predict_proba(test[FEATURES])[:, 1]
    out = df.join(sig.rename("p_up")).dropna(subset=["p_up"])
    return out


def simulate(df, spy_above_sma=None, conditions=()):
    """Simple daily long-only sim: long when p_up > 0.5 (and regime ok),
    sized by confidence if enabled; dip exit and cooldown applied day by day.
    Returns total return %."""
    p_up = df["p_up"].values
    close = df["close"].values
    high = df["high"].values
    n = len(df)
    equity = 1.0
    in_pos = False
    entry = 0.0
    peak = 0.0
    size = 1.0
    cooldown_until = -1
    for i in range(1, n):
        ret = close[i] / close[i - 1] - 1
        if in_pos:
            equity *= 1 + ret * size
            peak = max(peak, high[i])
            dip = (peak - close[i]) / peak if peak else 0.0
            if p_up[i] <= 0.5:
                in_pos = False
                if "cooldown" in conditions and ret < 0:
                    cooldown_until = i + COOLDOWN_DAYS
            elif "dip_exit" in conditions and dip >= DIP_PCT and close[i] > entry:
                in_pos = False
                if "cooldown" in conditions:
                    cooldown_until = i + COOLDOWN_DAYS
        else:
            if i <= cooldown_until:
                continue
            if p_up[i] > 0.5:
                if "regime_filter" in conditions and spy_above_sma is not None:
                    if not spy_above_sma.iloc[i]:
                        continue
                in_pos = True
                entry = close[i]
                peak = high[i]
                size = 1.0 + 2.0 * (p_up[i] - 0.5) if "confidence_sizing" in conditions else 1.0
    return (equity - 1) * 100


def choose_conditions(symbol, df, spy_df=None):
    """Backtest each condition on top of baseline; keep what helps by
    MIN_GAIN_PCT. Returns (enabled_list, results_dict)."""
    sig = wf_signals(df)
    if sig is None or len(sig) < 60:
        return [], {"error": "not enough data"}
    spy_above = None
    if spy_df is not None and len(spy_df) >= 60:
        spy_above = (spy_df["close"] > spy_df["close"].rolling(50).mean()).reindex(sig.index).ffill()

    base = simulate(sig, spy_above, ())
    results = {"baseline_return": round(base, 2)}
    enabled = []
    for cond in CONDITIONS:
        if cond == "regime_filter" and spy_above is None:
            results[cond] = "skipped (no SPY data)"
            continue
        r = simulate(sig, spy_above, (cond,))
        gain = r - base
        results[cond] = {"return": round(r, 2), "gain_vs_base": round(gain, 2)}
        if gain >= MIN_GAIN_PCT:
            enabled.append(cond)
    return enabled, results


def gate_universe(symbols, cache_dir="data"):
    """Run condition selection for every ticker with a cached history.
    SPY cache (if present) drives the regime-filter evaluation."""
    state = load_state()
    spy_df = None
    spy_path = os.path.join(cache_dir, "SPY.csv")
    if os.path.exists(spy_path):
        spy_df = pd.read_csv(spy_path, index_col=0, parse_dates=True)
    for sym in symbols:
        path = os.path.join(cache_dir, f"{sym}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            enabled, results = choose_conditions(sym, df, spy_df)
            state[sym] = {"enabled_conditions": enabled, "backtest": results}
            print(f"  {sym:6s} enabled={enabled or 'none'} base={results.get('baseline_return')}%")
        except Exception as e:
            print(f"  {sym}: gate failed - {e}")
    save_state(state)
    return state


if __name__ == "__main__":
    from paper_trading import current_universe
    universe, desc = current_universe()
    print(f"Strategy gate on {len(universe)} tickers ({desc})")
    gate_universe(universe)

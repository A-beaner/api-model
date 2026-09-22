"""
continuous_train.py - Continuous self-training layer for the stock models.

Runs on every CI cycle (daily). For each ticker in the current universe it:
  1. INCREMENTALLY updates a persistent price cache (data/<SYM>.csv) with the
     newest daily candles - Alpaca stock bars first (same source as the
     charts and orders), Polygon fallback - the dataset grows forever.
  2. RETRAINS the direction model with walk-forward validation and persists it
     (models/<SYM>.joblib) so paper_trading.py can load a fresh, pre-validated
     model instead of training from scratch.
  3. LOGS validation accuracy to metrics.json over time and flags DRIFT when a
     model's accuracy decays vs its own history.
  4. Runs strategy_gate.py so the model CHOOSES which trading conditions
     (regime filter, dip exit, cooldown, confidence sizing) actually help each
     ticker, saved to strategy_state.json.
  5. Archives everything learned into LEARNING.md (committed back by CI).

Reality check: no model reads markets flawlessly. This layer maximizes what is
actually achievable: ever-growing data, frequent retraining, self-selected
conditions, and hard metrics that tell you when the edge is gone.

Env: POLYGON_API_KEY (required; Alpaca keys used when set). Config:
RETRAIN_DAYS (default 7), DRIFT_TOLERANCE (default 0.10),
MIN_WF_ACCURACY (default 0.52).
"""

import json
import os
import time
import urllib.request
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from stock_model import engineer_features, FEATURES

DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
METRICS_FILE = os.environ.get("METRICS_FILE", "metrics.json")
RETRAIN_DAYS = int(os.environ.get("RETRAIN_DAYS", "7"))
DRIFT_TOLERANCE = float(os.environ.get("DRIFT_TOLERANCE", "0.10"))
MIN_WF_ACCURACY = float(os.environ.get("MIN_WF_ACCURACY", "0.52"))
WF_FOLDS = 3

POLYGON_MIN_INTERVAL = 12.2  # free tier: 5 requests/min
_last_call = [0.0]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def cache_path(sym):
    return os.path.join(DATA_DIR, f"{sym}.csv")


def model_path(sym):
    return os.path.join(MODEL_DIR, f"{sym}.joblib")


def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            return json.load(f)
    return {}


def save_metrics(metrics):
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def polygon_daily(symbol, key, days=40):
    """Fetch candles: Alpaca stock bars first, Polygon fallback (paced)."""
    from stock_model import get_daily
    try:
        return get_daily(symbol, polygon_key=None, days=days)
    except Exception:
        wait = POLYGON_MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        df = get_daily(symbol, polygon_key=key, days=days)
        _last_call[0] = time.time()
        return df


def update_cache(symbol, key):
    """Merge newest bars into the persistent per-ticker price cache."""
    fresh = polygon_daily(symbol, key, days=40)
    path = cache_path(symbol)
    if os.path.exists(path):
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        df = pd.concat([old, fresh])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        # First time: pull 2 full years to seed the cache
        df = polygon_daily(symbol, key, days=730)
    df.to_csv(path)
    return df


def walk_forward_accuracy(data, folds=WF_FOLDS):
    train = data.dropna(subset=["target"])
    n = len(train)
    if n < 120:
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


def train_and_save(symbol, df):
    """Train on ALL cached history, validate with walk-forward, persist."""
    data = engineer_features(df)
    wf_acc = walk_forward_accuracy(data)
    train = data.dropna(subset=["target"])
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(train[FEATURES], train["target"])
    joblib.dump({"model": model, "wf_acc": wf_acc,
                 "trained_at": datetime.utcnow().isoformat(),
                 "n_samples": len(train)}, model_path(symbol))
    return wf_acc


def load_model(symbol, max_age_days=RETRAIN_DAYS):
    """Load a persisted model if it exists, is fresh, and is not drifted.
    Returns (model, wf_acc) or None."""
    path = model_path(symbol)
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
        trained_at = datetime.fromisoformat(bundle["trained_at"])
        age = (datetime.utcnow() - trained_at).days
        if age > max_age_days:
            return None
        metrics = load_metrics().get(symbol, {})
        if metrics.get("drift"):
            return None
        return bundle["model"], bundle.get("wf_acc", 0.0)
    except Exception:
        return None


def detect_drift(symbol, wf_acc, metrics):
    """Flag drift when the latest walk-forward accuracy decays more than
    DRIFT_TOLERANCE below the trailing average of past validations."""
    hist = metrics.get(symbol, {}).get("accuracy_history", [])
    if len(hist) >= 2:
        trailing = sum(h["wf_acc"] for h in hist[-5:]) / min(len(hist), 5)
        return wf_acc < trailing - DRIFT_TOLERANCE
    return False


def train_universe(symbols, key):
    metrics = load_metrics()
    for sym in symbols:
        try:
            mp = model_path(sym)
            fresh_model = os.path.exists(mp) and load_model(sym) is not None
            df = update_cache(sym, key)
            entry = metrics.setdefault(sym, {"accuracy_history": []})

            if fresh_model:
                # Model still fresh - just log data update, skip retrain cost
                print(f"  {sym:6s} cache={len(df)}d, model fresh (<= {RETRAIN_DAYS}d), kept")
                continue

            wf_acc = train_and_save(sym, df)
            drift = detect_drift(sym, wf_acc, metrics)
            entry["accuracy_history"].append(
                {"date": datetime.utcnow().strftime("%Y-%m-%d"), "wf_acc": round(wf_acc, 4)})
            entry["last_wf_acc"] = round(wf_acc, 4)
            entry["drift"] = drift
            entry["tradable"] = bool(wf_acc >= MIN_WF_ACCURACY and not drift)
            flag = " DRIFT!" if drift else ""
            print(f"  {sym:6s} cache={len(df)}d, retrained wf_acc={wf_acc:.3f}{flag}")
        except Exception as e:
            print(f"  {sym}: failed - {e}")
    save_metrics(metrics)
    print(f"\nMetrics written to {METRICS_FILE}; models in {MODEL_DIR}/; cache in {DATA_DIR}/")


def write_learning(metrics, state, universe_desc):
    """Append a dated entry to LEARNING.md - the repo's archive of what the
    models have learned: validation accuracy, drift, and which conditions the
    backtests enabled per ticker."""
    from datetime import datetime as _dt
    lines = []
    lines.append(f"\n## {_dt.utcnow().strftime('%Y-%m-%d %H:%M')} UTC - {universe_desc}\n")
    lines.append("| Ticker | Walk-fwd acc | Drift | Tradable | Conditions enabled |")
    lines.append("|---|---|---|---|---|")
    for sym in sorted(metrics):
        m = metrics[sym]
        conds = state.get(sym, {}).get("enabled_conditions", [])
        lines.append(f"| {sym} | {m.get('last_wf_acc', '-') } | "
                     f"{'YES' if m.get('drift') else 'no'} | "
                     f"{'yes' if m.get('tradable') else 'NO'} | "
                     f"{', '.join(conds) if conds else '-'} |")
    regime = state.get("_global", {}).get("regime_filter")
    lines.append(f"\nGlobal regime filter (SPY > 50d SMA): "
                 f"{'ENABLED' if regime else 'disabled'}\n")
    entry = "\n".join(lines)
    existing = ""
    if os.path.exists("LEARNING.md"):
        with open("LEARNING.md") as f:
            existing = f.read()
    if not existing.startswith("# LEARNING"):
        header = ("# LEARNING.md - What the models have learned\n\n"
                  "Auto-written by continuous_train.py after every CI run. "
                  "Each entry archives walk-forward accuracy, model drift, and "
                  "which strategy conditions backtesting enabled per ticker.\n")
        existing = header + existing
    with open("LEARNING.md", "w") as f:
        f.write(existing + entry)
    print("LEARNING.md updated")


if __name__ == "__main__":
    poly_key = os.environ.get("POLYGON_API_KEY")
    if not poly_key:
        raise SystemExit("POLYGON_API_KEY env var is required.")
    # Reuse the bot's universe logic so we train exactly what it scans
    from paper_trading import current_universe
    universe, desc = current_universe()
    print(f"Continuous training on {len(universe)} tickers ({desc})")
    # SPY is always cached too - the regime filter needs it
    train_universe(sorted(set(universe) | {"SPY"}), poly_key)

    # Meta-layer: backtest which conditions help per ticker, then archive
    from strategy_gate import gate_universe, load_state, save_state
    state = gate_universe(sorted(set(universe) | {"SPY"}))
    votes = [1 for s, v in state.items()
             if s != "_global" and "regime_filter" in v.get("enabled_conditions", [])]
    total = max(1, len([s for s in state if s != "_global"]))
    state["_global"] = {"regime_filter": len(votes) > total / 2}
    save_state(state)
    metrics = load_metrics()
    write_learning(metrics, state, desc)

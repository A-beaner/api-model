"""
paper_trading.py - Paper trading via Alpaca's paper API (fake money, real market).

Uses the trained direction model to generate a daily signal for SYMBOL:
  - model predicts UP   -> ensure we hold POSITION_QTY shares (buy if flat)
  - model predicts DOWN -> go flat (sell any position)

If ALPACA_API_KEY / ALPACA_SECRET_KEY are not set, runs in DRY-RUN mode:
it computes the signal and prints the intended order without submitting anything.

Keys come from environment variables - NEVER hardcode secrets:
  STOCK_API_KEY       (Alpha Vantage - market data)
  ALPACA_API_KEY      (Alpaca paper key id)
  ALPACA_SECRET_KEY   (Alpaca paper secret)
  API_KEY             (NVIDIA - optional AI commentary)
"""

import json
import os
import urllib.error
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import joblib

from stock_model import fetch_daily, engineer_features, train_model, FEATURES, SYMBOL

ALPACA_BASE = "https://paper-api.alpaca.markets"  # paper environment only
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-20b"
QTY = int(os.environ.get("POSITION_QTY", "1"))


def alpaca(method, path, key, secret, body=None):
    req = urllib.request.Request(
        ALPACA_BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


def get_position(symbol, key, secret):
    try:
        return alpaca("GET", f"/v2/positions/{symbol}", key, secret)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def llm_commentary(context, api_key):
    prompt = (
        "A paper-trading bot made a decision. Context:\n" + context +
        "\n\nIn 2-3 sentences, comment as a risk manager: is the logic sound "
        "and what should be watched?"
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
    stock_key = os.environ.get("STOCK_API_KEY")
    if not stock_key:
        raise SystemExit("STOCK_API_KEY env var is required (Alpha Vantage key).")
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")
    dry_run = not (alpaca_key and alpaca_secret)
    nvidia_key = os.environ.get("API_KEY")

    print(f"Mode: {'DRY-RUN (no Alpaca keys - no orders will be sent)' if dry_run else 'LIVE PAPER TRADING'}")
    print(f"Symbol: {SYMBOL} | Position size: {QTY} share(s)")

    # --- Data + model ---
    df = fetch_daily(SYMBOL, stock_key)
    data = engineer_features(df)
    model, _ = train_model(data)
    joblib.dump(model, "stock_model.joblib")

    # --- Signal from the latest row (features only; target is NaN by design) ---
    latest = data[FEATURES].iloc[-1:]
    asof = data.index[-1].date()
    pred = int(model.predict(latest)[0])
    proba = model.predict_proba(latest)[0]
    conf = proba[pred]
    signal = "LONG" if pred == 1 else "FLAT"
    print(f"\nSignal as of {asof}: {signal} "
          f"(P(up)={proba[1]:.2f}, P(down)={proba[0]:.2f}, confidence {conf:.2f})")

    # --- Act ---
    action = "HOLD"
    if dry_run:
        if pred == 1:
            action = f"WOULD BUY {QTY} {SYMBOL} (market) if flat"
        else:
            action = f"WOULD SELL/CLOSE any {SYMBOL} position"
        print(f"DRY-RUN: {action}")
    else:
        account = alpaca("GET", "/v2/account", alpaca_key, alpaca_secret)
        print(f"Paper account: equity=${float(account['equity']):,.2f} "
              f"buying_power=${float(account['buying_power']):,.2f}")
        pos = get_position(SYMBOL, alpaca_key, alpaca_secret)
        holding = int(float(pos["qty"])) if pos else 0
        print(f"Current position: {holding} share(s) of {SYMBOL}")

        if pred == 1 and holding < QTY:
            order = alpaca("POST", "/v2/orders", alpaca_key, alpaca_secret, {
                "symbol": SYMBOL, "qty": str(QTY - holding), "side": "buy",
                "type": "market", "time_in_force": "day"})
            action = f"BUY {QTY - holding} {SYMBOL} -> order {order.get('id')} ({order.get('status')})"
        elif pred == 0 and holding > 0:
            order = alpaca("POST", "/v2/orders", alpaca_key, alpaca_secret, {
                "symbol": SYMBOL, "qty": str(holding), "side": "sell",
                "type": "market", "time_in_force": "day"})
            action = f"SELL {holding} {SYMBOL} -> order {order.get('id')} ({order.get('status')})"
        else:
            action = f"HOLD (position {holding}, signal {signal})"
        print(f"Action: {action}")

    if nvidia_key:
        context = (f"Symbol {SYMBOL}, signal {signal} with confidence {conf:.2f}, "
                   f"action: {action}, mode: {'dry-run' if dry_run else 'paper'}.")
        print(f"\n--- AI risk commentary ({LLM_MODEL} via NVIDIA API) ---")
        try:
            print(llm_commentary(context, nvidia_key))
        except Exception as e:
            print(f"LLM commentary failed: {e}")


if __name__ == "__main__":
    main()

# Trade Rationale — Why Each Stock Was Bought

This document records, for every BUY in `track_record.csv`, the exact chain of
decisions that led to the trade. It is the human-readable companion to the
machine log: `track_record.csv` says **what** happened; this file says **why**.

---

## The decision pipeline (every buy must pass all gates)

A stock is only bought if it clears each of these, in order:

1. **Universe membership** — ticker is in the current rotating window
   (`WINDOW_SIZE` tickers from `TICKER_POOL`, rotating every `ROTATE_EVERY_DAYS`)
   or is already held.
2. **Walk-forward validation** — the ticker's model scores
   `wf_acc >= MIN_WF_ACCURACY` (raised 0.52 → **0.54** on 2026-09-22 after the
   loss review) on expanding-window out-of-sample data. Fail → skipped.
3. **Signal strength** — `P(up) >= P_UP_MIN` (raised 0.5 → **0.55** on
   2026-09-22). The top `TOP_N` (3) by P(up) become picks.
4. **Market gate** — Alpaca clock confirms the market is open.
5. **Strategy conditions** — from `strategy_config.json`: regime filter
   (SPY > SMA-50), weekend rule, cooldown (no re-buy within 3 days of a sell),
   and (since 2026-09-22) **buys are blocked if any SELL failed that run**.
6. **Risk sizing** — quantity sized so the ATR stop (`STOP_ATR_MULT` × ATR-14)
   risks at most `RISK_PER_TRADE` (1%) of equity, capped by the slot budget
   (daily trade limit ÷ TOP_N) and `confidence_sizing` multiplier when enabled.
7. **Order type** — bracket order: market buy + ATR stop-loss + take-profit at
   `TP_RR` (2.0) reward:risk.

---

## Buy log

### 2026-09-15 05:10 — GS, NOW, MA — ALL FAILED (HTTP 403)
- **Picks:** GS 0.74 | NOW 0.67 | MA 0.65 (top-3 of 30 scanned, window period 2958)
- **Rationale:** First live run. All three had the highest P(up) in the window and
  passed walk-forward validation at the time.
- **Outcome:** All three orders rejected (403 Forbidden). No positions opened.
- **Lesson:** orders were attempted while the market was closed / with an auth
  issue — the market-open gate and 403-retry logic were added later.

### 2026-09-15 15:39 — GS, NOW, MA (~$48.1k total)
- **BUY 16 GS (~$15,815)**, **BUY 117 NOW (~$16,655)**, **BUY 29 MA (~$16,658)**
- **Picks:** GS 0.74 | NOW 0.67 | MA 0.65 — same signals, now accepted.
- **Rationale:** Highest-confidence validated signals; roughly equal-weight
  ~$16k slots (daily limit $50k ÷ 3), pre-risk-sizing era.
- **Outcome:** All three sold 09-17 within ~9–32h as the window rotated and
  signals changed. Equity fell 100,335 → 99,463 — the cluster where most of
  the early losses occurred (NOW in particular). **Churn at loss.**

### 2026-09-17 00:08 — COIN (~$16.5k)
- **BUY 96 COIN (~$16,523)** after SELL 117 NOW.
- **Picks:** COIN 0.69 | MA 0.69 | GS 0.65.
- **Rationale:** COIN entered the top-3 with the joint-highest P(up) after the
  rotation; NOW dropped out and was sold.
- **Outcome:** Sold the same day (09-17 23:56) when it fell out of the picks.
  Same-day round trip — pure churn cost.

### 2026-09-17 23:56 — JPM, PANW, MET (~$49.4k total)
- **BUY 47 JPM (~$16,399)**, **BUY 44 PANW (~$16,529)**, **BUY 171 MET (~$16,626)**
  after selling COIN, GS, MA.
- **Picks:** JPM 0.70 | PANW 0.64 | MET 0.63.
- **Rationale:** Full rotation — previous picks' signals faded, these three
  took the top slots.
- **Outcome:** JPM and PANW sold 09-20 (roughly flat, equity 99,720 → 99,938);
  MET sold 09-21. Another full portfolio turnover in ~24h.

### 2026-09-20 03:04 — MS, PNC (risk-sized era begins)
- **BUY 82 MS (~$16,612, stop $191.87, tp $224.00, risks $878 = 0.88% equity)**
- **BUY 71 PNC (~$16,555, stop $223.45, tp $252.61, risks $690 = 0.69% equity)**
- **Picks:** MS 0.68 | MET 0.58 | PNC 0.57.
- **Rationale:** First buys using ATR risk-based sizing + bracket orders
  (stop-loss and take-profit attached). Note MET was already held, so only two
  new buys.
- **Outcome:** MS still held (blocked from rebuy 09-21 by a failed 403); PNC
  sold 09-22.

### 2026-09-21 05:17 — BLK (the questionable one)
- **BUY 15 BLK (~$16,047, stop $1,018.65, tp $1,172.03, risks $767 = 0.77% equity)**
- **Picks:** BLK 0.69 | MS 0.65 | SCHW 0.59.
- **Rationale at the time:** BLK had the highest P(up) of 29 scanned tickers and
  passed the walk-forward gate then in effect (≥0.52).
- **Why it shouldn't have happened:** the next day's evaluation scored BLK at
  **wf_acc 0.4889 — worse than a coin flip, `tradable: false`**. The buy rested
  on a stale/optimistic validation. It was also placed while two matching SELLs
  (JPM, PANW) were failing with 403s.
- **Fix committed 2026-09-22:** `MIN_WF_ACCURACY` raised to 0.54 and
  "block all buys if any sell failed" — both would have blocked this trade.
- **Outcome:** BLK (15 sh) still held as of the last run, with its bracket
  stop/TP managing the exit.

### 2026-09-21 05:17 — MS, SCHW — FAILED (HTTP 403)
- BUY 82 MS and BUY 158 SCHW rejected; only BLK filled that run.

### 2026-09-22 00:40 — SCHW
- **BUY 158 SCHW (~$16,630, stop $101.26, tp $113.22, risks $630 = 0.63% equity)**
  after SELL 71 PNC.
- **Picks:** BLK 0.69 | MS 0.65 | SCHW 0.59 (unchanged window).
- **Rationale:** SCHW was the remaining unfilled pick from the prior run; the
  09-21 403 failure was retried successfully here.
- **Outcome:** Open at last record. **Note:** metrics on 09-22 show SCHW at
  wf_acc 0.4889 (`tradable: false`) — same stale-validation weakness as BLK.
  Would now be blocked by the 0.54 gate.

---

## What the rationale review revealed (2026-09-22)

1. **Churn was the biggest leak** — 4 full portfolio rotations in 7 days against
   a measured signal edge of only ~0.2% per trade. Fixed with `MIN_HOLD_DAYS=3`
   and the `cooldown` condition.
2. **Two buys (BLK, SCHW) rested on validation that didn't survive 24h** —
   fixed by raising `MIN_WF_ACCURACY` to 0.54.
3. **403 order failures corrupted the buy/sell pairing** — fixed with
   retry-with-backoff in `alpaca()` and the failed-sell buy block.
4. **Sells previously logged no prices** — fixed: every sell now records fill
   price and realized P&L vs entry in the track record.

*Rules current as of commits fb5b6ed, 3a508bd, 0b0d3d7. When `paper_trading.py`
changes the gates, update the pipeline section above.*

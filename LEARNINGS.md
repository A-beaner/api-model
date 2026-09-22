# LEARNINGS.md - Model Knowledge Archive

This file is the bot's long-term memory. `strategy_lab.py` appends a dated
entry after every evaluation cycle, recording which optional trading
conditions were tested against out-of-sample data, what the evidence showed,
and what the model decided to enable or disable in `strategy_config.json`.

## How to read this

- Each entry lists every condition with **ENABLED** or **OFF** and the
  measured evidence (JSON) behind the decision.
- Decisions are re-evaluated as the data cache grows, so conditions can turn
  on or off over time as the market regime changes.
- `strategy_config.json` is the machine-readable counterpart;
  `paper_trading.py` obeys it on every run.

## 2026-09-22 04:25 UTC

Evaluated 30 tickers, 1745 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00177, "fwd_below_sma50": 0.00329}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00273, "fwd_non_friday": 0.00218}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0329}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00258, "fwd_after_up_day": 0.00195}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

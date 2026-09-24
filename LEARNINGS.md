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

## 2026-09-22 22:22 UTC

Evaluated 30 tickers, 1769 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00192, "fwd_below_sma50": 0.00373}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00406, "fwd_non_friday": 0.00218}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0282}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00292, "fwd_after_up_day": 0.00211}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-09-22 22:55 UTC

Evaluated 30 tickers, 1769 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00192, "fwd_below_sma50": 0.00373}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00406, "fwd_non_friday": 0.00218}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0282}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00292, "fwd_after_up_day": 0.00211}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-09-23 04:23 UTC

Evaluated 30 tickers, 1769 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00192, "fwd_below_sma50": 0.00373}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00406, "fwd_non_friday": 0.00218}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0282}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00292, "fwd_after_up_day": 0.00211}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-09-23 22:27 UTC

Evaluated 30 tickers, 1799 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00121, "fwd_below_sma50": 0.00346}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00385, "fwd_non_friday": 0.00154}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0482}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00275, "fwd_after_up_day": 0.00115}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-09-24 03:53 UTC

Evaluated 40 tickers, 2364 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00152, "fwd_below_sma50": 0.00352}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.0026, "fwd_non_friday": 0.0021}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0393}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00254, "fwd_after_up_day": 0.00184}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-09-24 12:27 UTC

Evaluated 40 tickers, 2364 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00152, "fwd_below_sma50": 0.00352}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.0026, "fwd_non_friday": 0.0021}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0393}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00254, "fwd_after_up_day": 0.00184}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

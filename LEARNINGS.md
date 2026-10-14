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

## 2026-09-25 22:38 UTC

Evaluated 40 tickers, 2381 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.0016, "fwd_below_sma50": 0.00346}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00254, "fwd_non_friday": 0.00216}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.043}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00239, "fwd_after_up_day": 0.00207}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-09-29 23:06 UTC

Evaluated 40 tickers, 2382 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00168, "fwd_below_sma50": 0.00347}`
- **weekend_rule**: ENABLED - `{"fwd_friday": 0.0021, "fwd_non_friday": 0.00235}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.04}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00256, "fwd_after_up_day": 0.00204}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-09-30 17:41 UTC

Evaluated 40 tickers, 2376 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00175, "fwd_below_sma50": 0.00319}`
- **weekend_rule**: ENABLED - `{"fwd_friday": 0.00216, "fwd_non_friday": 0.00227}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0367}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00252, "fwd_after_up_day": 0.00199}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-09-30 23:08 UTC

Evaluated 40 tickers, 2376 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00175, "fwd_below_sma50": 0.00319}`
- **weekend_rule**: ENABLED - `{"fwd_friday": 0.00216, "fwd_non_friday": 0.00227}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0367}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00252, "fwd_after_up_day": 0.00199}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-10-02 23:09 UTC

Evaluated 50 tickers, 2961 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00144, "fwd_below_sma50": 0.00244}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00191, "fwd_non_friday": 0.0018}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0253}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00187, "fwd_after_up_day": 0.00178}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-10-06 00:54 UTC

Evaluated 50 tickers, 2953 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00139, "fwd_below_sma50": 0.00247}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00187, "fwd_non_friday": 0.00179}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0317}`
- **cooldown**: ENABLED - `{"fwd_after_down_day": 0.00177, "fwd_after_up_day": 0.00183}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-10-06 23:12 UTC

Evaluated 50 tickers, 2969 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00139, "fwd_below_sma50": 0.00254}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00195, "fwd_non_friday": 0.00181}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0366}`
- **cooldown**: ENABLED - `{"fwd_after_down_day": 0.00181, "fwd_after_up_day": 0.00186}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-10-07 23:40 UTC

Evaluated 50 tickers, 2969 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00148, "fwd_below_sma50": 0.00248}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00206, "fwd_non_friday": 0.00182}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0375}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00194, "fwd_after_up_day": 0.00178}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-10-09 23:25 UTC

Evaluated 60 tickers, 3541 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00171, "fwd_below_sma50": 0.0023}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00215, "fwd_non_friday": 0.00188}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0362}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00209, "fwd_after_up_day": 0.00177}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-10-13 01:34 UTC

Evaluated 60 tickers, 3543 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00184, "fwd_below_sma50": 0.00215}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00205, "fwd_non_friday": 0.00194}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.044}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00213, "fwd_after_up_day": 0.00179}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-10-14 00:17 UTC

Evaluated 60 tickers, 3561 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00177, "fwd_below_sma50": 0.00213}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00218, "fwd_non_friday": 0.00184}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0457}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00206, "fwd_after_up_day": 0.00175}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

## 2026-10-14 22:28 UTC

Evaluated 60 tickers, 3573 out-of-sample signals.

- **regime_filter**: OFF - `{"fwd_above_sma50": 0.00196, "fwd_below_sma50": 0.00231}`
- **weekend_rule**: OFF - `{"fwd_friday": 0.00265, "fwd_non_friday": 0.00197}`
- **exposure_cap**: OFF - `{"reason": "edge sufficient"}`
- **confidence_sizing**: ENABLED - `{"p_up_vs_fwd_corr": 0.0249}`
- **cooldown**: OFF - `{"fwd_after_down_day": 0.00228, "fwd_after_up_day": 0.00192}`
- **correlation_guard**: ENABLED - `{"reason": "always-on structural guard"}`
- **trailing_stop**: ENABLED - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **breakeven_stop**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **partial_take_profit**: OFF - `{"reason": "exit-side condition; set by risk policy, will A/B when trade count > 100"}`
- **earnings_blackout**: OFF - `{"reason": "undetermined - needs earnings calendar feed"}`

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

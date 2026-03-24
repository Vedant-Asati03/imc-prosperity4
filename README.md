# IMC Prosperity 4

## Setup & Run

```bash
uv sync

source .venv/bin/activate

backtest --round 0 --plot
```

> check `help` for more options

## What the current backtester enforces

- Round data loaded from `data/roundX/prices_round_*_day_*.csv`
- Position limits are read from `Trader.POSITION_LIMITS` and must include all products in the round data.
- Per-tick runtime cap: `Trader.run` must finish within `900ms`.
- `traderData` is truncated to `50_000` characters.
- `conversions` must be `0` or `None` (conversion simulation is not implemented).

## Important simulation limitation

- This is a **crossing-only** replay.
- It simulates immediate matches against the current order book.
- It does **not** simulate passive/resting orders getting filled later.

## Common errors

- `Strategy file not found`:
  - Create `strategy/roundX/main.py` for the selected round.
- `Missing POSITION_LIMITS entries`:
  - Add all round products to `Trader.POSITION_LIMITS`.
- `does not simulate conversions`:
  - Return `conversions = 0` (or `None`) from `run`.
- `Trader.run exceeded 900ms`:
  - Optimize strategy logic.

## Quick links & Reference docs

- [playground](https://prosperity.imc.com/game)
- [storyline](/resources/storyline.md)
- [trading-glossary](/resources/trading-glossary.md)
- [game-mechanics](/resources/game-mechanics.md)
- [algorithm-implementation](/resources/algorithm-implementation.md)

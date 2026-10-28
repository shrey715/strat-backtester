# strat-backtester

A pluggable options-strategy backtesting engine: second-by-second simulation, a realistic execution model, built-in risk controls, a full risk-metrics suite, a Streamlit dashboard and PDF tearsheet export.

It runs out of the box on a built-in synthetic market — **no data download needed** — and also reads raw NSE tick data.

```bash
pip install -e ".[app]"      # or: uv sync --extra app
stratbt app                  # dashboard at http://localhost:8501
stratbt run iron_condor      # or run a backtest straight from the terminal
```

---

## Contents

- [Quickstart](#quickstart) · [Strategies](#built-in-strategies) · [Execution & costs](#execution-model) · [Risk limits](#risk-limits)
- [Data providers](#data-providers) · [Write a strategy](#write-your-own-strategy) · [Brokers & going live](#brokers-and-going-live)
- [Docker](#docker) · [Layout](#project-layout) · [Tests](#tests) · [Limitations](#limitations)

---

## Quickstart

### Command line

```bash
stratbt strategies                                   # list strategies + parameters
stratbt run short_strangle --param width=3           # synthetic market, default execution
stratbt run iron_condor --execution realistic --pdf report.pdf
stratbt run atm_straddle --provider nse --data-dir data/allData
stratbt run short_straddle --max-daily-loss 2000 --max-contracts 4
```

### Python

```python
from stratbt import run_backtest, SyntheticProvider, ExecutionModel, RiskLimits
import stratbt.metrics as metrics

market = SyntheticProvider(n_days=21, seed=7, iv_premium=0.10)
result = run_backtest(
    "short_strangle", market, params={"width": 3},
    execution=ExecutionModel.realistic(),
    risk=RiskLimits(max_daily_loss=2_000),
)
print(metrics.summary_table(result).T)
```

`notebooks/00_quickstart.ipynb` walks through the same thing and shows how to register a custom strategy.

### Dashboard

`stratbt app` opens the Streamlit dashboard: pick a data source and a strategy (the parameter sliders are generated from the strategy's spec), set execution costs and risk limits, run, then explore the Overview / PnL / Risk / Activity / Detail / Costs / Log tabs and download a PDF tearsheet or the trade CSV. Point it at NSE tick data with `stratbt app --data-dir path/to/allData`.

---

## Built-in strategies

| Key | Structure | Notes |
|---|---|---|
| `atm_straddle` | Long ATM straddle, rolled every time the nearest strike changes | Maximum trade frequency; the original project strategy |
| `fixed_window` | Long ATM straddle, entered once, held, exited before close | Baseline without rolling |
| `short_straddle` | Short ATM straddle | Unlimited risk |
| `long_strangle` | Long OTM call + put, `width` strikes from ATM | |
| `short_strangle` | Short OTM call + put, `width` strikes from ATM | Unlimited risk |
| `iron_condor` | Short strangle at `width` + long wings `wing` strikes further out | Defined risk |

The window strategies share `entry_delay_s` / `exit_buffer_s`, and the multi-leg ones accept `take_profit` / `stop_loss` (in price points). Add your own multi-leg structure in a few lines — see `strategy/structures.py`.

---

## Execution model

By default an order fills **one second after the decision**. A strategy sees the price of second *t* only once that second is over, so filling at second *t* itself (the original behaviour) quietly looks ahead. `ExecutionModel` makes the choice explicit:

| Preset | Fill | Costs |
|---|---|---|
| `ExecutionModel.instant()` | same second (optimistic, legacy) | none |
| `ExecutionModel()` *(default)* | next second | none |
| `ExecutionModel.realistic()` | next second | 25 bps half-spread (1c floor), $0.65/contract |

Or set `delay_s`, `half_spread_bps`, `min_half_spread`, `commission_per_contract`, `commission_bps` directly. Fees are deducted from PnL and drawdowns, not bolted on afterwards. PnL is scaled by the provider's contract multiplier (100 for the synthetic US-style market; 1 = index points for NSE unless you pass `lot_sizes`).

Effect on the original strategy (rolling ATM straddle, NSE Nov 2022, both underliers, index points):

| Execution | Combined PnL |
|---|---|
| `instant()` — as originally published | −2,014 |
| next-second fills | −1,687 |
| next-second + 25 bps half-spread | −14,013 |

The strategy loses in every case; the point is that the headline number moves a lot with the fill assumption, which is why it is now a parameter.

Other engine behaviour worth knowing: positions still open at the last second are flattened (`flatten_at_close=True`); while orders are in flight the strategy is not asked for new ones; orders it cannot price are rejected and recorded in `DayResult.rejections`.

## Risk limits

`RiskLimits` are enforced by the broker between strategy and market — identically in backtest and (with a live adapter) in production:

- `max_daily_loss` — once net equity for the day hits the limit, everything is flattened and new opening orders are refused until the next day.
- `max_open_contracts` — total absolute contracts across all instruments.
- `max_order_qty` — size of any single order.

Orders that reduce risk are always allowed.

---

## Data providers

The engine only talks to a `DataProvider`. Three ship with the package:

| Provider | Use |
|---|---|
| `SyntheticProvider` | Seeded GBM underlying + Black–Scholes options (SPY/QQQ lookalikes, 09:30–16:00). Zero setup; ideal for developing strategies and testing plumbing. Results say nothing about real markets. |
| `NSETickProvider(data_dir, lot_sizes=None)` | Raw NSE tick CSVs laid out as `NSE_YYYYMMDD/{Options,Futures (Continuous)}/…` (not included — see below). |
| your own | Subclass `DataProvider` (≈5 methods: days, expiries, strikes, underlying series, option series). |

**NSE tick data is not shipped with this repository** — it is large and usually licensed. Put your copy in `data/allData/` (git-ignored) or set `STRATBT_DATA_DIR`. The tests and notebooks that need it skip themselves when it is absent.

## Write your own strategy

```python
from stratbt import BaseStrategy, Order, StrategySpec, ParamSpec, register
from stratbt.instruments import Instrument
from stratbt.strategy.base import nearest_strike

class BuyAtmCall(BaseStrategy):
    allow_short = False                      # set True if you sell options you don't hold

    def __init__(self, underlier, expiry, strikes, entry_delay_s=600):
        self.u, self.e, self.k = underlier, expiry, strikes
        self.delay, self.done = entry_delay_s, False

    def on_second(self, ts, futures_price, option_price, holdings):
        # ts: pd.Timestamp · option_price(inst) -> float | None · holdings: {inst: (avg_entry, signed_qty)}
        if self.done or (ts - ts.normalize()).total_seconds() < 9.5 * 3600 + self.delay:
            return []
        call = Instrument(self.u, self.e, nearest_strike(self.k, futures_price), "CE")
        if option_price(call) is None:
            return []
        self.done = True
        return [Order(call, "BUY")]

register(StrategySpec("buy_atm_call", "Buy ATM Call", BuyAtmCall, "One ATM call, held to the close.",
                      [ParamSpec("entry_delay_s", 600, 0, 3600, 60, "Entry delay (s)")]))
```

Registered strategies appear in `stratbt strategies`, `stratbt run` and the dashboard. Packages can also publish strategies without touching this repo via the `stratbt.strategies` entry-point group. A strategy only ever receives prices known at the current second and cannot read files, so it cannot see the future.

## Brokers and going live

`stratbt.broker.Broker` is the interface between a strategy's `Order`s and a market: `submit(order, ts, ref_price) -> Fill | None`, `holdings`, `equity(prices)`. `PaperBroker` implements it for backtests and paper trading (execution model + risk limits + P&L ledger).

A live adapter for a retail brokerage API would implement the same interface, so a strategy that backtests under `PaperBroker` runs unchanged against it. **No live adapter ships with this package**, and nothing here has been tested against a real brokerage — treat live trading as future work and start with paper mode.

---

## Docker

```bash
docker compose up --build            # http://localhost:8501
```

The image has no data baked in. By default it serves the synthetic market; to use NSE data, point the volume at it:

```bash
STRATBT_DATA_DIR=/path/to/allData docker compose up --build
```

## Project layout

```
src/stratbt/
├── providers/       DataProvider interface · SyntheticProvider · NSETickProvider
├── strategy/        BaseStrategy · rolling/fixed straddles · multi-leg structures · registry
├── engine.py        Backtest loop, DayResult / BacktestResult
├── broker.py        Broker interface · PaperBroker (fills, fees, risk)
├── execution.py     ExecutionModel (delay, spread, commissions)
├── risk.py          RiskLimits
├── position.py      Ledger: long/short, average cost, multiplier, fees
├── metrics.py       Scalar + risk metrics (VaR/CVaR, Ulcer, Omega, bootstrap Sharpe CI) and plots
├── report/          PDF tearsheet (fetch → layout → render)
├── dashboard/       Streamlit app
├── cli.py           `stratbt` command
├── loader.py, universe.py, instruments.py   NSE tick-file helpers
notebooks/           quickstart + the original NSE exploration
tests/               pytest suite (synthetic-market tests run anywhere)
```

## Tests

```bash
uv sync --extra app          # or: pip install -e ".[app]" pytest
uv run pytest
```

The synthetic-market, unit and dashboard tests need nothing else. Tests that read the real NSE dataset are skipped automatically when `data/allData` is absent.

## Limitations

- **Fills are against last-trade prices** (plus modelled spread), not a recorded order book; there is no market-impact or queue model.
- **No margin or buying-power model.** Short strategies can lose more than any account would allow; use `RiskLimits` as a stand-in.
- **No early exercise, assignment or expiry settlement** beyond flattening at the session close.
- The synthetic market uses constant volatility; real surfaces have skew and term structure.
- One session at a time: no overnight positions.

Not investment advice. Backtests are not predictions.

## License

[MIT](LICENSE)

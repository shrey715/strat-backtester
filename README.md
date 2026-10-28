# strat-backtester

Backtest options strategies second by second — on real prices for NSE, NASDAQ, NYSE and more (via Yahoo Finance) or on a built-in synthetic market — with a realistic execution model, risk limits, a risk-metrics suite, a Streamlit dashboard, a PDF tearsheet, and a live/paper trading runtime built on the same engine.

```bash
pip install -e ".[app]"      # or: uv sync --extra app
stratbt app                  # dashboard at http://localhost:8501
stratbt run iron_condor --provider yahoo --tickers SPY        # real SPY bars
stratbt run short_strangle --provider yahoo --tickers ^NSEI   # Nifty 50
stratbt run iron_condor                                       # offline synthetic market
```

---

## Contents

[Quickstart](#quickstart) · [What is real, what is modelled](#what-is-real-and-what-is-modelled) · [Strategies](#built-in-strategies) · [Execution & costs](#execution-model) · [Risk limits](#risk-limits) · [Data providers](#data-providers) · [Write a strategy](#write-your-own-strategy) · [Paper & live trading](#paper-and-live-trading) (incl. [real venues](#real-venues-alpaca--interactive-brokers--public)) · [Use from an AI agent (MCP)](#use-it-from-an-ai-agent-mcp) · [Reports](#reports) · [Docker](#docker) · [Layout](#project-layout) · [Tests](#tests) · [Limitations](#limitations)

---

## Quickstart

### Command line

```bash
stratbt strategies                                          # list strategies + parameters
stratbt run short_strangle --param width=3                  # synthetic market
stratbt run iron_condor --provider yahoo --tickers SPY QQQ --interval 5m --lookback 30 --execution realistic --pdf report.pdf
stratbt run short_straddle --provider yahoo --tickers RELIANCE.NS TCS.NS --interval 15m --multiplier 500
stratbt run iron_condor --provider yahoo --tickers ^NSEI ^NSEBANK
stratbt paper iron_condor --provider yahoo --tickers SPY --audit-log audit.jsonl      # paper-trade the latest session
```

### Python

```python
from stratbt import run_backtest, YahooFinanceProvider, ExecutionModel, RiskLimits
import stratbt.metrics as metrics

market = YahooFinanceProvider(["SPY", "QQQ"], interval="5m", lookback_days=30)
result = run_backtest("short_strangle", market, params={"width": 3},
                      execution=ExecutionModel.realistic(),
                      risk=RiskLimits(max_daily_loss=2_000))
print(metrics.summary_table(result).T)
```

`notebooks/00_quickstart.ipynb` runs offline and shows how to register your own strategy.

### Dashboard

`stratbt app` opens the Streamlit dashboard. Pick a data source (Yahoo Finance tickers or the synthetic market), a strategy (parameter sliders are generated from its spec), execution costs and risk limits, and press **Run backtest**. A landing page offers one-click examples. Results open with a plain-language "What happened" summary and six headline numbers, then tabs for Overview, Risk, Behaviour, Costs, Days & trades and a per-day Drill-down. Each run is kept in a history, so a **Compare** tab overlays strategies side by side. **Export** gives a PDF tearsheet, the trade CSV and the exact `stratbt run …` command that reproduces the run.

---

## What is real and what is modelled

| | Yahoo Finance provider | Synthetic provider |
|---|---|---|
| Underlying prices | **Real** intraday bars (US, NSE, …) | Simulated (GBM) |
| Option prices | **Modelled**: Black–Scholes at every second | Modelled: Black–Scholes |
| Implied vol | Trailing realised vol of the *previous* days × (1 + `iv_premium`, default 10%) | `annual_vol` × (1 + `iv_premium`) |

Yahoo publishes only the *current* option chain, never historical option quotes, so there is nothing real to replay for the options. Treat Yahoo-based results as "this strategy on real price paths, with options priced by a model" — a sound way to study structure, timing and costs, but not a substitute for tests on recorded option quotes.

Within a bar the underlying is held at the bar's open and steps to its close in the bar's final second, so a strategy never sees a bar's close before the bar has ended. Yahoo keeps about 29 days of 1-minute bars, 59 days of 5–30 minute bars and 2 years of hourly bars.

## Built-in strategies

| Key | Structure | Notes |
|---|---|---|
| `atm_straddle` | Long ATM straddle, rolled whenever the nearest strike changes | Very high trade frequency |
| `fixed_window` | Long ATM straddle, entered once, held, exited before close | Baseline without rolling |
| `short_straddle` | Short ATM straddle | Unlimited risk |
| `long_strangle` | Long OTM call + put, `width` strikes from ATM | |
| `short_strangle` | Short OTM call + put, `width` strikes from ATM | Unlimited risk |
| `iron_condor` | Short strangle at `width` plus long wings `wing` strikes further out | Defined risk |

The multi-leg ones accept `take_profit` / `stop_loss` (in price points) and all window strategies share `entry_delay_s` / `exit_buffer_s`. A new structure is a few lines — see `strategy/structures.py`.

## Execution model

By default an order fills **one second after the decision**. A strategy sees the price of second *t* only once that second is over, so filling in second *t* itself would quietly look ahead.

| Preset | Fill | Costs |
|---|---|---|
| `ExecutionModel.instant()` | same second (optimistic) | none |
| `ExecutionModel()` *(default)* | next second | none |
| `ExecutionModel.realistic()` | next second | 25 bps half-spread (1c floor), $0.65/contract |

Set `delay_s`, `half_spread_bps`, `min_half_spread`, `commission_per_contract` or `commission_bps` directly for anything else. Fees and spread are deducted from PnL and drawdowns, and reported separately (gross → slippage → fees → net). PnL is scaled by the contract multiplier (100 for US-style USD underlyings; 1 = index points elsewhere unless you pass one, e.g. `multiplier=75` for a Nifty lot).

Also: positions still open at the last second are flattened; while orders are in flight the strategy is not asked for new ones; unpriceable orders are rejected and recorded in `DayResult.rejections`.

## Risk limits

`RiskLimits` are enforced by the broker between strategy and market — identically in backtest and in the live runner:

- `max_daily_loss` — once net equity for the day hits the limit, everything is flattened and new opening orders are refused until the next day.
- `max_open_contracts` — total absolute contracts across all instruments.
- `max_order_qty` — size of any single order.

Orders that reduce risk are always allowed.

---

## Data providers

The engine only talks to a `DataProvider`. Shipped:

| Provider | Use |
|---|---|
| `YahooFinanceProvider(tickers, interval, lookback_days, iv_premium, …)` | Real bars for any Yahoo symbol: US (`SPY`, `AAPL`), NSE (`^NSEI`, `^NSEBANK`, `RELIANCE.NS`), others. One market per run (shared trading hours). Downloads are cached on disk (`~/.cache/stratbt`, override with `STRATBT_CACHE_DIR`). Needs `pip install 'strat-backtester[data]'`. |
| `SyntheticProvider(n_days, seed, annual_vol, iv_premium, …)` | Seeded GBM underlying, fully offline and reproducible. Good for developing strategies and for tests. |
| your own | Subclass `DataProvider` (days, expiries, strikes, an underlying series and an option series), or `ModelPricedProvider` and supply only the underlying and a vol assumption. Plug a recorded options feed in here and nothing else changes. |

The Yahoo provider infers session hours from the bars, drops incomplete sessions (half-days, today's unfinished session), guesses currency and risk-free rate from the symbol, and picks a conventional strike step (about 0.25% of the price). All of these can be overridden.

## Write your own strategy

```python
from stratbt import BaseStrategy, Order, StrategySpec, ParamSpec, register
from stratbt.instruments import Instrument
from stratbt.strategy.base import nearest_strike

class BuyAtmCall(BaseStrategy):
    allow_short = False                      # True if you sell options you don't hold
    unlimited_risk = False                   # True for naked short options (reports then warn)

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

Registered strategies appear in `stratbt strategies`, `stratbt run`, `stratbt paper` and the dashboard. Packages can also ship strategies through the `stratbt.strategies` entry-point group. A strategy only receives prices known at the current second and cannot read files, so it cannot see the future.

---

## Paper and live trading

`stratbt.live.LiveRunner` runs the **same decision loop as the backtester** against a stream of prices and any `Broker`:

```python
from stratbt import PaperBroker, ExecutionModel
from stratbt.live import LiveRunner, ReplayFeed, AuditLog, KillSwitch

feed = ReplayFeed(provider, day, "SPY")                      # or your own QuoteFeed
strategy = registry.get("iron_condor").factory(provider)("SPY", feed.expiry, feed.strikes)
broker = PaperBroker(100, ExecutionModel.realistic(), allow_short=True)
summary = LiveRunner(strategy, feed, broker, ExecutionModel.realistic(),
                     audit=AuditLog("audit.jsonl"), kill=KillSwitch("STOP"),
                     state_path="state.json").run()
```

What it adds on top of the backtest loop:

- **Reconciliation** — on start, the broker's positions must match the saved state or it refuses to trade (`adopt_existing=True` overrides, and is logged).
- **Audit log** — append-only JSON lines: every decision, fill, rejection, risk halt and kill.
- **Kill switch** — create the file (or call `kill()`): everything is flattened and the runner stops.
- **State persistence** after every fill, so a restart can resume.
- **Parity** — a test replays sessions through the runner and the backtester and requires identical fills and PnL for four strategies, two execution models and a risk-halt scenario.

`stratbt paper` runs the latest (or a chosen) session this way from the command line, with no venue involved.

### Real venues (Alpaca · Interactive Brokers · Public)

The same runner can trade against a real quote feed and broker. Three adapters ship, each a *feed* (quotes → per-second snapshots) plus a *broker* (orders → venue):

| Venue | SDK | Markets | Paper account? | Install |
|---|---|---|---|---|
| **Alpaca** | `alpaca-py` | US equities and options | Yes (separate keys) | `pip install 'strat-backtester[alpaca]'` |
| **Interactive Brokers** | `ib_async` | Almost everywhere, including NSE | Yes (`DU…` accounts) | `pip install 'strat-backtester[ibkr]'` |
| **Public.com** | `publicdotcom-py` | US equities and options | **No** — dry-run uses Public's pre-flight check | `pip install 'strat-backtester[public]'` |

```bash
export ALPACA_API_KEY=… ALPACA_SECRET_KEY=…           # credentials live in the environment, never on the command line
stratbt live iron_condor --broker alpaca --ticker SPY                                  # 1. dry-run (default): nothing is sent
stratbt live iron_condor --broker alpaca --ticker SPY --mode paper --max-daily-loss 200   # 2. Alpaca's paper account
STRATBT_CONFIRM_REAL_MONEY=I_ACCEPT_REAL_MONEY_TRADING_RISK \
  stratbt live iron_condor --broker alpaca --ticker SPY --mode live --max-daily-loss 100    # 3. real money
```

**Safety model** — it is deliberately hard to lose money by accident:

- **Three modes, safest first.** `dry-run` (the default) watches real quotes, makes real decisions and records what it *would* do, but **never sends an order** — and where a venue offers a validation call (Public's pre-flight) it uses it. `paper` uses the venue's paper account. `live` is real money.
- **Real money needs two deliberate acts**: a confirmation token (typed at an interactive prompt, or set in `STRATBT_CONFIRM_REAL_MONEY`) *and* a daily-loss limit. `paper` also requires the loss limit.
- **Small by default:** one contract per order, four open at once. Callers can raise these but never silently exceed them.
- **Marketable limit orders only** — never market orders — priced off the live quote. Orders are refused on stale quotes, one-sided markets, or spreads wider than 30% of the mid.
- **Unfilled orders are cancelled** after a timeout; partial fills are booked as what they were; every order carries a unique id so a retry can't double-send.
- **Mode must match the account.** Interactive Brokers: rehearsal modes refuse a non-`DU` account and `live` refuses a paper one, so a wrong port can't turn a test into real trading or the reverse.
- **Reconciliation on start** against the venue's true positions; **kill switch** (create the `…STOP` file or press Ctrl-C: flatten and stop); **audit log** of every decision and fill in `~/.stratbt/live/`; and if the feed dies with positions open the run says so loudly instead of exiting quietly.

**What is and isn't verified.** The runner is proven identical to the backtester (parity tests). Each adapter is tested end to end against fakes that mimic its SDK and record exactly what would be sent — correct option symbols, open/close intent, limit prices, timeout cancels, partial fills, mode gates — and the safety rules were mutation-tested (breaking each one makes the suite fail). **No adapter has been run against a real brokerage**, so before trusting one: run `dry-run` for a full session, then `paper` (where it exists), then the smallest possible live order. Public's adapter in particular is written against the SDK's documented models and its position-sign and cost-basis conventions for short options should be confirmed with a one-contract test.

**Adding a venue** means implementing two small classes: a `PollingFeed` (`fetch_underlying`, `fetch_options`) and a `LiveBroker` (`_place`, `positions`, optionally `_preflight`). Everything above — modes, caps, quote checks, order tracking, ledger — is inherited. Zerodha Kite (NSE), Tradier, tastytrade and Schwab all have Python SDKs and would follow the same shape.

The MCP server (below) intentionally exposes **no** live-trading tool: an agent can research and rehearse, never trade.

---

## Use it from an AI agent (MCP)

`stratbt mcp` serves the backtester over the [Model Context Protocol](https://modelcontextprotocol.io), so an agent can research strategies the same way you do: pick a market, run a backtest with realistic costs, read the findings, drill into risk and costs, compare runs and export a PDF.

```bash
pip install -e ".[agent]"
stratbt mcp                                           # stdio — desktop agents and IDEs
stratbt mcp --transport streamable-http --port 8765   # remote clients, at http://127.0.0.1:8765/mcp
```

Register it with a client:

```bash
claude mcp add stratbt -- stratbt mcp                 # Claude Code
```
```json
{ "mcpServers": { "stratbt": { "command": "stratbt", "args": ["mcp"] } } }
```

| Tool | What the agent gets |
|---|---|
| `list_strategies` | Strategies with parameters, bounds and risk profile (`unlimited_risk`, `allows_short`) |
| `check_market_data` | Whether tickers have usable data: days, session hours, currency, contract multiplier |
| `run_backtest` | A `run_id`, headline numbers (net/gross PnL, costs, Sharpe with 95% interval, drawdown, win rate), plain-language `observations`, `warnings`, and the config used. Reports progress while running |
| `get_run` | Per-run detail by section: `statistics`, `risk`, `costs`, `behaviour`, `daily`, `trades` |
| `compare_runs` | Side-by-side table, with a note when the runs aren't comparable |
| `export_report` | Writes the PDF tearsheet to the output directory (`--output-dir`, default `~/stratbt-reports`) and returns the path |
| `paper_trade` | Rehearses one session through the live runner with a paper broker, returning the audit trail |

Plus resources (`stratbt://strategies`, `stratbt://runs`, `stratbt://runs/{run_id}`) and two prompts (`evaluate_strategy`, `compare_strategies`) that steer an agent toward an honest evaluation: realistic costs first, compare against the optimistic fill model, read the confidence interval, and say that the options are modelled.

**Safety by construction.** The server has no tool that places orders; `paper_trade` is a local simulation and says so in its output. File writes are limited to PDFs in the output directory (file names are sanitised and cannot escape it), inputs are bounded (at most 4 tickers, 120 synthetic days) and validated, and bad input comes back as a readable error the agent can correct. Every result carries its caveats: modelled option prices, small samples, and the absence of trading costs when none were applied.

---

## Reports

**PDF tearsheet** (`--pdf`, the Export menu, or the `export_report` tool) — seven purposeful pages, drawn with seaborn on a validated, colour-blind-safe palette with direct labels instead of legends where possible: a summary (headline KPIs, run configuration, equity curve and auto-written key observations such as "trading costs turned a gross gain into a net loss" or "only 21 days: the Sharpe interval includes zero"); performance (equity, drawdown, daily PnL); risk (VaR/CVaR, skew, Ulcer, Sharpe interval, distribution, drawdown episodes); behaviour (intraday path, morning vs afternoon, weekday, expiry day); costs (gross → slippage → fees → net, and a break-even sensitivity); the full daily table and statistics; and definitions, assumptions and limitations.

**Metrics** (`stratbt.metrics`) — Sharpe, Sortino, Calmar, drawdown and duration, win rate, profit factor, VaR/CVaR, skewness, kurtosis, Ulcer index, recovery factor, Omega, bootstrap Sharpe interval, cost breakdown and sensitivity.

## Docker

```bash
docker compose up --build            # http://localhost:8501
STRATBT_SOURCE=synthetic docker compose up --build    # offline
```

Downloaded bars are kept in a named volume. The container needs outbound internet for Yahoo Finance.

## Project layout

```
src/stratbt/
├── providers/       DataProvider · ModelPricedProvider · YahooFinanceProvider · SyntheticProvider
├── strategy/        BaseStrategy · rolling/fixed straddles · multi-leg structures · registry
├── engine.py        Backtest loop, DayResult / BacktestResult
├── broker.py        Broker interface · PaperBroker (fills, fees, risk)
├── live.py          LiveRunner · QuoteFeed · ReplayFeed · AuditLog · KillSwitch · reconciliation
├── adapters/        Alpaca · Interactive Brokers · Public: feeds + brokers, safety model, OSI symbols
├── execution.py     ExecutionModel (delay, spread, commissions)
├── risk.py          RiskLimits
├── position.py      Ledger: long/short, average cost, multiplier, fees, slippage
├── metrics.py       Scalar and risk metrics, cost breakdown
├── report/          PDF tearsheet (data → pages → seaborn charts / layout / theme)
├── dashboard/       Streamlit app + Plotly charts (interactive)
├── mcp_server.py    Model Context Protocol server for AI agents
├── palette.py       One colour palette for the dashboard and the PDF
└── cli.py           `stratbt` command (strategies · run · paper · live · app · mcp)
notebooks/           quickstart (offline) · positions & metrics walkthrough
tests/               pytest suite — runs offline
```

## Tests

```bash
uv sync --extra app --extra agent     # or: pip install -e ".[app,agent]" pytest
uv run pytest
```

The whole suite runs offline (Yahoo tests use a fake downloader). The MCP tests talk to the server through the SDK's own client, including real stdio and HTTP subprocesses.

## Limitations

- **Option prices on Yahoo data are modelled**, not recorded quotes: no skew, no term structure, and the vol assumption is yours.
- Fills are against mid prices plus a modelled spread, not an order book; no market impact or queue model.
- **No margin or buying-power model.** Short-premium strategies can lose more than any account would allow; use `RiskLimits` as a stand-in.
- No early exercise, assignment or settlement beyond closing positions at the session end; no overnight positions.
- Expiry calendars are a weekly rule you set, not the exchange's real calendar; holidays are inferred from the data.
- Intraday history from Yahoo is short (weeks), so statistics are noisy; a positive result is a hypothesis, not evidence.

Not investment advice. Backtests are not predictions.

## License

[MIT](LICENSE)

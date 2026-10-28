# src/stratbt/mcp_server.py
"""
Model Context Protocol server: lets an AI agent research options strategies.

An agent can list strategies, check market data, run a backtest with realistic
costs, read the findings, drill into risk and costs, compare runs, export a PDF
tearsheet, and rehearse a strategy in paper mode — all as tool calls.

What this server deliberately cannot do: place orders anywhere. Paper mode is a
local simulation. There is no brokerage connection, so an agent using it can
research and rehearse but never trade.

Run it:   stratbt mcp                       (stdio, for desktop agents / IDEs)
          stratbt mcp --transport streamable-http --port 8765
"""
import functools
import inspect
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
import numpy as np
import pandas as pd
from pydantic import Field

try:
    from mcp.server.mcpserver import Context, MCPServer
    from mcp.server.mcpserver.exceptions import ResourceError, ToolError
    from mcp.types import ToolAnnotations
except ImportError as e:  # pragma: no cover
    raise ImportError("The MCP server needs the 'mcp' package (version 2 or newer):  "
                      "pip install 'strat-backtester[agent]'") from e

from stratbt import __version__
from stratbt.broker import PaperBroker
from stratbt.engine import BacktestEngine, BacktestResult
from stratbt.execution import ExecutionModel
from stratbt.live import AuditLog, LiveRunner, ReplayFeed
from stratbt.providers import DataProvider, SyntheticProvider, YahooFinanceProvider
from stratbt.providers.yahoo import INTERVAL_MAX_DAYS
from stratbt.report import build_pdf_report
from stratbt.report.data import ReportData, build_report_data
from stratbt.risk import RiskLimits
from stratbt.strategy import registry

MAX_TICKERS = 4
MAX_SYNTHETIC_DAYS = 120
MAX_RUNS = 25
MAX_TRADES_RETURNED = 500

INSTRUCTIONS = """\
Research tool for options strategies (straddles, strangles, iron condors, your own).

Typical flow: list_strategies -> check_market_data (optional) -> run_backtest -> read the
`observations` and `warnings` -> get_run for risk / costs / daily detail -> compare_runs -> export_report.

Read these before drawing conclusions:
* Underlying prices from source='yahoo' are real; OPTION PRICES ARE MODELLED (Black-Scholes on trailing
  realised volatility) because Yahoo has no historical option quotes. source='synthetic' is a simulator
  and says nothing about real markets.
* Results are short samples (Yahoo keeps weeks of intraday bars). A positive result is a hypothesis,
  never evidence of an edge; always look at sharpe_ci95 and the costs breakdown.
* Use execution='realistic' (spread + commissions) before believing a profit; 'instant' is optimistic.
* This server cannot trade. paper_trade is a local simulation. Nothing here is investment advice.
"""

Execution = Literal["default", "realistic", "instant"]
Source = Literal["yahoo", "synthetic"]


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------

_ANTICIPATED = (ValueError, KeyError, RuntimeError, ImportError)


def _message(e: Exception) -> str:
    return str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)


def _as(error_type):
    """
    Turn anticipated failures (bad input, unknown ids, data that could not be loaded) into the
    SDK's user-facing error type, so the agent reads *why* and can correct itself. Anything else
    is a genuine crash and stays opaque to the client.
    """
    def deco(fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*a, **k):
                try:
                    return await fn(*a, **k)
                except _ANTICIPATED as e:
                    raise error_type(_message(e)) from e
        else:
            @functools.wraps(fn)
            def wrapper(*a, **k):
                try:
                    return fn(*a, **k)
                except _ANTICIPATED as e:
                    raise error_type(_message(e)) from e
        return wrapper
    return deco


_tool_errors, _resource_errors = _as(ToolError), _as(ResourceError)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _clean(obj: Any) -> Any:
    """Make anything JSON-safe: numpy -> python, NaN/inf -> None, dates -> ISO strings."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _clean(obj.reset_index().to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _clean(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (datetime, pd.Timestamp, date)):
        return obj.isoformat()
    return obj


def _r(v: float | None, d: int = 2) -> float | None:
    return None if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else round(float(v), d)


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

@dataclass
class Run:
    id: str
    result: BacktestResult
    data: ReportData
    request: dict
    created: datetime = field(default_factory=datetime.now)


class RunStore:
    """Bounded in-memory history of runs so later tool calls can refer to `run_id`."""

    def __init__(self, limit: int = MAX_RUNS):
        self._runs: OrderedDict[str, Run] = OrderedDict()
        self._n, self._limit, self._lock = 0, limit, threading.Lock()

    def add(self, result: BacktestResult, data: ReportData, request: dict) -> Run:
        with self._lock:
            self._n += 1
            run = Run(f"run_{self._n}", result, data, request)
            self._runs[run.id] = run
            while len(self._runs) > self._limit:
                self._runs.popitem(last=False)
            return run

    def get(self, run_id: str) -> Run:
        run = self._runs.get(run_id)
        if run is None:
            known = ", ".join(self._runs) or "none yet — call run_backtest first"
            raise ValueError(f"Unknown run_id '{run_id}'. Known runs: {known}")
        return run

    def all(self) -> list[Run]:
        return list(self._runs.values())


class ProviderCache:
    """Building a Yahoo provider downloads data; reuse identical requests."""

    def __init__(self, limit: int = 8):
        self._items: OrderedDict[tuple, DataProvider] = OrderedDict()
        self._limit, self._lock = limit, threading.Lock()

    def get(self, key: tuple, build) -> DataProvider:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        provider = build()
        with self._lock:
            self._items[key] = provider
            while len(self._items) > self._limit:
                self._items.popitem(last=False)
        return provider


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def _make_provider(cache: ProviderCache, source: str, tickers: list[str] | None, interval: str,
                   lookback_days: int, iv_premium: float, expiry_weekday: int, multiplier: float | None,
                   synthetic_days: int, seed: int) -> DataProvider:
    if source == "synthetic":
        if not 1 <= synthetic_days <= MAX_SYNTHETIC_DAYS:
            raise ValueError(f"synthetic_days must be between 1 and {MAX_SYNTHETIC_DAYS}")
        return cache.get(("syn", synthetic_days, seed), lambda: SyntheticProvider(n_days=synthetic_days, seed=seed))
    tickers = [t.strip().upper() for t in (tickers or []) if t and t.strip()]
    if not tickers:
        raise ValueError("source='yahoo' needs `tickers`, e.g. ['SPY'] or ['^NSEI'] or ['RELIANCE.NS'].")
    if len(tickers) > MAX_TICKERS:
        raise ValueError(f"At most {MAX_TICKERS} tickers per run (they must also share trading hours).")
    if interval not in INTERVAL_MAX_DAYS:
        raise ValueError(f"interval must be one of {sorted(INTERVAL_MAX_DAYS)}")
    key = ("yahoo", tuple(tickers), interval, lookback_days, iv_premium, expiry_weekday, multiplier)

    def build():
        try:
            return YahooFinanceProvider(tickers, interval=interval, lookback_days=lookback_days, iv_premium=iv_premium,
                                        expiry_weekday=expiry_weekday, multiplier=multiplier)
        except ValueError:
            raise
        except ImportError as e:
            raise ValueError(str(e)) from e
        except Exception as e:      # network / Yahoo hiccups
            raise RuntimeError(f"Could not load Yahoo Finance data for {tickers}: {e}") from e

    return cache.get(key, build)


def _execution(preset: str, delay_s: int | None, half_spread_bps: float | None,
               commission_per_contract: float | None) -> ExecutionModel:
    base = {"default": ExecutionModel(), "realistic": ExecutionModel.realistic(), "instant": ExecutionModel.instant()}[preset]
    over = {"delay_s": delay_s, "half_spread_bps": half_spread_bps, "commission_per_contract": commission_per_contract}
    over = {k: v for k, v in over.items() if v is not None}
    return ExecutionModel(**{**vars(base), **over}) if over else base


def _strategy_listing() -> list[dict]:
    out = []
    for spec in registry.available().values():
        out.append({
            "key": spec.key, "label": spec.label, "description": spec.description,
            "parameters": [{"name": p.name, "default": p.default, "min": p.min, "max": p.max,
                            "step": p.step, "description": p.label} for p in spec.params],
            "allows_short": spec.cls.allow_short, "unlimited_risk": spec.cls.unlimited_risk,
        })
    return out


def _headline(d: ReportData) -> dict:
    lo, hi = d.sharpe_ci
    return {
        "net_pnl": _r(d.net_pnl), "gross_pnl": _r(d.gross_pnl),
        "costs": {"fees": _r(d.fees), "slippage": _r(d.slippage), "total": _r(d.fees + d.slippage)},
        "sharpe": _r(d.sharpe), "sharpe_ci95": [_r(lo), _r(hi)],
        "max_drawdown": _r(d.max_dd), "max_drawdown_days": d.max_dd_days,
        "day_win_rate": _r(d.win_rate, 3), "profit_factor": _r(d.profit_factor),
        "trades": d.n_trades, "trading_days": d.n_days, "unit": d.unit,
        "best_day": {"date": d.best_day[0], "pnl": _r(d.best_day[1])},
        "worst_day": {"date": d.worst_day[0], "pnl": _r(d.worst_day[1])},
    }


def _per_series(d: ReportData) -> dict:
    keep = {"total": "total_pnl", "mean": "mean_daily_pnl", "std": "std_daily_pnl", "sharpe": "sharpe",
            "sortino": "sortino", "max_dd": "max_drawdown", "win": "day_win_rate", "pf": "profit_factor",
            "var": "var_95", "cvar": "cvar_95"}
    out = {}
    for col in d.stats.columns:
        out[col] = {new: _r(float(d.stats.loc[old, col]), 3) for old, new in keep.items() if old in d.stats.index}
    return out


def _warnings(d: ReportData, request: dict) -> list[str]:
    w = []
    if request["source"] == "yahoo":
        w.append("Option prices are modelled (Black-Scholes on trailing realised vol); only the underlying is real.")
    else:
        w.append("Synthetic data: results say nothing about real markets.")
    if d.n_days < 30:
        w.append(f"Small sample ({d.n_days} days): statistics are noisy; read sharpe_ci95, not the point estimate.")
    ex = request["execution"]
    if ex["delay_s"] == 0:
        w.append("Same-second fills are optimistic (the strategy trades on a price it could not yet have seen).")
    if not (ex["half_spread_bps"] or ex["commission_per_contract"]):
        w.append("No trading costs were applied; rerun with execution='realistic' before trusting a profit.")
    if d.n_rejections:
        w.append(f"{d.n_rejections} orders were rejected by risk limits or missing prices.")
    return w


def _summary_text(d: ReportData, label: str) -> str:
    h = _headline(d)
    return (f"{label} on {', '.join(d.underliers)} over {d.n_days} trading days ({d.period_label}): net PnL "
            f"{h['net_pnl']:,.2f} {d.unit} after {h['costs']['total']:,.2f} of costs; Sharpe {h['sharpe']} "
            f"(95% CI {h['sharpe_ci95'][0]} to {h['sharpe_ci95'][1]}), max drawdown {h['max_drawdown']:,.2f}, "
            f"{h['trades']:,} trades.")


def _config_echo(request: dict, d: ReportData) -> dict:
    return {"strategy": request["strategy"], "parameters": request["params"], "source": request["source"],
            "underliers": d.underliers, "interval": request.get("interval"), "period": d.period_label,
            "execution": request["execution"], "risk_limits": request["risk"],
            "session": request.get("session"), "contract_multiplier": request.get("multiplier")}


def _run_payload(run: Run) -> dict:
    d = run.data
    return {
        "run_id": run.id, "summary": _summary_text(d, d.strategy_name), "headline": _headline(d),
        "observations": d.observations, "per_underlier": _per_series(d),
        "warnings": _warnings(d, run.request), "config": _config_echo(run.request, d),
        "next_steps": [f"get_run(run_id='{run.id}', sections=['risk','costs','daily'])",
                       f"export_report(run_id='{run.id}')", "compare_runs(run_ids=[...]) after more runs"],
    }


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

def build_server(output_dir: str | Path | None = None) -> MCPServer:
    """Create the MCP server. `output_dir` is where export_report writes (default ~/stratbt-reports)."""
    out_dir = Path(output_dir or Path.home() / "stratbt-reports").expanduser()
    store, providers = RunStore(), ProviderCache()
    server = MCPServer("stratbt", title="Options Strategy Backtester", version=__version__, instructions=INSTRUCTIONS,
                       log_level="WARNING")

    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
    read_net = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True)

    # -- discovery ---------------------------------------------------------
    @server.tool(annotations=read_only)
    @_tool_errors
    def list_strategies() -> dict[str, Any]:
        """List the available option strategies with their parameters, bounds and risk profile."""
        return {"strategies": _clean(_strategy_listing())}

    @server.tool(annotations=read_net)
    @_tool_errors
    async def check_market_data(
        tickers: Annotated[list[str], Field(description="Yahoo symbols, e.g. ['SPY'], ['^NSEI','^NSEBANK'], ['RELIANCE.NS']. Max 4, one market.")],
        interval: Annotated[str, Field(description="Bar size: 1m, 2m, 5m, 15m, 30m, 60m or 90m")] = "5m",
        lookback_days: Annotated[int, Field(ge=1, le=729, description="Calendar days of history (capped to what Yahoo keeps for the interval)")] = 30,
    ) -> dict[str, Any]:
        """Check that tickers have usable data before running a backtest; reports trading days, hours, currency and contract defaults."""
        p = await anyio.to_thread.run_sync(lambda: _make_provider(
            providers, "yahoo", tickers, interval, lookback_days, 0.10, 4, None, 21, 7))
        days = p.trading_days
        return _clean({"tickers": p.underliers, "interval": interval, "trading_days": len(days),
                       "first_day": days[0], "last_day": days[-1],
                       "session": f"{p.session_open:%H:%M}-{p.session_close:%H:%M} (exchange local time)",
                       "unit": p.unit, "contract_multiplier": {t: p.multiplier(t) for t in p.underliers},
                       "note": "Underlying bars are real; option prices will be modelled."})

    # -- backtest ----------------------------------------------------------
    @server.tool(annotations=read_net)
    @_tool_errors
    async def run_backtest(
        strategy: Annotated[str, Field(description="Strategy key from list_strategies, e.g. 'iron_condor'")],
        ctx: Context,
        tickers: Annotated[list[str] | None, Field(description="Yahoo symbols (source='yahoo'). Max 4, one market per run.")] = None,
        source: Annotated[Source, Field(description="'yahoo' = real underlying bars + modelled options; 'synthetic' = offline simulator")] = "yahoo",
        interval: Annotated[str, Field(description="Bar size for Yahoo data: 1m, 2m, 5m, 15m, 30m, 60m, 90m")] = "5m",
        lookback_days: Annotated[int, Field(ge=1, le=729, description="Calendar days of Yahoo history (capped per interval)")] = 30,
        params: Annotated[dict[str, float] | None, Field(description="Strategy parameters, e.g. {'width': 3}. See list_strategies.")] = None,
        execution: Annotated[Execution, Field(description="'realistic' = next-second fills + 25bps half-spread + $0.65/contract; 'default' = next-second, no costs; 'instant' = same-second, no costs (optimistic)")] = "realistic",
        delay_s: Annotated[int | None, Field(ge=0, le=30, description="Override fill delay in seconds")] = None,
        half_spread_bps: Annotated[float | None, Field(ge=0, le=1000, description="Override half-spread (bps of price)")] = None,
        commission_per_contract: Annotated[float | None, Field(ge=0, le=100, description="Override commission per contract")] = None,
        max_daily_loss: Annotated[float | None, Field(gt=0, description="Flatten and stop for the day at this loss (currency)")] = None,
        max_open_contracts: Annotated[int | None, Field(gt=0, description="Cap on total contracts held")] = None,
        underliers: Annotated[list[str] | None, Field(description="Subset of the loaded tickers to trade (default all)")] = None,
        start_date: Annotated[str | None, Field(description="ISO date, inclusive")] = None,
        end_date: Annotated[str | None, Field(description="ISO date, inclusive")] = None,
        iv_premium: Annotated[float, Field(ge=-0.5, le=2.0, description="Yahoo only: implied vol = trailing realised vol x (1 + this)")] = 0.10,
        expiry_weekday: Annotated[int, Field(ge=0, le=4, description="Weekly expiry day, 0=Mon..4=Fri")] = 4,
        multiplier: Annotated[float | None, Field(gt=0, description="Contract multiplier (default 100 for USD underlyings, 1 otherwise)")] = None,
        synthetic_days: Annotated[int, Field(ge=1, le=MAX_SYNTHETIC_DAYS, description="source='synthetic': number of trading days")] = 21,
        seed: Annotated[int, Field(description="source='synthetic': RNG seed")] = 7,
    ) -> dict[str, Any]:
        """
        Backtest a strategy second by second and return headline results, plain-language findings and caveats.

        Returns a run_id for follow-up calls (get_run, compare_runs, export_report). Rolling strategies
        (atm_straddle) can take a minute on long windows.
        """
        spec = registry.get(strategy)                     # KeyError -> tool error listing valid keys
        resolved = spec.resolve(**(params or {}))
        provider = await anyio.to_thread.run_sync(lambda: _make_provider(
            providers, source, tickers, interval, lookback_days, iv_premium, expiry_weekday, multiplier,
            synthetic_days, seed))
        ex = _execution(execution, delay_s, half_spread_bps, commission_per_contract)
        risk = RiskLimits(max_daily_loss=max_daily_loss, max_open_contracts=max_open_contracts)
        uls = [u.upper() if source == "yahoo" else u for u in (underliers or provider.underliers)]
        bad = [u for u in uls if u not in provider.underliers]
        if bad:
            raise ValueError(f"Unknown underliers {bad}; available: {provider.underliers}")
        start = date.fromisoformat(start_date) if start_date else None
        end = date.fromisoformat(end_date) if end_date else None
        engine = BacktestEngine(provider, spec.factory(provider, **resolved), execution=ex, risk=risk)

        def work() -> BacktestResult:
            def progress(ul, day, done, total):
                try:
                    anyio.from_thread.run(ctx.report_progress, done, total, f"{ul} {day}")
                except Exception:       # progress is best-effort
                    pass
            return engine.run_all(uls, start, end, progress=progress)

        result = await anyio.to_thread.run_sync(work)
        if not result.day_results:
            raise ValueError("No results: no trading days matched the period/underliers.")
        result.meta.update(strategy=spec.label, strategy_key=spec.key, params=resolved,
                           allow_short=spec.cls.allow_short, unlimited_risk=spec.cls.unlimited_risk)
        data = build_report_data(result)
        request = {"strategy": spec.key, "params": resolved, "source": source, "interval": interval if source == "yahoo" else None,
                   "execution": dict(vars(ex)), "risk": dict(vars(risk)),
                   "session": f"{provider.session_open:%H:%M}-{provider.session_close:%H:%M}",
                   "multiplier": {u: provider.multiplier(u) for u in uls}}
        return _clean(_run_payload(store.add(result, data, request)))

    # -- inspect ------------------------------------------------------------
    @server.tool(annotations=read_only)
    @_tool_errors
    def get_run(
        run_id: Annotated[str, Field(description="Id returned by run_backtest")],
        sections: Annotated[list[Literal["summary", "statistics", "risk", "costs", "behaviour", "daily", "trades"]], Field(
            description="What to return. Default: summary")] = ["summary"],  # noqa: B006
        max_trades: Annotated[int, Field(ge=1, le=MAX_TRADES_RETURNED, description="Row limit for the trades section")] = 100,
    ) -> dict[str, Any]:
        """Return detail for a previous run: statistics, risk, costs, behaviour patterns, daily PnL or the trade list."""
        run = store.get(run_id)
        d, out = run.data, {"run_id": run.id}
        for sec in sections:
            if sec == "summary":
                out["summary"] = _run_payload(run)
            elif sec == "statistics":
                out["statistics"] = _per_series(d)
            elif sec == "risk":
                lo, hi = d.sharpe_ci
                out["risk"] = {
                    "metrics": {c: {"var_95": _r(d.stats.loc["var", c]), "cvar_95": _r(d.stats.loc["cvar", c]),
                                    "skew": _r(d.stats.loc["skew", c]), "excess_kurtosis": _r(d.stats.loc["kurt", c]),
                                    "ulcer_index": _r(d.stats.loc["ulcer", c]), "omega": _r(d.stats.loc["omega", c])}
                                for c in d.stats.columns},
                    "sharpe_ci95": _clean(d.sharpe_ci_df.round(3)),
                    "drawdown_episodes": [dict(zip(["series", "trough", "depth", "recovered"], r)) for r in d.drawdown_rows],
                    "note": d.risk_note}
            elif sec == "costs":
                out["costs"] = {"breakdown": _clean(d.cost_breakdown.round(2)),
                                "break_even_extra_bps": _r(d.breakeven_bps, 1),
                                "sensitivity": _clean(d.cost_sens[["cost_bps", "net_pnl"]]),
                                "activity": dict(d.activity_rows),
                                "meaning": "gross = mid-price fills with no fees; slippage = spread paid; net = gross - slippage - fees"}
            elif sec == "behaviour":
                out["behaviour"] = {"morning_vs_afternoon": _clean(d.am_pm.round(2)),
                                    "by_weekday": _clean(d.weekday.round(2)), "expiry_vs_other_days": _clean(d.expiry.round(2))}
            elif sec == "daily":
                frame = d.daily.copy()
                if len(d.underliers) > 1:
                    frame["Combined"] = d.combined
                frame["Cumulative"] = d.cum_combined
                out["daily"] = _clean(frame.round(2).rename_axis("date"))
            elif sec == "trades":
                t = run.result.trades_df
                out["trades"] = {"total": len(t), "returned": min(len(t), max_trades),
                                 "rows": _clean(t.head(max_trades).drop(columns=["instrument"]))}
        return out

    @server.tool(annotations=read_only)
    @_tool_errors
    def compare_runs(run_ids: Annotated[list[str], Field(min_length=2, description="Two or more run ids")]) -> dict[str, Any]:
        """Compare runs side by side (net PnL, risk, costs, configuration)."""
        runs = [store.get(r) for r in run_ids]
        rows = []
        for r in runs:
            h = _headline(r.data)
            rows.append({"run_id": r.id, "strategy": r.data.strategy_name, "parameters": r.request["params"],
                         "underliers": r.data.underliers, "net_pnl": h["net_pnl"], "costs": h["costs"]["total"],
                         "sharpe": h["sharpe"], "sharpe_ci95": h["sharpe_ci95"], "max_drawdown": h["max_drawdown"],
                         "day_win_rate": h["day_win_rate"], "trades": h["trades"],
                         "fill_delay_s": r.request["execution"]["delay_s"], "unit": h["unit"]})
        notes = []
        if len({(r.request["source"], tuple(r.data.underliers), r.data.period_label) for r in runs}) > 1:
            notes.append("These runs use different data (source, underliers or period): compare with care.")
        if len({r.data.unit for r in runs}) > 1:
            notes.append("PnL units differ between runs.")
        best = max(rows, key=lambda x: x["net_pnl"] if x["net_pnl"] is not None else -1e18)
        return _clean({"runs": rows, "highest_net_pnl": best["run_id"], "notes": notes,
                       "caution": "Highest PnL is not the best strategy: check sharpe_ci95, drawdown and sample size."})

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False))
    @_tool_errors
    def export_report(
        run_id: Annotated[str, Field(description="Id returned by run_backtest")],
        filename: Annotated[str | None, Field(description="Optional file name (a plain name, not a path); default report_<run_id>.pdf")] = None,
    ) -> dict[str, Any]:
        """Write a 7-page PDF tearsheet for a run into the server's output directory and return its path."""
        run = store.get(run_id)
        name = Path(filename or f"report_{run.id}.pdf").name
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name) or f"report_{run.id}.pdf"
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        pdf = build_pdf_report(run.result)
        path.write_bytes(pdf)
        return {"path": str(path), "bytes": len(pdf), "pages": 7, "run_id": run.id}

    # -- paper trading ------------------------------------------------------
    @server.tool(annotations=read_net)
    @_tool_errors
    async def paper_trade(
        strategy: Annotated[str, Field(description="Strategy key from list_strategies")],
        ticker: Annotated[str, Field(description="One underlier (a Yahoo symbol, or SPY/QQQ for synthetic)")] = "SPY",
        source: Annotated[Source, Field(description="'yahoo' (latest real session) or 'synthetic'")] = "yahoo",
        day: Annotated[str, Field(description="'last' or an ISO date within the loaded window")] = "last",
        interval: Annotated[str, Field(description="Yahoo bar size")] = "5m",
        params: Annotated[dict[str, float] | None, Field(description="Strategy parameters")] = None,
        execution: Annotated[Execution, Field(description="Fill model")] = "realistic",
        max_daily_loss: Annotated[float | None, Field(gt=0)] = None,
        max_open_contracts: Annotated[int | None, Field(gt=0)] = None,
        audit_events: Annotated[int, Field(ge=0, le=200, description="How many audit-log events to return")] = 30,
    ) -> dict[str, Any]:
        """
        Rehearse a strategy for one session through the live-trading runner with a PAPER broker.

        This replays a recorded session as if it were arriving live and applies reconciliation,
        risk limits and an audit log. It is a simulation: no orders are sent anywhere.
        """
        spec = registry.get(strategy)
        resolved = spec.resolve(**(params or {}))
        provider = await anyio.to_thread.run_sync(lambda: _make_provider(
            providers, source, [ticker], interval, 30, 0.10, 4, None, 21, 7))
        sym = ticker.upper() if source == "yahoo" else ticker
        if sym not in provider.underliers:
            raise ValueError(f"Unknown ticker '{ticker}'; available: {provider.underliers}")
        d = provider.trading_days[-1] if day == "last" else date.fromisoformat(day)
        ex = _execution(execution, None, None, None)

        def work():
            feed = ReplayFeed(provider, d, sym)
            strat = spec.factory(provider, **resolved)(sym, feed.expiry, feed.strikes)
            broker = PaperBroker(provider.multiplier(sym), ex, RiskLimits(max_daily_loss=max_daily_loss,
                                 max_open_contracts=max_open_contracts), allow_short=strat.allow_short)
            return LiveRunner(strat, feed, broker, ex, audit=AuditLog()).run(), broker

        summary, broker = await anyio.to_thread.run_sync(work)
        return _clean({
            "mode": "paper (simulation: no orders were sent anywhere)", "strategy": spec.label, "ticker": sym, "day": d,
            "net_pnl": _r(summary.net_pnl), "unit": provider.unit, "fills": summary.n_fills,
            "rejections": summary.n_rejections, "halted_by_risk_limit": summary.halted,
            "positions_at_end": len(broker.holdings),
            "audit_log": [{k: v for k, v in e.items() if k != "at"} for e in summary.audit.events[:audit_events]],
            "audit_log_total_events": len(summary.audit.events)})

    # -- resources ----------------------------------------------------------
    @server.resource("stratbt://strategies", name="strategies", mime_type="application/json",
                     description="All available strategies with parameters and risk profile.")
    @_resource_errors
    def strategies_resource() -> dict[str, Any]:
        return {"strategies": _clean(_strategy_listing())}

    @server.resource("stratbt://runs", name="runs", mime_type="application/json",
                     description="Runs completed in this session.")
    @_resource_errors
    def runs_resource() -> dict[str, Any]:
        return _clean({"runs": [{"run_id": r.id, "summary": _summary_text(r.data, r.data.strategy_name)} for r in store.all()]})

    @server.resource("stratbt://runs/{run_id}", name="run", mime_type="application/json",
                     description="Headline results, findings and caveats for one run.")
    @_resource_errors
    def run_resource(run_id: str) -> dict[str, Any]:
        return _clean(_run_payload(store.get(run_id)))

    # -- prompts ------------------------------------------------------------
    @server.prompt(title="Evaluate a strategy honestly")
    def evaluate_strategy(strategy: str = "iron_condor", ticker: str = "SPY") -> str:
        return (
            f"Evaluate the '{strategy}' options strategy on {ticker} using the stratbt tools.\n"
            "1. Call list_strategies and confirm the parameters.\n"
            f"2. Call check_market_data for ['{ticker}'] and note how many days are available.\n"
            "3. Run the backtest with execution='realistic'. Then run it again with execution='instant'. "
            "The gap between the two is the cost of realism.\n"
            "4. Call get_run with sections ['risk','costs'] on the realistic run.\n"
            "5. Report: net PnL after costs, the Sharpe confidence interval, max drawdown, how much of the gross profit "
            "costs consumed, and whether the sample size supports any conclusion.\n"
            "State clearly that option prices are modelled and that a short backtest is a hypothesis, not evidence. "
            "Do not give investment advice.")

    @server.prompt(title="Compare strategies on one underlying")
    def compare_strategies(ticker: str = "SPY", strategies: str = "short_strangle,iron_condor,long_strangle") -> str:
        return (
            f"Compare these strategies on {ticker}: {strategies}.\n"
            "Run each with execution='realistic' and identical data settings, then call compare_runs. "
            "Rank by risk-adjusted outcome, not raw PnL: use the Sharpe interval, max drawdown, and note which have "
            "unlimited risk. Flag any difference that is smaller than the noise implied by the sample size.")

    return server


def main(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765, output_dir: str | None = None) -> None:
    server = build_server(output_dir)
    if transport == "stdio":
        server.run("stdio")
    else:
        server.run(transport, host=host, port=port)


if __name__ == "__main__":
    main()

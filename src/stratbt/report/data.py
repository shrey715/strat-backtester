# src/stratbt/report/data.py
"""
Fetch stage of the report pipeline: every number, table and sentence the
pages need is computed here, once, from a BacktestResult. Pages only lay
this out; they never touch pandas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

import stratbt.metrics as metrics
from stratbt import __version__
from stratbt.engine import BacktestResult
from stratbt.report import theme as T

SMALL_SAMPLE_DAYS = 30


@dataclass
class ReportData:
    # ── identity ──────────────────────────────────────────────────────────
    strategy_name: str
    period_label: str
    generated_at: str
    version: str
    unit: str
    underliers: list[str]
    n_days: int
    n_runs: int
    provider: str

    # ── series ────────────────────────────────────────────────────────────
    daily: pd.DataFrame               # date x underlier, NaN -> 0
    combined: pd.Series               # daily PnL summed across underliers
    cum: pd.DataFrame                 # cumulative per underlier
    cum_combined: pd.Series
    drawdown: pd.Series               # combined underwater curve (<= 0)

    # ── headline numbers (combined) ───────────────────────────────────────
    net_pnl: float
    gross_pnl: float
    fees: float
    slippage: float
    sharpe: float
    sharpe_ci: tuple[float, float]
    max_dd: float
    max_dd_days: int
    win_rate: float
    profit_factor: float
    n_trades: int
    best_day: tuple[object, float]
    worst_day: tuple[object, float]

    # ── tables (already display-ready strings unless noted) ───────────────
    config_rows: list[tuple[str, str]]
    stats_cols: list[str]             # columns shown in the statistics tables
    stats_rows: list[list[str]]       # [label, v1, v2, ...]
    risk_rows: list[list[str]]
    sharpe_ci_df: pd.DataFrame        # numeric: sharpe, ci_low, ci_high
    drawdown_rows: list[list[str]]
    daily_rows: list[list[str]]       # date, per-underlier, combined, cumulative
    daily_headers: list[str]
    cost_rows: list[list[str]]
    cost_breakdown: pd.DataFrame      # numeric
    cost_sens: pd.DataFrame           # numeric
    breakeven_bps: float
    activity_rows: list[tuple[str, str]]

    # ── behaviour (numeric) ───────────────────────────────────────────────
    intraday: pd.DataFrame            # time-of-day x underlier avg MTM (1-min)
    am_pm: pd.DataFrame               # underlier x [morning, afternoon] totals
    weekday: pd.DataFrame             # weekday, avg_pnl, count (combined)
    expiry: pd.DataFrame              # day_type, avg_pnl, count (combined)
    total_rolls: int
    avg_rolls_per_day: float

    # ── prose ─────────────────────────────────────────────────────────────
    observations: list[str]
    captions: dict[str, str] = field(default_factory=dict)
    risk_note: str = ""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _stats_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily-PnL-only metrics for each column of `daily` (metrics x columns, numeric)."""
    view = SimpleNamespace(daily_pnl=daily)
    out = {
        "total": metrics.total_pnl(view), "mean": metrics.mean_daily_pnl(view),
        "std": metrics.std_daily_pnl(view),
        "best": daily.max(), "worst": daily.min(),
        "sharpe": metrics.sharpe_ratio(view), "sortino": metrics.sortino_ratio(view),
        "calmar": metrics.calmar_ratio(view), "max_dd": metrics.max_drawdown(view),
        "dd_days": metrics.max_drawdown_duration(view), "win": metrics.win_rate(view),
        "pf": metrics.profit_factor(view),
        "var": metrics.value_at_risk(view), "cvar": metrics.conditional_value_at_risk(view),
        "skew": metrics.pnl_skewness(view), "kurt": metrics.pnl_kurtosis(view),
        "ulcer": metrics.ulcer_index(view), "recovery": metrics.recovery_factor(view),
        "omega": metrics.omega_ratio(view),
    }
    return pd.DataFrame(out).T


def _safe(fn, default):
    try:
        v = fn()
        return default if v is None else v
    except Exception:
        return default


def _config_rows(result: BacktestResult, label: str, unit: str, n_days: int,
                 underliers: list[str]) -> list[tuple[str, str]]:
    m = result.meta
    rows = [("Strategy", label)]
    params = m.get("params") or {}
    shown = {k: v for k, v in params.items() if v not in (0, 0.0, None) or k in ("width", "wing")}
    if shown:
        rows.append(("Parameters", ", ".join(f"{k} = {v:g}" if isinstance(v, (int, float)) else f"{k} = {v}"
                                              for k, v in shown.items())))
    rows.append(("Market data", f"{m.get('provider', 'n/a')} · {', '.join(underliers)}"))
    ex = m.get("execution")
    if ex:
        fill = "same-second fills" if ex["delay_s"] == 0 else f"fills {ex['delay_s']}s after decision"
        costs = []
        if ex["half_spread_bps"]:
            costs.append(f"{ex['half_spread_bps']:g} bps half-spread")
        if ex["commission_per_contract"]:
            costs.append(f"{ex['commission_per_contract']:g}/contract")
        if ex["commission_bps"]:
            costs.append(f"{ex['commission_bps']:g} bps commission")
        rows.append(("Execution", fill + (", " + ", ".join(costs) if costs else ", no trading costs")))
    rk = m.get("risk")
    if rk is not None:
        parts = []
        if rk["max_daily_loss"]:
            parts.append(f"daily loss ≤ {T.fmoney(rk['max_daily_loss'], unit)}")
        if rk["max_open_contracts"]:
            parts.append(f"≤ {rk['max_open_contracts']} open contracts")
        if rk["max_order_qty"]:
            parts.append(f"order size ≤ {rk['max_order_qty']}")
        rows.append(("Risk limits", ", ".join(parts) if parts else "none"))
    rows.append(("Sample", f"{n_days} trading days · PnL in {unit}"))
    return rows


def _observations(d: dict) -> list[str]:
    """Plain-language findings, most important first. Pure function of the numbers in `d`."""
    unit, obs = d["unit"], []
    m = lambda v, **kw: T.fmoney(v, unit, **kw)
    net, gross = d["net_pnl"], d["gross_pnl"]

    verb = "gained" if net >= 0 else "lost"
    line = (f"The strategy {verb} {m(abs(net))} net over {d['n_days']} trading days "
            f"({m(net / d['n_days'], d=1, signed=True)} per day on average).")
    obs.append(line)

    cost = d["fees"] + d["slippage"]
    if cost > 0:
        if gross > 0 and net < 0:
            obs.append(f"Trading costs ({m(cost)}) turned a gross gain of {m(gross)} into a net loss.")
        elif gross > 0:
            obs.append(f"Trading costs absorbed {cost / gross:.0%} of the gross gain "
                       f"({m(cost)} of {m(gross)}).")
        else:
            obs.append(f"Trading costs added {m(cost)} to a gross loss of {m(abs(gross))}.")

    lo, hi = d["sharpe_ci"]
    if d["n_days"] < SMALL_SAMPLE_DAYS:
        tail = ""
        if not (math.isnan(lo) or math.isnan(hi)):
            tail = (f" The 95% bootstrap interval on Sharpe is {T.fnum(lo, 1)} to {T.fnum(hi, 1)}"
                    + (" — it includes zero, so skill is not distinguishable from luck."
                       if lo <= 0 <= hi else "."))
        obs.append(f"Small sample: only {d['n_days']} days of data.{tail}")
    elif not (math.isnan(lo) or math.isnan(hi)) and lo <= 0 <= hi:
        obs.append("The 95% bootstrap interval on Sharpe includes zero: the edge is not statistically established.")

    wd, wv = d["worst_day"]
    if wv < 0 and net != 0:
        share = abs(wv) / abs(net) if net < 0 else abs(wv) / max(d["gross_wins"], 1e-9)
        what = "the total net loss" if net < 0 else "the combined profit of all winning days"
        obs.append(f"Worst day: {wd:%b %d} at {m(wv)}"
                   + (f", equal to {share:.0%} of {what}." if share >= 0.25 else "."))

    if d["max_dd"] > 0:
        obs.append(f"Maximum drawdown was {m(d['max_dd'])}"
                   + (f", lasting {d['max_dd_days']} of {d['n_days']} days." if d["max_dd_days"] else "."))

    if d["short_premium"]:
        obs.append("This structure has open-ended loss potential: observed drawdowns understate what a "
                   "large gap or volatility spike could do.")

    if d["rejections"]:
        obs.append(f"The broker rejected {d['rejections']} orders (risk limits or missing prices); "
                   "see the run configuration.")
    return obs[:6]


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def build_report_data(result: BacktestResult, strategy_name: str = "", period_label: str = "") -> ReportData:
    unit = result.unit
    label = result.meta.get("strategy") or strategy_name or "Backtest"
    daily = result.daily_pnl.fillna(0.0)
    underliers = list(daily.columns)
    n_days = len(daily)
    combined = daily.sum(axis=1) if len(underliers) else pd.Series(dtype=float)
    cum = daily.cumsum()
    cum_combined = combined.cumsum()
    drawdown = (cum_combined - cum_combined.cummax()).clip(upper=0.0)

    if not period_label and n_days:
        a, b = daily.index.min(), daily.index.max()
        period_label = f"{a:%b %d} – {b:%b %d, %Y}"

    multi = len(underliers) > 1
    stats = _stats_frame(daily) if underliers else pd.DataFrame()
    if multi:
        stats["Combined"] = _stats_frame(pd.DataFrame({"Combined": combined}))["Combined"]
    stats_cols = list(stats.columns)
    comb_key = "Combined" if multi else (underliers[0] if underliers else None)

    def S(row: str, col: str | None = None) -> float:
        return float(stats.loc[row, col or comb_key]) if comb_key else float("nan")

    # ── costs ─────────────────────────────────────────────────────────────
    breakdown = metrics.cost_breakdown(result)
    net = float(combined.sum()) if n_days else 0.0
    fees = float(breakdown.loc["Total", "fees"]) if "Total" in breakdown.index else 0.0
    slippage = float(breakdown.loc["Total", "slippage"]) if "Total" in breakdown.index else 0.0
    gross = net + fees + slippage

    # ── sharpe CI (combined) ──────────────────────────────────────────────
    ci_view = SimpleNamespace(daily_pnl=daily if not multi else pd.concat([daily, combined.rename("Combined")], axis=1))
    ci_df = _safe(lambda: metrics.sharpe_confidence_interval(ci_view), pd.DataFrame(columns=["sharpe", "ci_low", "ci_high"]))
    ci = (float(ci_df.loc[comb_key, "ci_low"]), float(ci_df.loc[comb_key, "ci_high"])) if comb_key in ci_df.index else (float("nan"),) * 2

    # ── display tables ────────────────────────────────────────────────────
    summary = _safe(lambda: metrics.summary_table(result), pd.DataFrame())
    hold_secs = {u: float(summary.loc[u, "avg_holding_secs"]) for u in summary.index} if not summary.empty else {}
    trades_by = result.trades_df.groupby("underlier").size().to_dict() if not result.trades_df.empty else {}
    n_trades = int(sum(trades_by.values()))
    if multi:
        trades_by["Combined"] = n_trades
        hold_secs["Combined"] = float(np.mean(list(hold_secs.values()))) if hold_secs else float("nan")

    fm = lambda v, **kw: T.fmoney(v, unit, **kw)
    stat_def = [
        ("Total PnL",            "total",   lambda v: fm(v, d=0)),
        ("Mean daily PnL",       "mean",    lambda v: fm(v, d=1)),
        ("Std. dev. of daily PnL", "std",   lambda v: fm(v, d=1)),
        ("Best day",             "best",    lambda v: fm(v, d=0)),
        ("Worst day",            "worst",   lambda v: fm(v, d=0)),
        ("Sharpe (annualised)",  "sharpe",  lambda v: T.fnum(v, 2)),
        ("Sortino (annualised)", "sortino", lambda v: T.fnum(v, 2)),
        ("Calmar",               "calmar",  lambda v: T.fnum(v, 2)),
        ("Max drawdown",         "max_dd",  lambda v: fm(v, d=0)),
        ("Longest drawdown",     "dd_days", lambda v: f"{int(v)} days" if not math.isnan(v) else T.DASH),
        ("Day win rate",         "win",     lambda v: T.fpct(v)),
        ("Profit factor",        "pf",      lambda v: T.fnum(v, 2)),
    ]
    stats_rows = [[lab] + [f(float(stats.loc[key, c])) for c in stats_cols] for lab, key, f in stat_def]
    stats_rows.append(["Avg holding time"] + [T.fduration(hold_secs.get(c, float("nan"))) for c in stats_cols])
    stats_rows.append(["Trades"] + [f"{trades_by.get(c, 0):,}" for c in stats_cols])

    risk_def = [
        ("VaR 95% (1-day loss)", "var",  lambda v: fm(v, d=0)),
        ("CVaR 95% (avg tail loss)", "cvar", lambda v: fm(v, d=0)),
        ("Skewness",             "skew", lambda v: T.fnum(v, 2)),
        ("Excess kurtosis",      "kurt", lambda v: T.fnum(v, 2)),
        ("Ulcer index",          "ulcer", lambda v: T.fnum(v, 1)),
        ("Recovery factor",      "recovery", lambda v: T.fnum(v, 2)),
        ("Omega ratio",          "omega", lambda v: T.fnum(v, 2)),
    ]
    risk_rows = [[lab] + [f(float(stats.loc[key, c])) for c in stats_cols] for lab, key, f in risk_def]

    dd_view = SimpleNamespace(daily_pnl=daily if not multi else pd.DataFrame({"Combined": combined}))
    dd_tab = _safe(lambda: metrics.drawdown_table(dd_view, top_n=5), pd.DataFrame())
    if not dd_tab.empty:      # drop rounding-dust episodes (< 1% of the deepest one)
        dd_tab = dd_tab[dd_tab["depth"] >= 0.01 * dd_tab["depth"].max()]
    drawdown_rows = [[str(r["underlier"]), str(r["trough_date"]), fm(-r["depth"], d=0),
                      str(r["recovery_date"]) if r["recovery_date"] is not None else "ongoing"]
                     for _, r in dd_tab.iterrows()] if not dd_tab.empty else []

    cum_run = cum_combined
    daily_rows = []
    for dt in daily.index:
        row = [f"{dt:%a %b %d}"] + [fm(daily.loc[dt, u], d=0, signed=True) for u in underliers]
        if multi:
            row.append(fm(combined.loc[dt], d=0, signed=True))
        row.append(fm(cum_run.loc[dt], d=0, signed=True))
        daily_rows.append(row)
    daily_headers = ["Date"] + underliers + (["Combined"] if multi else []) + ["Cumulative"]

    be = _safe(lambda: metrics.breakeven_bps(result), 0.0)
    max_bps = 20.0 if be <= 0 else float(min(max(20.0, be * 1.25), 2000.0))
    cost_sens = _safe(lambda: metrics.cost_sensitivity(result, [float(x) for x in np.round(np.linspace(0, max_bps, 9), 1)]),
                      pd.DataFrame())
    cost_rows = []
    for u, r in breakdown.iterrows():
        cost_rows.append([str(u), fm(r["gross"], d=0), fm(-r["slippage"], d=0), fm(-r["fees"], d=0), fm(r["net"], d=0)])

    notional = _safe(lambda: float(result.notional_df["notional"].sum()), 0.0)
    rpd = metrics.rolls_per_day(result) if result.day_results else pd.DataFrame({"rolls": [0]})
    total_rolls = int(rpd["rolls"].sum())
    activity_rows = [
        ("Trades", f"{n_trades:,}"),
        ("Trades per day", T.fnum(n_trades / n_days, 1) if n_days else T.DASH),
        ("Total notional traded", fm(notional, d=0)),
        ("Fees per trade", fm(fees / n_trades, d=2) if n_trades else T.DASH),
        ("Slippage per trade", fm(slippage / n_trades, d=2) if n_trades else T.DASH),
        ("Rolls per day (avg)", T.fnum(rpd["rolls"].mean(), 0) if total_rolls else "none (no rolling)"),
        ("Break-even extra cost", (f"{be:.1f} bps of premium" if be > 0 else "n/a — not profitable before extra costs")),
    ]

    # ── behaviour ─────────────────────────────────────────────────────────
    intraday = _safe(lambda: metrics.avg_intraday_mtm(result), pd.DataFrame())
    if not intraday.empty:
        t = pd.to_datetime(intraday.index.astype(str), format="%H:%M:%S")
        intraday = intraday.copy()
        intraday.index = t
        intraday = intraday.resample("1min").last()
    ma = _safe(lambda: metrics.morning_afternoon_split(result.day_results), pd.DataFrame())
    am_pm = (ma.groupby("underlier")[["morning_pnl", "afternoon_pnl"]].sum() if not ma.empty
             else pd.DataFrame(columns=["morning_pnl", "afternoon_pnl"]))
    wk = _safe(lambda: metrics.pnl_by_weekday(result), pd.DataFrame())
    if not wk.empty:
        wk = wk.assign(w=wk["avg_pnl"] * wk["count"]).groupby("weekday", sort=False).agg(w=("w", "sum"), count=("count", "sum"))
        wk["avg_pnl"] = wk["w"] / wk["count"]
        wk = wk.reset_index()[["weekday", "avg_pnl", "count"]]
    ex = _safe(lambda: metrics.expiry_day_pnl(result), pd.DataFrame())
    if not ex.empty:
        ex = ex.assign(w=ex["avg_pnl"] * ex["count"]).groupby("day_type", sort=False).agg(w=("w", "sum"), count=("count", "sum"))
        ex["avg_pnl"] = ex["w"] / ex["count"]
        ex = ex.reset_index()[["day_type", "avg_pnl", "count"]]

    rejections = sum(len(dr.rejections) for dr in result.day_results)
    best = (combined.idxmax(), float(combined.max())) if n_days else (None, 0.0)
    worst = (combined.idxmin(), float(combined.min())) if n_days else (None, 0.0)
    short_premium = bool(result.meta.get("unlimited_risk"))

    facts = dict(unit=unit, n_days=n_days, net_pnl=net, gross_pnl=gross, fees=fees, slippage=slippage,
                 sharpe_ci=ci, worst_day=worst, max_dd=S("max_dd") if comb_key else 0.0,
                 max_dd_days=int(S("dd_days")) if comb_key and not math.isnan(S("dd_days")) else 0,
                 gross_wins=float(combined[combined > 0].sum()) if n_days else 0.0,
                 short_premium=short_premium, rejections=rejections)
    observations = _observations(facts) if n_days else ["No results were produced."]

    captions = {
        "equity": (f"Net PnL {T.fmoney(net, unit, signed=True)} after {n_days} days"
                   + (f"; costs took {T.fmoney(fees + slippage, unit)}." if fees + slippage > 0 else ".")),
        "drawdown": (f"Deepest drop {T.fmoney(S('max_dd'), unit)} below the running peak."
                     if comb_key and S("max_dd") > 0 else "No drawdown."),
        "daily": (f"{int((combined > 0).sum())} winning days of {n_days}; "
                  f"best {T.fmoney(best[1], unit, signed=True)}, worst {T.fmoney(worst[1], unit, signed=True)}."
                  if n_days else ""),
    }
    risk_note = (f"Daily PnL in {unit}, zero risk-free rate, {n_days} observations. "
                 "VaR/CVaR are historical (empirical) losses, positive = loss.")

    return ReportData(
        strategy_name=label, period_label=period_label, generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        version=__version__, unit=unit, underliers=underliers, n_days=n_days, n_runs=len(result.day_results),
        provider=str(result.meta.get("provider", "")),
        daily=daily, combined=combined, cum=cum, cum_combined=cum_combined, drawdown=drawdown,
        net_pnl=net, gross_pnl=gross, fees=fees, slippage=slippage,
        sharpe=S("sharpe") if comb_key else float("nan"), sharpe_ci=ci,
        max_dd=S("max_dd") if comb_key else 0.0,
        max_dd_days=facts["max_dd_days"], win_rate=S("win") if comb_key else float("nan"),
        profit_factor=S("pf") if comb_key else float("nan"), n_trades=n_trades,
        best_day=best, worst_day=worst,
        config_rows=_config_rows(result, label, unit, n_days, underliers),
        stats_cols=stats_cols, stats_rows=stats_rows, risk_rows=risk_rows,
        sharpe_ci_df=ci_df, drawdown_rows=drawdown_rows, daily_rows=daily_rows, daily_headers=daily_headers,
        cost_rows=cost_rows, cost_breakdown=breakdown, cost_sens=cost_sens, breakeven_bps=be,
        activity_rows=activity_rows,
        intraday=intraday, am_pm=am_pm, weekday=wk, expiry=ex,
        total_rolls=total_rolls, avg_rolls_per_day=float(rpd["rolls"].mean()),
        observations=observations, captions=captions, risk_note=risk_note,
    )

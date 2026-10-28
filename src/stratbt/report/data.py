# streamlit_app/report/data.py
"""
Data stage of the report pipeline: pulls every number, table, and series the
PDF needs out of a BacktestResult exactly once, before any rendering happens.
Nothing in this module touches matplotlib — it can be unit tested with plain
pandas assertions.

Fields added vs the original flat report.py:
  sharpe_ci         — bootstrap 95% CI on annualised Sharpe (statistical validity)
  morning_afternoon — AM vs PM PnL split per underlier per day
  pnl_by_weekday    — average daily PnL by day-of-week per underlier
  expiry_day        — expiry-day vs non-expiry-day average PnL
  round_trips       — per-leg (CE/PE) round-trip statistics
  notional_per_day  — daily total notional traded per underlier
  var_series        — VaR95 per underlier (for histogram annotation)
  cvar_series       — CVaR95 per underlier (for histogram annotation)
"""
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from stratbt.engine import BacktestResult
import stratbt.metrics as metrics


@dataclass
class ReportData:
    # ── metadata ──────────────────────────────────────────────────────────
    strategy_name:        str
    period_label:         str
    generated_at:         str

    # ── headline scalars ──────────────────────────────────────────────────
    daily_pnl:            pd.DataFrame  # per-underlier, NaN filled with 0
    combined_pnl:         pd.Series     # summed across underliers
    n_trades:             int
    n_day_underlier_runs: int
    avg_rolls_per_day:    float
    total_rolls:          int
    win_rate_label:       str
    breakeven_bps:        float

    # ── existing tables ───────────────────────────────────────────────────
    summary_table:        pd.DataFrame  # underlier x 14 metrics
    risk_table:           pd.DataFrame  # underlier x 7 risk metrics
    drawdown_table:       pd.DataFrame  # top-N drawdown episodes
    cost_sensitivity:     pd.DataFrame  # net PnL at various cost assumptions
    rolling_sharpe:       pd.DataFrame  # 5-day rolling Sharpe per underlier
    rolls_per_day:        pd.DataFrame  # daily roll count per underlier

    # ── new: statistical validity ─────────────────────────────────────────
    sharpe_ci:            pd.DataFrame  # sharpe, ci_low, ci_high per underlier
    var_series:           pd.Series     # VaR95 loss (positive = loss) per underlier
    cvar_series:          pd.Series     # CVaR95 loss per underlier

    # ── new: pattern analysis ─────────────────────────────────────────────
    morning_afternoon:    pd.DataFrame  # date, underlier, morning_pnl, afternoon_pnl
    pnl_by_weekday:       pd.DataFrame  # weekday, underlier, avg_pnl, count
    expiry_day:           pd.DataFrame  # underlier, day_type, avg_pnl, count
    round_trips:          pd.DataFrame  # per-leg CE/PE round-trip stats (may be empty)
    notional_per_day:     pd.DataFrame  # date, underlier, notional
    unit:                 str = "pts"   # PnL unit label ("pts", "USD", ...)


def build_report_data(
    result: BacktestResult,
    strategy_name: str,
    period_label: str,
) -> ReportData:
    daily = result.daily_pnl.fillna(0)
    rpd   = metrics.rolls_per_day(result)
    wr    = metrics.win_rate(result)

    return ReportData(
        # metadata
        strategy_name        = strategy_name,
        period_label         = period_label,
        generated_at         = datetime.now().strftime("%Y-%m-%d %H:%M"),
        # scalars
        daily_pnl            = daily,
        combined_pnl         = daily.sum(axis=1),
        n_trades             = len(result.trades_df),
        n_day_underlier_runs = len(result.day_results),
        avg_rolls_per_day    = rpd["rolls"].mean(),
        total_rolls          = int(rpd["rolls"].sum()),
        win_rate_label       = " / ".join(f"{wr[ul]:.0%}" for ul in wr.index),
        breakeven_bps        = metrics.breakeven_bps(result),
        # existing tables
        summary_table        = metrics.summary_table(result),
        risk_table           = metrics.risk_table(result),
        drawdown_table       = metrics.drawdown_table(result),
        cost_sensitivity     = metrics.cost_sensitivity(result),
        rolling_sharpe       = metrics.rolling_sharpe(result, window=5),
        rolls_per_day        = rpd,
        # statistical validity
        sharpe_ci            = metrics.sharpe_confidence_interval(result),
        var_series           = metrics.value_at_risk(result),
        cvar_series          = metrics.conditional_value_at_risk(result),
        # pattern analysis
        morning_afternoon    = metrics.morning_afternoon_split(result.day_results),
        pnl_by_weekday       = metrics.pnl_by_weekday(result),
        expiry_day           = metrics.expiry_day_pnl(result),
        round_trips          = metrics.round_trips_by_leg(result.trades_df),
        notional_per_day     = metrics.notional_per_day(result),
        unit                 = result.unit,
    )

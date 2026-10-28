# src/stratbt/report/pages.py
"""
Layout stage: composes ReportData into pages. Everything here is positioning;
numbers and sentences come from data.py, drawing from charts.py/layout.py.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.figure import Figure

from stratbt.report import charts, theme as T
from stratbt.report.data import ReportData
from stratbt.report.layout import Page

X0, W = T.MARGIN, T.CONTENT_W
ROWS_PER_PAGE = 30


def _legend_items(data: ReportData, with_combined: bool = True):
    items = [(u, T.series_color(u, data.underliers)) for u in data.underliers]
    if with_combined and len(data.underliers) > 1:
        items.append(("Combined", T.COMBINED))
    return items


def _stat_table(p: Page, data: ReportData, x: float, y: float, w: float, rows: list[list[str]],
                label_w: float = 1.9) -> float:
    n = max(1, len(data.stats_cols))
    cw = [label_w] + [(w - label_w) / n] * n
    return p.table(x, y, ["Metric"] + data.stats_cols, rows, cw, row_h=0.2, size=7.6)


# --------------------------------------------------------------------------
def summary_page(data: ReportData) -> Page:
    p = Page("Summary")
    p.text(X0, 0.78, "OPTIONS BACKTEST REPORT", size=7.5, color=T.MUTED, weight="bold")
    p.text(X0, 0.98, data.strategy_name, size=24, weight="bold")
    sub = f"{data.period_label}  ·  {data.n_days} trading days  ·  {', '.join(data.underliers)}"
    p.text(X0, 1.52, sub, size=9.5, color=T.INK_2)

    lo, hi = data.sharpe_ci
    ci = f"95% CI {T.fnum(lo, 1)} to {T.fnum(hi, 1)}" if lo == lo and hi == hi else "interval n/a"
    cost = data.fees + data.slippage
    p.kpi_cards(X0, 1.9, W, 0.98, [
        ("Net PnL", T.fmoney(data.net_pnl, data.unit, signed=True),
         f"gross {T.fmoney(data.gross_pnl, data.unit, signed=True)}" if cost else "no trading costs"),
        ("Sharpe", T.fnum(data.sharpe, 2), ci),
        ("Max drawdown", T.fmoney(data.max_dd, data.unit), f"longest {data.max_dd_days} days"),
        ("Day win rate", T.fpct(data.win_rate), f"profit factor {T.fnum(data.profit_factor, 2)}"),
        ("Trades", f"{data.n_trades:,}", f"{T.fnum(data.n_trades / max(data.n_days, 1), 0)} per day"),
        ("Costs paid", T.fmoney(cost, data.unit), f"fees {T.fcompact(data.fees, data.unit)} + slip. {T.fcompact(data.slippage, data.unit)}"),
    ])

    p.heading(X0, 3.12, "Run configuration")
    p.kv(X0, 3.42, 3.15, data.config_rows, key_w=0.95, size=7.8)

    cx, cw = X0 + 3.5, W - 3.5
    p.heading(cx, 3.12, "Cumulative net PnL")
    if len(data.underliers) > 1:
        p.legend(cx + cw - 2.6, 3.14, _legend_items(data))
    ax = p.ax(cx + 0.55, 3.48, cw - 1.35, 2.2)
    charts.equity(ax, data)

    p.heading(X0, 5.95, "Key observations")
    y = 6.28
    for line in data.observations:
        p.text(X0 + 0.02, y, "▪", size=8, color=T.BLUE)
        y = p.text(X0 + 0.2, y, line, size=8.6, color=T.INK, wrap=W - 0.3, line_h=1.3) + 0.045
    return p


def performance_page(data: ReportData) -> Page:
    p = Page("Performance")
    p.heading(X0, 0.8, "Cumulative net PnL", sub=data.captions["equity"])
    if len(data.underliers) > 1:
        p.legend(X0 + W - 2.6, 0.82, _legend_items(data))
    charts.equity(p.ax(X0 + 0.55, 1.38, W - 1.45, 2.15), data)

    p.heading(X0, 3.78, "Drawdown from running peak", sub=data.captions["drawdown"])
    charts.drawdown(p.ax(X0 + 0.55, 4.35, W - 1.45, 1.05), data)

    p.heading(X0, 5.62, "Daily net PnL (all underliers)", sub=data.captions["daily"])
    charts.daily_bars(p.ax(X0 + 0.55, 6.2, W - 1.45, 1.45), data)
    return p


def risk_page(data: ReportData) -> Page:
    p = Page("Risk")
    lw = 5.0
    p.heading(X0, 0.8, "Risk metrics", sub=data.risk_note, w=lw)
    n = max(1, len(data.stats_cols))
    y = p.table(X0, 1.6, ["Metric"] + data.stats_cols, data.risk_rows, [2.0] + [(lw - 2.0) / n] * n, row_h=0.215)

    p.heading(X0, y + 0.3, "Largest drawdown episodes", sub="Peak-to-trough losses of the combined equity curve.", w=lw)
    if data.drawdown_rows:
        p.table(X0, y + 0.95, ["Series", "Trough", "Depth", "Recovered"], data.drawdown_rows,
                [1.1, 1.25, 1.2, 1.45], align=["l", "l", "r", "l"], row_h=0.215)
    else:
        p.text(X0, y + 0.95, "No drawdowns.", size=8.5, color=T.MUTED)

    rx, rw = X0 + lw + 0.5, W - lw - 0.5
    lo, hi = data.sharpe_ci
    p.heading(rx, 0.8, "Sharpe ratio with 95% interval", w=rw,
              sub=("The interval includes zero: the data cannot separate skill from luck." if lo == lo and lo <= 0 <= hi
                   else "Bootstrap over days; wide intervals are normal for short samples."))
    charts.sharpe_ci(p.ax(rx + 0.55, 1.6, rw - 0.8, 1.75 if len(data.sharpe_ci_df) > 1 else 1.2), data)
    p.heading(rx, 3.85, "Distribution of daily PnL", w=rw, sub="Dashed: 5th-percentile loss (VaR); dotted: average of days beyond it (CVaR).")
    charts.histogram(p.ax(rx + 0.2, 4.7, rw - 0.4, 2.3), data)
    return p


def behaviour_page(data: ReportData) -> Page:
    p = Page("Behaviour")
    half = (W - 0.5) / 2
    p.heading(X0, 0.8, "Average intraday PnL path", w=half, sub="Mean mark-to-market by time of day across all days.")
    charts.intraday(p.ax(X0 + 0.55, 1.45, half - 1.05, 2.1), data)
    p.heading(X0 + half + 0.5, 0.8, "Morning vs afternoon", w=half, sub="Total PnL earned before and after 12:00.")
    charts.am_pm(p.ax(X0 + half + 1.05, 1.45, half - 0.55, 2.1), data)

    p.heading(X0, 4.1, "Average PnL by weekday", w=half, sub="Per underlier-day; n = number of observations.")
    charts.weekday(p.ax(X0 + 0.55, 4.75, half - 0.75, 2.1), data)
    p.heading(X0 + half + 0.5, 4.1, "Expiry day vs other days", w=half, sub="Time decay and gamma peak on expiry day.")
    charts.expiry(p.ax(X0 + half + 1.05, 4.75, half - 0.75, 2.1), data)
    return p


def costs_page(data: ReportData) -> Page:
    p = Page("Costs & execution")
    half = (W - 0.5) / 2
    gross = data.gross_pnl
    p.heading(X0, 0.8, "From gross to net PnL", w=half,
              sub=(f"Trading costs were {T.fmoney(data.fees + data.slippage, data.unit)}"
                   + (f", {(data.fees + data.slippage) / abs(gross):.0%} of gross PnL." if gross else ".")))
    charts.cost_waterfall(p.ax(X0 + 0.55, 1.5, half - 0.85, 2.3), data)
    p.heading(X0 + half + 0.5, 0.8, "Sensitivity to extra costs", w=half,
              sub="Net PnL if every fill paid additional basis points of premium.")
    charts.cost_sensitivity(p.ax(X0 + half + 1.05, 1.5, half - 0.75, 2.3), data)

    p.heading(X0, 4.3, "Cost breakdown", w=half)
    y_end = p.table(X0, 4.65, ["", "Gross", "Slippage", "Fees", "Net"], data.cost_rows,
                    [1.3] + [(half - 1.3) / 4] * 4, row_h=0.23, bold_last=len(data.cost_rows) > 1)
    p.text(X0, y_end + 0.15, "Gross = what mid-price fills with no fees would have earned. Slippage = spread paid "
           "versus the mid price. All figures are already reflected in net PnL.", size=7.5, color=T.MUTED, wrap=half)
    p.heading(X0 + half + 0.5, 4.3, "Trading activity", w=half)
    p.kv(X0 + half + 0.5, 4.65, half, data.activity_rows, key_w=1.7, size=8)
    return p


def statistics_pages(data: ReportData) -> list[Page]:
    pages = []
    rows = data.daily_rows
    chunks = [rows[i:i + ROWS_PER_PAGE] for i in range(0, len(rows), ROWS_PER_PAGE)] or [[]]
    lw = 4.9
    for k, chunk in enumerate(chunks):
        p = Page("Statistics" if k == 0 else "Daily results (continued)")
        n = len(data.daily_headers)
        cw = [1.1] + [(lw - 1.1) / (n - 1)] * (n - 1)
        if k == 0:
            p.heading(X0, 0.8, "Daily net PnL", w=lw)
        else:
            p.heading(X0, 0.8, "Daily net PnL (continued)", w=lw)
        p.table(X0, 1.2, data.daily_headers, chunk, cw, row_h=0.2, size=7.4)
        if k == 0:
            rx, rw = X0 + lw + 0.4, W - lw - 0.4
            p.heading(rx, 0.8, "Performance statistics", w=rw)
            _stat_table(p, data, rx, 1.2, rw, data.stats_rows, label_w=1.75)
        pages.append(p)
    return pages


def appendix_page(data: ReportData) -> Page:
    p = Page("Methodology")
    half = (W - 0.5) / 2
    p.heading(X0, 0.8, "Definitions", w=half)
    defs = [
        ("Net PnL", "Sum of realised PnL after fees and spread costs; open positions are closed at the session end."),
        ("Sharpe", "Mean daily PnL ÷ its standard deviation, ×√252. Risk-free rate is zero. With few days it is very noisy; the interval is a bootstrap over days."),
        ("Max drawdown", "Largest peak-to-trough fall of cumulative daily PnL."),
        ("VaR / CVaR 95%", "The loss exceeded on the worst 5% of days, and the average loss on those days (historical, not modelled)."),
        ("Profit factor", "Sum of winning days ÷ sum of losing days."),
        ("Calmar / Sortino", "Return per unit of drawdown; return per unit of downside volatility."),
        ("Ulcer index", "Root-mean-square depth of drawdown: penalises deep, long drawdowns."),
        ("Omega ratio", "Probability-weighted gains over losses relative to a zero threshold."),
        ("Slippage", "Half the bid/ask spread paid on each fill relative to the mid price."),
    ]
    y = 1.25
    for k, v in defs:
        p.text(X0, y, k, size=8, weight="bold")
        y = p.text(X0 + 1.2, y, v, size=7.8, color=T.INK_2, wrap=half - 1.2) + 0.07

    rx = X0 + half + 0.5
    p.heading(rx, 0.8, "Assumptions and limitations", w=half)
    lims = [
        "Fills use last-trade or modelled mid prices plus a modelled spread, not a recorded order book. There is no market impact or queue model.",
        "Orders fill after the configured delay; a strategy only ever sees prices that were already known at decision time.",
        "No margin or buying-power model. Short-premium strategies can lose more than any account would allow.",
        "No early exercise, assignment or expiry settlement beyond closing positions at the session end.",
        "One trading session at a time; nothing is held overnight.",
        "Short samples cannot establish an edge. Treat any positive result as a hypothesis to test out of sample.",
    ]
    y = 1.25
    for line in lims:
        p.text(rx, y, "▪", size=8, color=T.MUTED)
        y = p.text(rx + 0.18, y, line, size=7.8, color=T.INK_2, wrap=half - 0.2) + 0.07
    p.heading(rx, y + 0.2, "Reproducibility", w=half)
    p.kv(rx, y + 0.55, half, [(k, v) for k, v in data.config_rows], key_w=0.95, size=7.4)
    p.text(X0, T.PAGE_H - 0.95, "Backtested results are hypothetical and are not a prediction or investment advice.",
           size=7.8, color=T.MUTED, style="italic")
    return p


def build_all_pages(data: ReportData) -> list[tuple[Figure, str]]:
    with mpl.rc_context(T.RC):
        pages = [summary_page(data), performance_page(data), risk_page(data), behaviour_page(data),
                 costs_page(data), *statistics_pages(data), appendix_page(data)]
    total = len(pages)
    out = []
    for i, p in enumerate(pages, 1):
        with mpl.rc_context(T.RC):
            p.chrome(data.strategy_name, i, total, data.generated_at, data.version)
        out.append((p.fig, p.section))
    return out

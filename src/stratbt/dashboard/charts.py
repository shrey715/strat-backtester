# src/stratbt/dashboard/charts.py
"""
Plotly figures for the dashboard. Overview-level charts take the same
`ReportData` the PDF uses, so the two always agree. No template is set:
Streamlit's theme (light or dark) is applied by `st.plotly_chart`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from stratbt.palette import BLUE, ORANGE, SERIES_COLORS
from stratbt.report.data import ReportData

GAIN = BLUE
LOSS = "#e34948"
MUTED = "#8b8a85"


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _money_fmt(unit: str) -> dict:
    """Axis/hover number format for a unit: $ prefix for USD, ₹ for INR, otherwise a suffix."""
    if unit == "USD":
        return dict(tickformat="$,.0f", hoverformat="$,.0f")
    if unit == "INR":
        return dict(tickprefix="₹", hoverformat="₹,.0f")
    return dict(ticksuffix="" if unit in ("", "pts") else f" {unit}", tickformat=",.0f", hoverformat=",.1f")


def _layout(fig: go.Figure, unit: str | None = None, height: int = 330, legend: bool = True,
            ytitle: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(zeroline=True, zerolinecolor=MUTED, zerolinewidth=1, title=ytitle,
                   **(_money_fmt(unit) if unit else {})),
    )
    return fig


# ── overview ────────────────────────────────────────────────────────────

def equity(d: ReportData) -> go.Figure:
    fig = go.Figure()
    for u in d.underliers:
        fig.add_scatter(x=d.cum.index, y=d.cum[u], name=u, mode="lines",
                        line=dict(color=SERIES_COLORS[u], width=2), hovertemplate="%{y:,.0f}")
    if len(d.underliers) > 1:
        fig.add_scatter(x=d.cum_combined.index, y=d.cum_combined, name="Combined", mode="lines",
                        line=dict(color="#8b8a85", width=3, dash="dot"), hovertemplate="%{y:,.0f}")
    return _layout(fig, d.unit, 360, legend=len(d.underliers) > 1)


def drawdown(d: ReportData) -> go.Figure:
    fig = go.Figure(go.Scatter(x=d.drawdown.index, y=d.drawdown, mode="lines", name="Drawdown",
                               line=dict(color=LOSS, width=1.8), fill="tozeroy", fillcolor=_rgba(LOSS, 0.15),
                               hovertemplate="%{y:,.0f}"))
    return _layout(fig, d.unit, 260, legend=False)


def daily_bars(d: ReportData) -> go.Figure:
    v = d.combined
    fig = go.Figure(go.Bar(x=v.index, y=v, marker_color=[GAIN if a >= 0 else LOSS for a in v], name="Net PnL",
                           hovertemplate="%{y:,.0f}"))
    fig.update_traces(marker_line_width=0)
    fig.update_layout(bargap=0.25)
    return _layout(fig, d.unit, 260, legend=False)


# ── risk ────────────────────────────────────────────────────────────────

def sharpe_ci(d: ReportData) -> go.Figure:
    df = d.sharpe_ci_df.dropna(how="all")
    fig = go.Figure()
    for n, r in df.iterrows():
        fig.add_scatter(x=[r["ci_low"], r["ci_high"]], y=[n, n], mode="lines", showlegend=False,
                        line=dict(color=SERIES_COLORS[n], width=5), hoverinfo="skip")
        fig.add_scatter(x=[r["sharpe"]], y=[n], mode="markers", showlegend=False,
                        marker=dict(color=SERIES_COLORS[n], size=11, line=dict(color="white", width=2)),
                        hovertemplate=f"Sharpe %{{x:.2f}}  (95% CI {r['ci_low']:.1f} to {r['ci_high']:.1f})<extra>{n}</extra>")
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_layout(height=max(160, 70 * len(df) + 70), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="annualised Sharpe", zeroline=False), yaxis=dict(autorange="reversed"))
    return fig


def histogram(d: ReportData) -> go.Figure:
    v = d.combined.values
    fig = go.Figure(go.Histogram(x=v, nbinsx=int(min(14, max(5, len(v) ** 0.5 * 1.6))), marker_color=BLUE,
                                 opacity=0.9, name="days", hovertemplate="%{x}: %{y} days<extra></extra>"))
    if len(v) >= 2:
        q = float(np.quantile(v, 0.05))
        tail = v[v <= q]
        fig.add_vline(x=q, line_color=LOSS, line_dash="dash", annotation_text="VaR 95%", annotation_position="top")
        if len(tail):
            fig.add_vline(x=float(tail.mean()), line_color=LOSS, line_dash="dot",
                          annotation_text="CVaR", annotation_position="bottom")
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_layout(bargap=0.06, height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      xaxis=dict(title=f"daily net PnL ({d.unit})", **_money_fmt(d.unit)),
                      yaxis=dict(title="days"))
    return fig


# ── behaviour ───────────────────────────────────────────────────────────

def intraday(d: ReportData) -> go.Figure:
    fig = go.Figure()
    for u in d.intraday.columns:
        fig.add_scatter(x=d.intraday.index, y=d.intraday[u], name=u, mode="lines",
                        line=dict(color=SERIES_COLORS[u], width=2), hovertemplate="%{y:,.1f}")
    fig = _layout(fig, d.unit, 300, legend=len(d.intraday.columns) > 1)
    fig.update_xaxes(tickformat="%H:%M")
    return fig


def _bars(labels, values, unit, counts=None) -> go.Figure:
    text = [f"n={c}" for c in counts] if counts is not None else None
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=[GAIN if v >= 0 else LOSS for v in values],
                           text=text, textposition="outside", hovertemplate="%{y:,.0f}<extra></extra>"))
    fig.update_layout(bargap=0.35)
    return _layout(fig, unit, 280, legend=False)


def am_pm(d: ReportData) -> go.Figure:
    df = d.am_pm
    fig = go.Figure()
    fig.add_bar(x=df.index, y=df["morning_pnl"], name="Morning (before 12:00)", marker_color=BLUE)
    fig.add_bar(x=df.index, y=df["afternoon_pnl"], name="Afternoon", marker_color=ORANGE)
    fig.update_layout(barmode="group", bargap=0.3)
    return _layout(fig, d.unit, 280)


def weekday(d: ReportData) -> go.Figure:
    df = d.weekday
    return _bars([w[:3] for w in df["weekday"]], df["avg_pnl"].tolist(), d.unit, df["count"].astype(int).tolist())


def expiry(d: ReportData) -> go.Figure:
    df = d.expiry
    return _bars(df["day_type"].tolist(), df["avg_pnl"].tolist(), d.unit, df["count"].astype(int).tolist())


# ── costs ───────────────────────────────────────────────────────────────

def cost_waterfall(d: ReportData) -> go.Figure:
    t = d.cost_breakdown.loc["Total"]
    fig = go.Figure(go.Waterfall(
        x=["Gross PnL (mid prices)", "Slippage", "Fees", "Net PnL"],
        y=[t["gross"], -t["slippage"], -t["fees"], t["net"]],
        measure=["absolute", "relative", "relative", "total"],
        increasing=dict(marker_color=GAIN), decreasing=dict(marker_color=MUTED), totals=dict(marker_color=GAIN if t["net"] >= 0 else LOSS),
        connector=dict(line=dict(color=MUTED, width=1)), hovertemplate="%{y:,.0f}<extra></extra>"))
    return _layout(fig, d.unit, 300, legend=False)


def cost_sensitivity(d: ReportData) -> go.Figure:
    cs = d.cost_sens
    fig = go.Figure(go.Scatter(x=cs["cost_bps"], y=cs["net_pnl"], mode="lines+markers", name="Net PnL",
                               line=dict(color=BLUE, width=2.5), marker=dict(size=7),
                               hovertemplate="%{x} bps: %{y:,.0f}<extra></extra>"))
    if 0 < d.breakeven_bps <= cs["cost_bps"].max():
        fig.add_vline(x=d.breakeven_bps, line_dash="dash", line_color=MUTED,
                      annotation_text=f"break-even {d.breakeven_bps:.1f} bps", annotation_position="top right")
    fig = _layout(fig, d.unit, 300, legend=False)
    fig.update_xaxes(title="additional cost, bps of premium per fill")
    return fig


# ── comparison ──────────────────────────────────────────────────────────

def compare_equity(runs: list) -> go.Figure:
    fig = go.Figure()
    palette = [BLUE, ORANGE, "#1baf7a", "#6250d6", "#e87ba4", "#008300"]
    for i, r in enumerate(runs):
        s = r["data"].cum_combined
        fig.add_scatter(x=list(range(1, len(s) + 1)), y=s.values, name=r["label"], mode="lines",
                        line=dict(color=palette[i % len(palette)], width=2.4), hovertemplate="%{y:,.0f}")
    fig = _layout(fig, runs[0]["data"].unit, 380)
    fig.update_xaxes(title="trading day")
    return fig


# ── single-day drill-down ───────────────────────────────────────────────

def day_mtm(dr, unit: str) -> go.Figure:
    y, x = dr.mtm.values, dr.mtm.index
    fig = go.Figure()
    fig.add_scatter(x=x, y=np.where(y >= 0, y, 0), fill="tozeroy", fillcolor=_rgba(GAIN, 0.15),
                    line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=x, y=np.where(y < 0, y, 0), fill="tozeroy", fillcolor=_rgba(LOSS, 0.15),
                    line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=x, y=y, mode="lines", line=dict(color="#8b8a85", width=1.8), name="Net MTM",
                    hovertemplate="%{y:,.2f}")
    return _layout(fig, unit, 300, legend=False)


def day_price_strike(dr, underlying: pd.Series, unit: str) -> go.Figure:
    color = SERIES_COLORS[dr.underlier]
    fig = go.Figure()
    fig.add_scatter(x=underlying.index, y=underlying.values, name="Underlying", mode="lines",
                    line=dict(color=MUTED, width=1.4), hovertemplate="%{y:,.2f}")
    log = dr.holdings_log
    if not log.empty:
        fig.add_scatter(x=log.index, y=log["strike"], name="Strike held", mode="lines", line_shape="hv",
                        line=dict(color=color, width=2.4), hovertemplate="%{y:,.0f}")
    fig = _layout(fig, None, 320)
    fig.update_yaxes(title="price / strike")
    return fig


def day_legs(dr, unit: str) -> go.Figure:
    log = dr.holdings_log
    fig = go.Figure()
    if not log.empty:
        fig.add_scatter(x=log.index, y=log["ce_mtm_price"], name="Call", mode="lines", line=dict(color=BLUE, width=1.6))
        fig.add_scatter(x=log.index, y=log["pe_mtm_price"], name="Put", mode="lines", line=dict(color=ORANGE, width=1.6))
        fig.add_scatter(x=log.index, y=log["ce_mtm_price"] + log["pe_mtm_price"], name="Straddle value",
                        mode="lines", line=dict(color=MUTED, width=2, dash="dot"))
    return _layout(fig, None, 320, ytitle="option price")


def day_realized(dr, unit: str) -> go.Figure:
    """Net MTM vs the realised-PnL staircase (steps whenever a position is closed)."""
    from stratbt.position import Position
    fig = go.Figure()
    fig.add_scatter(x=dr.mtm.index, y=dr.mtm.values, name="Net MTM", mode="lines", line=dict(color=BLUE, width=1.6))
    if dr.trades:
        ledger = Position(multiplier=dr.multiplier, allow_short=True)
        ts, realized = [dr.mtm.index[0]], [0.0]
        for t in sorted(dr.trades, key=lambda x: x.timestamp):
            (ledger.buy if t.side == "BUY" else ledger.sell)(t.timestamp, t.instrument, t.price, t.qty, fee=t.fee)
            ts.append(t.timestamp)
            realized.append(ledger.net_realized)
        ts.append(dr.mtm.index[-1])
        realized.append(ledger.net_realized)
        fig.add_scatter(x=ts, y=realized, name="Realised", mode="lines", line_shape="hv",
                        line=dict(color=ORANGE, width=2.4, dash="dash"))
    return _layout(fig, unit, 300)

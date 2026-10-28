# src/stratbt/dashboard/app.py
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import stratbt.metrics as metrics
from stratbt.engine import BacktestEngine, BacktestResult
from stratbt.execution import ExecutionModel
from stratbt.palette import SERIES_COLORS
from stratbt.position import Position
from stratbt.providers import DataProvider, NSETickProvider, SyntheticProvider
from stratbt.report import build_pdf_report
from stratbt.risk import RiskLimits
from stratbt.strategy import registry

st.set_page_config(
    page_title="Strategy Backtester",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR_ENV = "STRATBT_DATA_DIR"
DEFAULT_NSE_DIR = Path(os.environ.get(DATA_DIR_ENV, "data/allData"))

POSITIVE_COLOR = "#2ca02c"
NEGATIVE_COLOR = "#d62728"
GRID_COLOR = "#d5d5d5"


def _u() -> str:
    """Unit label of the result currently being displayed."""
    return st.session_state.get("unit", "pts")


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _layout(title: str, y_title: str, height: int = 400) -> dict:
    """
    Shared Plotly layout for every chart in the app.
    Legend sits below the plot with a wide bottom margin so it never
    collides with the title or crowded date tick labels.
    """
    return dict(
        title=dict(text=title, font=dict(size=14), pad=dict(b=10)),
        yaxis_title=y_title,
        hovermode="x unified",
        height=height,
        template="plotly_white",
        margin=dict(l=60, r=30, t=55, b=90),
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        xaxis=dict(tickangle=-30),
    )


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def plotly_cumulative_pnl(result: BacktestResult) -> go.Figure:
    daily = result.daily_pnl.fillna(0)
    cum   = daily.cumsum()
    combo = cum.sum(axis=1)

    fig = go.Figure()
    for col in cum.columns:
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum[col], name=col, mode="lines",
            line=dict(width=2.5, color=SERIES_COLORS.get(col, "#888")),
        ))
    fig.add_trace(go.Scatter(
        x=combo.index, y=combo.values, name="Combined", mode="lines",
        line=dict(width=3, dash="dash", color=SERIES_COLORS["Combined"]),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.update_layout(**_layout("Cumulative PnL", f"PnL ({_u()})"))
    return fig


def plotly_daily_pnl(result: BacktestResult) -> go.Figure:
    daily = result.daily_pnl.fillna(0).reset_index().melt(
        id_vars="date", var_name="underlier", value_name="pnl"
    )
    fig = px.bar(
        daily, x="date", y="pnl", color="underlier", barmode="group",
        color_discrete_map=SERIES_COLORS,
        labels={"pnl": f"PnL ({_u()})", "date": ""},
    )
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.update_layout(**_layout("Daily PnL by Underlier", f"PnL ({_u()})"))
    return fig


def plotly_drawdown(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    for col in result.daily_pnl.fillna(0).columns:
        cum   = result.daily_pnl[col].fillna(0).cumsum()
        dd    = -(cum.cummax() - cum)
        color = SERIES_COLORS.get(col, "#888")
        fig.add_trace(go.Scatter(
            x=cum.index, y=dd.values, name=col, mode="lines",
            line=dict(width=2, color=color),
            fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.12),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.update_layout(**_layout("Drawdown", f"Drawdown ({_u()})"))
    return fig


def plotly_rolling_sharpe(result: BacktestResult, window: int = 5) -> go.Figure:
    daily = result.daily_pnl.fillna(0)
    rs    = (daily.rolling(window).mean() / daily.rolling(window).std()) * np.sqrt(252)

    fig = go.Figure()
    for col in rs.columns:
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs[col], name=col, mode="lines",
            line=dict(width=2.5, color=SERIES_COLORS.get(col, "#888")),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.update_layout(**_layout(f"Rolling {window}-Day Sharpe Ratio", "Sharpe (annualized)"))
    return fig


def plotly_rolls_per_day(result: BacktestResult) -> go.Figure:
    rpd = metrics.rolls_per_day(result)
    fig = px.bar(
        rpd, x="date", y="rolls", color="underlier", barmode="group",
        color_discrete_map=SERIES_COLORS,
        labels={"rolls": "Rolls", "date": ""},
    )
    fig.update_layout(**_layout("Rolls per Day", "Number of rolls"))
    return fig


def plotly_intraday_mtm(dr) -> go.Figure:
    y = dr.mtm.values
    x = dr.mtm.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=np.where(y >= 0, y, 0),
        fill="tozeroy", fillcolor=_hex_to_rgba(POSITIVE_COLOR, 0.15),
        line=dict(width=0), showlegend=False, name="Profit",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=np.where(y < 0, y, 0),
        fill="tozeroy", fillcolor=_hex_to_rgba(NEGATIVE_COLOR, 0.15),
        line=dict(width=0), showlegend=False, name="Loss",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=SERIES_COLORS["Combined"], width=2), name="MTM",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.update_layout(**_layout(
        f"Intraday MTM — {dr.underlier} {dr.date}  ({dr.final_pnl:+.1f} {_u()})",
        f"MTM PnL ({_u()})", height=360,
    ))
    return fig


def plotly_strike_held(dr) -> go.Figure:
    log = dr.holdings_log
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=log.index, y=log["strike"], mode="lines", line_shape="hv",
        line=dict(color=SERIES_COLORS["NIFTY"], width=2), name="Strike",
    ))
    fig.update_layout(**_layout(f"Strike Held — {dr.underlier} {dr.date}", "Strike", height=360))
    visited = sorted(log["strike"].dropna().unique().astype(int))
    fig.update_yaxes(tickvals=visited, tickfont=dict(size=9))
    return fig


def plotly_cost_sensitivity(result: BacktestResult) -> go.Figure:
    cs     = metrics.cost_sensitivity(result)
    colors = [NEGATIVE_COLOR if v < 0 else POSITIVE_COLOR for v in cs["net_pnl"]]
    fig    = go.Figure(go.Bar(
        x=cs["cost_bps"].astype(str), y=cs["net_pnl"], marker_color=colors,
        text=cs["net_pnl"].round(0).astype(int), textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.update_layout(**_layout("Net PnL vs Transaction Cost", f"Net PnL ({_u()})"))
    fig.update_xaxes(title_text="Cost (bps per trade)", tickangle=0)
    return fig


def plotly_pnl_histogram(result: BacktestResult, show_var: bool = False) -> go.Figure:
    daily = result.daily_pnl.fillna(0)
    fig   = go.Figure()
    for col in daily.columns:
        fig.add_trace(go.Histogram(
            x=daily[col], name=col, nbinsx=15,
            marker_color=SERIES_COLORS.get(col, "#888"), opacity=0.75,
        ))
    fig.add_vline(x=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    if show_var:
        var = metrics.value_at_risk(result)
        for col in daily.columns:
            fig.add_vline(
                x=-var[col], line_dash="dash", line_color=NEGATIVE_COLOR, line_width=1.5,
                annotation_text=f"{col} VaR95", annotation_font_size=9,
            )
    fig.update_layout(**_layout("Daily PnL Distribution", "Frequency"), barmode="overlay")
    fig.update_xaxes(title_text=f"Daily PnL ({_u()})")
    return fig


def plotly_sharpe_ci(result: BacktestResult) -> go.Figure:
    ci  = metrics.sharpe_confidence_interval(result)
    fig = go.Figure(go.Bar(
        x=ci.index, y=ci["sharpe"],
        error_y=dict(
            type="data", symmetric=False,
            array=ci["ci_high"] - ci["sharpe"],
            arrayminus=ci["sharpe"] - ci["ci_low"],
        ),
        marker_color=[SERIES_COLORS.get(ul, "#888") for ul in ci.index],
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.update_layout(**_layout("Sharpe Ratio (95% Bootstrap CI)", "Sharpe (annualized)"))
    fig.update_xaxes(tickangle=0)
    return fig


def plotly_pnl_waterfall(result: BacktestResult) -> go.Figure:
    daily  = result.daily_pnl.fillna(0).sum(axis=1)
    dates  = [str(d) for d in daily.index]
    values = daily.values.tolist()
    fig = go.Figure(go.Waterfall(
        x=dates, y=values, measure=["relative"] * len(values),
        connector=dict(line=dict(color=GRID_COLOR, width=0.5)),
        increasing=dict(marker_color=POSITIVE_COLOR),
        decreasing=dict(marker_color=NEGATIVE_COLOR),
    ))
    fig.update_layout(**_layout("Daily PnL Waterfall (Combined)", f"Cumulative PnL ({_u()})"))
    return fig


@st.cache_data(show_spinner=False)
def _underlying_cached(_provider: DataProvider, provider_key: str, day: date, underlier: str) -> pd.Series:
    return _provider.underlying_prices(day, underlier)


def _underlying(dr) -> pd.Series:
    return _underlying_cached(st.session_state["provider"], st.session_state["provider_key"],
                              dr.date, dr.underlier)


def plotly_futures_through_day(dr) -> go.Figure:
    color = SERIES_COLORS.get(dr.underlier, "#888")
    fut   = _underlying(dr)
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=fut.index, y=fut.values, mode="lines", name="Futures",
        line=dict(color=color, width=2),
    ))
    fig.update_layout(**_layout(f"Underlying Price — {dr.underlier} {dr.date}", f"Price ({_u()})", height=340))
    return fig


def plotly_futures_vs_strike(dr) -> go.Figure:
    color = SERIES_COLORS.get(dr.underlier, "#888")
    fut   = _underlying(dr)
    log   = dr.holdings_log

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fut.index, y=fut.values, name="Underlying", mode="lines",
        line=dict(color="#999999", width=1.5), yaxis="y1",
    ))
    if not log.empty:
        fig.add_trace(go.Scatter(
            x=log.index, y=log["strike"], name="Strike", mode="lines", line_shape="hv",
            line=dict(color=color, width=2.5), yaxis="y2",
        ))
    layout = _layout(f"Underlying vs Strike — {dr.underlier} {dr.date}", "Underlying", height=360)
    layout["margin"] = dict(l=60, r=70, t=55, b=90)
    fig.update_layout(**layout, yaxis2=dict(title="Strike", overlaying="y", side="right", showgrid=False))
    return fig


def plotly_leg_prices(dr) -> go.Figure:
    log = dr.holdings_log
    if log.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=log.index, y=log["ce_mtm_price"], name="CE", mode="lines",
        line=dict(color=SERIES_COLORS["NIFTY"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=log.index, y=log["pe_mtm_price"], name="PE", mode="lines",
        line=dict(color=SERIES_COLORS["BANKNIFTY"], width=1.5),
    ))
    combined = log["ce_mtm_price"] + log["pe_mtm_price"]
    fig.add_trace(go.Scatter(
        x=log.index, y=combined, name="Total", mode="lines",
        line=dict(color=SERIES_COLORS["Combined"], width=2.5, dash="dash"),
    ))
    fig.update_layout(**_layout(f"Option Prices — {dr.underlier} {dr.date}", "Price", height=340))
    return fig


def plotly_realized_staircase(dr) -> go.Figure:
    """
    MTM equity vs the realized-PnL staircase. The staircase steps whenever a
    position is (partly) closed; the gap to the MTM line is unrealized PnL.
    """
    trades = dr.trades
    if not trades:
        return go.Figure()

    ledger     = Position(multiplier=dr.multiplier, allow_short=True)
    timestamps = [dr.mtm.index[0]]
    realized   = [0.0]
    for t in sorted(trades, key=lambda x: x.timestamp):
        (ledger.buy if t.side == "BUY" else ledger.sell)(t.timestamp, t.instrument, t.price, t.qty, fee=t.fee)
        timestamps.append(t.timestamp)
        realized.append(ledger.net_realized)
    timestamps.append(dr.mtm.index[-1])
    realized.append(ledger.net_realized)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dr.mtm.index, y=dr.mtm.values, name="MTM", mode="lines",
        line=dict(color=SERIES_COLORS["NIFTY"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=realized, name="Realized", mode="lines", line_shape="hv",
        line=dict(color=POSITIVE_COLOR, width=2.5, dash="dash"),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.update_layout(**_layout(f"MTM vs Realized — {dr.underlier} {dr.date}", f"PnL ({_u()})", height=360))
    return fig


def plotly_roll_interval_dist(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    for ul in sorted({dr.underlier for dr in result.day_results}):
        gaps = []
        for dr in result.day_results:
            if dr.underlier != ul or dr.holdings_log.empty:
                continue
            log_r = dr.holdings_log.reset_index()
            rolls = log_r[log_r["strike"] != log_r["strike"].shift(1)].iloc[1:]
            gaps.extend(rolls["timestamp"].diff().dt.total_seconds().dropna().tolist())
        fig.add_trace(go.Histogram(
            x=gaps, name=ul, nbinsx=50,
            marker_color=SERIES_COLORS.get(ul, "#888"), opacity=0.7,
        ))
    fig.update_layout(**_layout("Seconds Between Rolls", "Frequency"), barmode="overlay")
    fig.update_xaxes(title_text="Seconds between rolls", range=[0, 300], tickangle=0)
    return fig


def plotly_morning_vs_afternoon(result: BacktestResult) -> go.Figure:
    df  = metrics.morning_afternoon_split(result.day_results)
    fig = go.Figure()
    for ul in df["underlier"].unique():
        sub   = df[df["underlier"] == ul]
        color = SERIES_COLORS.get(ul, "#888")
        fig.add_trace(go.Bar(
            x=[str(d) for d in sub["date"]], y=sub["morning_pnl"],
            name=f"{ul} AM", marker_color=color, opacity=0.9,
        ))
        fig.add_trace(go.Bar(
            x=[str(d) for d in sub["date"]], y=sub["afternoon_pnl"],
            name=f"{ul} PM", marker_color=color, opacity=0.5,
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.update_layout(**_layout("Morning vs Afternoon PnL", f"PnL ({_u()})"), barmode="group")
    return fig


def plotly_pnl_by_weekday(result: BacktestResult) -> go.Figure:
    df  = metrics.pnl_by_weekday(result)
    fig = px.bar(
        df, x="weekday", y="avg_pnl", color="underlier", barmode="group",
        color_discrete_map=SERIES_COLORS,
        labels={"weekday": "", "avg_pnl": f"Average PnL ({_u()})"},
    )
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.update_layout(**_layout("Average PnL by Day of Week", f"Avg PnL ({_u()})"))
    fig.update_xaxes(tickangle=0)
    return fig


def plotly_expiry_day_comparison(result: BacktestResult) -> go.Figure:
    df  = metrics.expiry_day_pnl(result)
    fig = px.bar(
        df, x="underlier", y="avg_pnl", color="day_type", barmode="group",
        color_discrete_map={"Expiry day": NEGATIVE_COLOR, "Non-expiry": SERIES_COLORS["Combined"]},
        text="count",
        labels={"avg_pnl": f"Average PnL ({_u()})", "count": "N days"},
    )
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.update_layout(**_layout("Expiry vs Non-Expiry Days", f"Avg PnL ({_u()})"))
    fig.update_xaxes(tickangle=0)
    return fig


def plotly_avg_intraday_mtm(result: BacktestResult) -> go.Figure:
    avg_df = metrics.avg_intraday_mtm(result)
    fig    = go.Figure()
    for col in avg_df.columns:
        fig.add_trace(go.Scatter(
            x=[str(tt) for tt in avg_df.index], y=avg_df[col], name=col, mode="lines",
            line=dict(color=SERIES_COLORS.get(col, "#888"), width=2.5),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.update_layout(**_layout("Average Intraday MTM (All Days)", f"PnL ({_u()})"))
    fig.update_xaxes(nticks=8, tickangle=0)
    return fig


def plotly_cross_underlier_scatter(result: BacktestResult) -> go.Figure:
    daily  = result.daily_pnl.fillna(0)
    ax, ay = daily.columns[0], daily.columns[1]
    corr   = daily[ax].corr(daily[ay])
    fig    = px.scatter(
        daily, x=ax, y=ay, trendline="ols",
        labels={ax: f"{ax} daily PnL", ay: f"{ay} daily PnL"},
    )
    fig.update_traces(marker=dict(color=SERIES_COLORS[ax], size=8))
    fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color=GRID_COLOR, line_width=1)
    fig.update_layout(**_layout(f"{ax} vs {ay} Daily PnL  (r = {corr:.2f})", f"{ay} PnL ({_u()})"))
    fig.update_xaxes(tickangle=0)
    return fig


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Preparing market data…")
def _make_provider(kind: str, nse_dir: str, seed: int, n_days: int) -> DataProvider:
    if kind == "Synthetic market":
        return SyntheticProvider(n_days=n_days, seed=seed)
    return NSETickProvider(nse_dir)


def _nse_available() -> bool:
    return DEFAULT_NSE_DIR.is_dir() and any(DEFAULT_NSE_DIR.glob("NSE_*"))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

EXEC_PRESETS = {
    "Next-second, no costs": ExecutionModel(),
    "Instant fills, no costs": ExecutionModel.instant(),
    "Realistic costs": ExecutionModel.realistic(),
}

with st.sidebar:
    st.header("Configuration")

    st.subheader("Data")
    kinds = ["Synthetic market", "NSE tick data"]
    kind = st.radio("Source", kinds, index=1 if _nse_available() else 0, horizontal=True,
                    label_visibility="collapsed")
    nse_dir, seed, n_days = str(DEFAULT_NSE_DIR), 7, 21
    if kind == "Synthetic market":
        seed   = st.number_input("Seed", 0, 10_000, 7, help="Same seed = same market")
        n_days = st.slider("Trading days", 5, 60, 21)
    else:
        nse_dir = st.text_input("Data folder", nse_dir,
                                help="Folder containing NSE_YYYYMMDD day directories")

    try:
        provider = _make_provider(kind, nse_dir, int(seed), int(n_days))
        provider_key = f"{kind}|{nse_dir}|{seed}|{n_days}"
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        st.stop()
    all_days = provider.trading_days

    st.divider()
    st.subheader("Strategy")
    specs = registry.available()
    strategy_key = st.selectbox("Strategy", list(specs), format_func=lambda k: specs[k].label,
                                label_visibility="collapsed")
    spec = specs[strategy_key]
    st.caption(spec.description)

    strategy_params: dict = {}
    for ps in spec.params:
        cast = int if ps.is_int else float
        strategy_params[ps.name] = st.slider(
            ps.label or ps.name, cast(ps.min), cast(ps.max), cast(ps.default), cast(ps.step)
        )

    st.divider()
    st.subheader("Universe")
    underliers = st.multiselect("Underliers", provider.underliers, default=provider.underliers)

    _fmt = lambda d: d.strftime("%b %d")
    if all_days:
        col_from, col_to = st.columns(2)
        start_date = col_from.selectbox("From", all_days, index=0, format_func=_fmt)
        end_date   = col_to.selectbox("To", all_days, index=len(all_days) - 1, format_func=_fmt)
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        selected_days = [d for d in all_days if start_date <= d <= end_date]
    else:
        start_date, end_date = None, None
        selected_days = []

    st.divider()
    with st.expander("Execution & risk", expanded=False):
        preset = st.selectbox("Fill model", list(EXEC_PRESETS))
        base = EXEC_PRESETS[preset]
        delay_s = st.number_input("Fill delay (s)", 0, 30, base.delay_s,
                                  help="Seconds between decision and fill. 0 = same-second (optimistic).")
        spread_bps = st.number_input("Half-spread (bps)", 0.0, 1000.0, base.half_spread_bps, 5.0)
        commission = st.number_input("Commission / contract", 0.0, 100.0, base.commission_per_contract, 0.05)
        max_loss   = st.number_input("Max daily loss (0 = off)", 0.0, 1e9, 0.0, 100.0,
                                     help="Flatten and halt for the day once net loss reaches this amount.")
        max_open   = st.number_input("Max open contracts (0 = off)", 0, 10_000, 0)

    st.divider()
    run_button = st.button(
        "Run Backtest", type="primary", width="stretch",
        disabled=len(underliers) == 0 or len(selected_days) == 0,
    )
    st.caption(f"{len(all_days)} trading days · {provider.name} · PnL in {provider.unit}")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Strategy Backtester")
st.divider()

# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------

if run_button:
    execution = ExecutionModel(
        delay_s=int(delay_s), half_spread_bps=float(spread_bps),
        min_half_spread=base.min_half_spread if spread_bps > 0 else 0.0,
        commission_per_contract=float(commission), commission_bps=base.commission_bps,
    )
    risk = RiskLimits(max_daily_loss=max_loss or None, max_open_contracts=int(max_open) or None)
    engine  = BacktestEngine(provider, spec.factory(provider, **strategy_params),
                             execution=execution, risk=risk)
    n_steps = len(selected_days) * len(underliers)
    done    = 0
    results = []
    skipped: list[tuple[str, object]] = []

    progress = st.progress(0, text="Starting…")
    for ul in underliers:
        for day in selected_days:
            progress.progress(done / n_steps, text=f"Running {ul} · {day.strftime('%b %d')}  ({done}/{n_steps})")
            try:
                results.append(engine.run_day(day, ul))
            except ValueError:
                skipped.append((ul, day))
            done += 1
    progress.progress(1.0, text="Complete")

    if skipped:
        by_ul: dict[str, list] = {}
        for ul, day in skipped:
            by_ul.setdefault(ul, []).append(day)
        for ul, days in by_ul.items():
            if len(days) == len(selected_days):
                st.error(f"**{ul}** — no data found for any selected day. Remove it from Underliers and re-run.")
            else:
                st.warning(f"**{ul}** — skipped {len(days)} day(s) with missing data.")

    if not results:
        st.error("No results produced. Check the underlier and date selection above.")
        st.stop()

    st.session_state["result"]        = BacktestResult(results, unit=provider.unit)
    st.session_state["unit"]          = provider.unit
    st.session_state["provider"]      = provider
    st.session_state["provider_key"]  = provider_key
    st.session_state["strategy_name"] = spec.label
    st.session_state["date_range"]    = (start_date, end_date)
    st.session_state["n_days"]        = len(selected_days) - len(skipped) // max(len(underliers), 1)
    st.session_state.pop("pdf_report", None)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if "result" not in st.session_state:
    st.info("Configure a strategy in the sidebar and click **Run Backtest** to begin.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Available days", len(all_days))
    col2.metric("Period", f"{all_days[0]:%b %d, %Y} – {all_days[-1]:%b %d, %Y}" if all_days else "—")
    col3.metric("Underliers", ", ".join(provider.underliers))
    st.stop()

result        = st.session_state["result"]
strategy_name = st.session_state["strategy_name"]
_d0, _d1      = st.session_state.get("date_range", (None, None))
_n_days_run   = st.session_state.get("n_days", "?")

daily     = result.daily_pnl.fillna(0)
combined  = daily.sum().sum()
n_trades  = len(result.trades_df)
avg_rolls = metrics.rolls_per_day(result)["rolls"].mean()
wr        = metrics.win_rate(result)

_wr_label = " / ".join(str(ul) for ul in wr.index)
_wr_value = " / ".join(f"{wr[ul]:.0%}" for ul in wr.index)
_period   = f"{_d0.strftime('%b %d')} – {_d1.strftime('%b %d')}" if _d0 and _d1 else "—"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Period", f"{_period} ({_n_days_run}d)")
col2.metric("Combined PnL", f"{combined:+,.1f} {_u()}")
col3.metric("Total Trades", f"{n_trades:,}")
col4.metric("Avg Rolls/Day", f"{avg_rolls:.0f}")
col5.metric(f"Win Rate {_wr_label}", _wr_value)

col_report, _ = st.columns([1, 4])
if col_report.button("Generate PDF Report"):
    with st.spinner("Building report…"):
        st.session_state["pdf_report"] = build_pdf_report(result, strategy_name, _period)

if "pdf_report" in st.session_state:
    st.download_button(
        label="Download PDF Report",
        data=st.session_state["pdf_report"],
        file_name=f"report_{strategy_name.replace(' ', '_').lower()}.pdf",
        mime="application/pdf",
    )

st.divider()

tab_ov, tab_pnl, tab_risk, tab_act, tab_detail, tab_cost, tab_log = st.tabs(
    ["Overview", "PnL", "Risk", "Activity", "Detail", "Costs", "Log"]
)

with tab_ov:
    st.subheader("Summary Statistics")
    st.dataframe(metrics.summary_table(result).style.format(precision=2), width="stretch")

    st.subheader("Daily PnL Calendar")
    st.dataframe(
        result.daily_pnl.style.format("{:.1f}").background_gradient(
            cmap="RdYlGn", axis=None,
            vmin=result.daily_pnl.min().min(), vmax=result.daily_pnl.max().max(),
        ),
        width="stretch",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plotly_pnl_histogram(result), width="stretch")
    with col2:
        st.plotly_chart(plotly_pnl_waterfall(result), width="stretch")

with tab_pnl:
    st.plotly_chart(plotly_cumulative_pnl(result), width="stretch")
    st.plotly_chart(plotly_daily_pnl(result), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plotly_morning_vs_afternoon(result), width="stretch")
    with col2:
        st.plotly_chart(plotly_pnl_by_weekday(result), width="stretch")

with tab_risk:
    st.dataframe(metrics.risk_table(result).style.format(precision=2), width="stretch")

    st.plotly_chart(plotly_sharpe_ci(result), width="stretch")
    st.plotly_chart(plotly_pnl_histogram(result, show_var=True), width="stretch")

    st.plotly_chart(plotly_drawdown(result), width="stretch")

    dd_table = metrics.drawdown_table(result)
    if not dd_table.empty:
        st.dataframe(dd_table, width="stretch")

    window = st.slider("Rolling Sharpe window (days)", 3, 10, 5)
    st.plotly_chart(plotly_rolling_sharpe(result, window), width="stretch")

    st.plotly_chart(plotly_expiry_day_comparison(result), width="stretch")

    if len(daily.columns) >= 2:
        corr = daily.iloc[:, 0].corr(daily.iloc[:, 1])
        st.metric(f"{daily.columns[0]} / {daily.columns[1]} PnL correlation", f"{corr:.3f}")
        st.plotly_chart(plotly_cross_underlier_scatter(result), width="stretch")

with tab_act:
    total_rolls = metrics.rolls_per_day(result)["rolls"].sum()
    if total_rolls == 0:
        st.info("No rolls — this strategy holds a fixed position.")
    else:
        st.plotly_chart(plotly_rolls_per_day(result), width="stretch")
        st.plotly_chart(plotly_roll_interval_dist(result), width="stretch")
    st.plotly_chart(plotly_avg_intraday_mtm(result), width="stretch")

with tab_detail:
    avail_dates = sorted({dr.date for dr in result.day_results})
    avail_uls   = sorted({dr.underlier for dr in result.day_results})

    col1, col2 = st.columns(2)
    sel_date = col1.selectbox("Date", avail_dates)
    sel_ul   = col2.selectbox("Underlier", avail_uls)

    dr = next((d for d in result.day_results if d.date == sel_date and d.underlier == sel_ul), None)

    if not dr:
        st.warning("No result for this combination.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final PnL", f"{dr.final_pnl:+.2f} {_u()}")
        col2.metric("Trades", len(dr.trades))
        col3.metric("Peak MTM", f"{dr.mtm.max():+.2f} {_u()}")
        col4.metric("Trough MTM", f"{dr.mtm.min():+.2f} {_u()}")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plotly_futures_through_day(dr), width="stretch")
        with col2:
            if not dr.holdings_log.empty:
                st.plotly_chart(plotly_leg_prices(dr), width="stretch")

        st.plotly_chart(plotly_futures_vs_strike(dr), width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plotly_intraday_mtm(dr), width="stretch")
        with col2:
            if not dr.holdings_log.empty:
                st.plotly_chart(plotly_strike_held(dr), width="stretch")

        st.plotly_chart(plotly_realized_staircase(dr), width="stretch")

with tab_cost:
    be_bps = metrics.breakeven_bps(result)
    if be_bps >= 0:
        st.metric("Break-even Cost", f"{be_bps:.2f} bps", help="Cost per trade at which net PnL = 0")
    else:
        st.metric(
            "Break-even Cost", "N/A (loss-making at zero cost)",
            help=f"Needs a {abs(be_bps):.2f} bps/trade subsidy to break even",
        )

    st.plotly_chart(plotly_cost_sensitivity(result), width="stretch")

    cs = metrics.cost_sensitivity(result)
    st.dataframe(cs.style.format(precision=2), width="stretch")
    st.caption(
        "Extra cost = total notional × bps ÷ 10,000, charged on top of any fees "
        "already in the run (set under Execution & risk). Notional uses the contract multiplier."
    )

with tab_log:
    trades = result.trades_df

    col1, col2, col3 = st.columns(3)
    f_ul = col1.multiselect(
        "Underlier", trades["underlier"].unique().tolist(),
        default=trades["underlier"].unique().tolist(),
    )
    f_side = col2.multiselect("Side", ["BUY", "SELL"], default=["BUY", "SELL"])
    f_date = col3.selectbox("Date", ["All"] + sorted(trades["date"].astype(str).unique().tolist()))

    filtered = trades[trades["underlier"].isin(f_ul) & trades["side"].isin(f_side)]
    if f_date != "All":
        filtered = filtered[filtered["date"].astype(str) == f_date]

    st.caption(f"Showing {len(filtered):,} of {len(trades):,} trades")
    st.dataframe(filtered, width="stretch", height=500)

    st.download_button(
        label="Download CSV",
        data=filtered.to_csv(index=False),
        file_name=f"trades_{strategy_name.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

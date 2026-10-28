# src/stratbt/dashboard/app.py
import importlib.util
import os
import shlex
from datetime import datetime

import pandas as pd
import streamlit as st

from stratbt.dashboard import charts
from stratbt.engine import BacktestEngine, BacktestResult
from stratbt.execution import ExecutionModel
from stratbt.providers import DataProvider, SyntheticProvider, YahooFinanceProvider
from stratbt.providers.yahoo import INTERVAL_MAX_DAYS
from stratbt.report import build_pdf_report, theme as T
from stratbt.report.data import build_report_data
from stratbt.risk import RiskLimits
from stratbt.strategy import registry

st.set_page_config(page_title="Strategy Backtester", layout="wide", initial_sidebar_state="expanded")

MAX_RUNS = 6
HAS_YAHOO = importlib.util.find_spec("yfinance") is not None
POPULAR = {
    "US ETFs": ["SPY", "QQQ", "IWM", "DIA"],
    "US stocks": ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"],
    "India (NSE)": ["^NSEI", "^NSEBANK", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"],
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SOURCES = ["Yahoo Finance", "Synthetic"]
# STRATBT_SOURCE=synthetic|yahoo picks the starting data source (e.g. offline containers, tests)
_env_source = {"synthetic": "Synthetic", "yahoo": "Yahoo Finance"}.get(os.environ.get("STRATBT_SOURCE", "").lower())
DEFAULT_SOURCE = _env_source or (SOURCES[0] if HAS_YAHOO else "Synthetic")

PRESETS = {
    "Next-second": ExecutionModel(),
    "Instant": ExecutionModel.instant(),
    "Realistic": ExecutionModel.realistic(),
}
PRESET_HELP = {
    "Next-second": "Fills one second after the decision, no costs. Removes look-ahead; nothing else.",
    "Instant": "Fills in the decision second at the mid price. Optimistic: reproduces the original behaviour.",
    "Realistic": "Next-second fills, 25 bps half-spread (1c floor) and $0.65 per contract.",
}
EXAMPLES = [
    ("Short strangle", "short_strangle", {"width": 3}, "Realistic",
     "Sell a call and a put three strikes out and hold. Premium collection with open-ended risk."),
    ("Iron condor", "iron_condor", {}, "Realistic",
     "The same idea with protective wings, so the loss is capped."),
    ("Rolling straddle", "atm_straddle", {}, "Realistic",
     "Trades thousands of times a day. Watch what costs do to it."),
]

if st.session_state.pop("_scroll_top", False):
    st.html("""<script>
      const d = window.parent.document;
      (d.querySelector('[data-testid="stMain"]') || d.querySelector('section.main') || d.documentElement).scrollTo({top: 0});
    </script>""", unsafe_allow_javascript=True)

st.html("""
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
  [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 650; }
  [data-testid="stMetricLabel"] p { font-size: .72rem; letter-spacing: .04em; text-transform: uppercase; opacity: .75; }
  button[data-baseweb="tab"] { font-size: .95rem; }
  h1 { font-weight: 700; letter-spacing: -.02em; }
  [data-testid="stSidebar"] [data-testid="stExpander"] { border-color: rgba(128,128,128,.25); }
</style>
""")


# ---------------------------------------------------------------------------
# data + engine helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Preparing market data…", ttl=3600)
def _make_provider(kind: str, tickers: tuple, interval: str, lookback: int, iv_premium: float,
                   expiry_weekday: int, multiplier: float, seed: int, n_days: int) -> DataProvider:
    if kind == "Synthetic":
        return SyntheticProvider(n_days=n_days, seed=seed)
    return YahooFinanceProvider(list(tickers), interval=interval, lookback_days=lookback, iv_premium=iv_premium,
                                expiry_weekday=expiry_weekday, multiplier=multiplier or None)


@st.cache_data(show_spinner=False)
def _underlying(_provider: DataProvider, provider_key: str, day, underlier: str) -> pd.Series:
    return _provider.underlying_prices(day, underlier)


def _apply_preset() -> None:
    m = PRESETS[st.session_state["preset"] or "Next-second"]
    st.session_state.update(delay_s=m.delay_s, spread_bps=m.half_spread_bps, commission=m.commission_per_contract)


def _money(v, unit, **kw) -> str:
    return T.fmoney(v, unit, **kw)


def _esc(s: str) -> str:
    """Streamlit markdown renders $...$ as LaTeX; money strings must be escaped."""
    return s.replace("$", "\\$")


def _cli_command(strategy_key, params, src, underliers, start, end, execution, risk) -> str:
    parts = ["stratbt", "run", strategy_key]
    if src["kind"] == "Synthetic":
        parts += ["--provider", "synthetic", "--seed", str(src["seed"]), "--days", str(src["n_days"])]
    else:
        parts += ["--provider", "yahoo", "--tickers", *src["tickers"], "--interval", src["interval"],
                  "--lookback", str(src["lookback"]), "--iv-premium", f"{src['iv_premium']:g}",
                  "--expiry-weekday", str(src["expiry_weekday"])]
        if src["multiplier"]:
            parts += ["--multiplier", f"{src['multiplier']:g}"]
    parts += ["--underliers", *underliers]
    if start:
        parts += ["--start", start.isoformat()]
    if end:
        parts += ["--end", end.isoformat()]
    parts += [f"--param={k}={v:g}" for k, v in params.items()]
    parts += ["--delay-s", str(execution.delay_s)]
    if execution.half_spread_bps:
        parts += ["--half-spread-bps", f"{execution.half_spread_bps:g}"]
    if execution.min_half_spread:
        parts += ["--min-half-spread", f"{execution.min_half_spread:g}"]
    if execution.commission_per_contract:
        parts += ["--commission", f"{execution.commission_per_contract:g}"]
    if risk.max_daily_loss:
        parts += ["--max-daily-loss", f"{risk.max_daily_loss:g}"]
    if risk.max_open_contracts:
        parts += ["--max-contracts", str(risk.max_open_contracts)]
    return " ".join(shlex.quote(p) for p in parts)


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

st.session_state.setdefault("runs", [])
st.session_state.setdefault("run_counter", 0)
st.session_state.setdefault("delay_s", 1)
st.session_state.setdefault("spread_bps", 0.0)
st.session_state.setdefault("commission", 0.0)

sb = st.sidebar
sb.markdown("### Strategy Backtester")
run_slot = sb.container()

with sb.expander("1 · Market data", expanded=True):
    kind = st.segmented_control("Source", SOURCES, default=DEFAULT_SOURCE,
                                key="source", label_visibility="collapsed") or SOURCES[-1]
    src = dict(kind=kind, tickers=(), interval="5m", lookback=30, iv_premium=0.10, expiry_weekday=4,
               multiplier=0.0, seed=7, n_days=21)
    if kind == "Synthetic":
        st.caption("A seeded random market with Black–Scholes options. No download needed; "
                   "results say nothing about real markets.")
        c1, c2 = st.columns(2)
        src["seed"] = int(c1.number_input("Seed", 0, 99_999, 7, key="seed", help="Same seed, same market."))
        src["n_days"] = int(c2.number_input("Days", 5, 120, 21, key="ndays"))
    elif not HAS_YAHOO:
        st.error("Yahoo Finance support is not installed.  `pip install 'strat-backtester[data]'`")
        st.stop()
    else:
        options = [t for group in POPULAR.values() for t in group]
        src["tickers"] = tuple(st.multiselect(
            "Tickers", options, default=["SPY"], key="tickers", accept_new_options=True,
            help="Any Yahoo Finance symbol: US (SPY, AAPL), India (^NSEI, RELIANCE.NS), and more. "
                 "Use one market per run: tickers must share trading hours."))
        c1, c2 = st.columns(2)
        src["interval"] = c1.selectbox("Bar size", ["1m", "5m", "15m", "30m", "60m"], index=1, key="interval")
        cap = INTERVAL_MAX_DAYS[src["interval"]]
        src["lookback"] = int(c2.number_input("Days back", 5, cap, min(30, cap), key=f"lookback_{src['interval']}",
                                              help=f"Yahoo keeps about {cap} days of {src['interval']} bars."))
        if st.toggle("Option pricing assumptions", key="adv_pricing"):
            src["iv_premium"] = st.slider("Implied vol above realised", 0, 50, 10, 5, key="iv_prem",
                                          help="Implied vol = trailing realised vol × (1 + this).") / 100
            src["expiry_weekday"] = WEEKDAYS.index(st.selectbox("Weekly expiry", WEEKDAYS, index=4, key="expday"))
            src["multiplier"] = float(st.number_input("Contract multiplier (0 = auto)", 0.0, 100_000.0, 0.0, key="mult",
                                                      help="Auto: 100 for USD underlyings, 1 (index points) otherwise."))
        st.caption("Underlying prices are **real** Yahoo bars. Option prices are **modelled** (Black–Scholes on "
                   "trailing realised volatility): Yahoo has no historical option quotes.")
    try:
        if kind == "Yahoo Finance" and not src["tickers"]:
            st.info("Pick at least one ticker.")
            st.stop()
        provider = _make_provider(kind, src["tickers"], src["interval"], src["lookback"], src["iv_premium"],
                                  src["expiry_weekday"], src["multiplier"], src["seed"], src["n_days"])
        provider_key = repr(src)
    except (ValueError, ImportError, OSError) as e:
        st.error(str(e))
        st.stop()
    all_days = provider.trading_days
    st.caption(f"{len(all_days)} trading days · {provider.session_open:%H:%M}–{provider.session_close:%H:%M} · PnL in {provider.unit}")

with sb.expander("2 · Strategy", expanded=True):
    specs = registry.available()
    strategy_key = st.selectbox("Strategy", list(specs), format_func=lambda k: specs[k].label, key="strategy",
                                label_visibility="collapsed")
    spec = specs[strategy_key]
    st.caption(spec.description)
    strategy_params: dict = {}
    for ps in spec.params:
        cast = int if ps.is_int else float
        strategy_params[ps.name] = st.slider(ps.label or ps.name, cast(ps.min), cast(ps.max), cast(ps.default),
                                             cast(ps.step), key=f"param_{strategy_key}_{ps.name}")

with sb.expander("3 · Universe & period", expanded=False):
    # keyed by the data source so the selection resets when the available underliers change
    underliers = st.multiselect("Underliers", provider.underliers, default=provider.underliers,
                                key=f"underliers_{abs(hash(provider_key)) % 10**8}")
    if len(all_days) > 1:
        start_date, end_date = st.select_slider("Period", all_days, value=(all_days[0], all_days[-1]),
                                                format_func=lambda d: d.strftime("%b %d"), key="period")
    else:
        start_date = end_date = all_days[0] if all_days else None
    selected_days = [d for d in all_days if start_date and start_date <= d <= end_date]
    st.caption(f"{len(selected_days)} days selected")

with sb.expander("4 · Execution & risk", expanded=False):
    st.segmented_control("Fill model", list(PRESETS), default="Next-second", key="preset",
                         on_change=_apply_preset, label_visibility="collapsed")
    st.caption(_esc(PRESET_HELP.get(st.session_state.get("preset") or "Next-second", "")))
    delay_s = st.number_input("Fill delay (s)", 0, 30, key="delay_s",
                              help="Seconds between decision and fill. 0 = same-second (optimistic). "
                                   "While orders are in flight the strategy is not asked for new ones.")
    spread_bps = st.number_input("Half-spread (bps of price)", 0.0, 1000.0, step=5.0, key="spread_bps")
    commission = st.number_input("Commission per contract", 0.0, 100.0, step=0.05, key="commission")
    st.divider()
    max_loss = st.number_input("Max daily loss (0 = off)", 0.0, 1e9, 0.0, 100.0, key="max_loss",
                               help="Flatten everything and stop trading for the day once net loss reaches this.")
    max_open = st.number_input("Max open contracts (0 = off)", 0, 10_000, 0, key="max_open")

base_exec = PRESETS.get(st.session_state.get("preset") or "Next-second", ExecutionModel())
execution_ui = ExecutionModel(
    delay_s=int(delay_s), half_spread_bps=float(spread_bps),
    min_half_spread=base_exec.min_half_spread if spread_bps > 0 else 0.0,
    commission_per_contract=float(commission), commission_bps=base_exec.commission_bps,
)
risk_ui = RiskLimits(max_daily_loss=max_loss or None, max_open_contracts=int(max_open) or None)

can_run = bool(underliers) and bool(selected_days)
run_clicked = run_slot.button("Run backtest", type="primary", width="stretch", disabled=not can_run, key="run")
run_slot.caption(f"{spec.label} · {', '.join(underliers)} · {len(selected_days)} days" if can_run
                 else "Choose at least one underlier and a period to run.")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _run(strategy_key_: str, params: dict, execution: ExecutionModel, risk: RiskLimits) -> None:
    spec_ = specs[strategy_key_]
    engine = BacktestEngine(provider, spec_.factory(provider, **params), execution=execution, risk=risk)
    total = len(selected_days) * len(underliers)
    results, skipped = [], []
    with st.status(f"Running {spec_.label}…", expanded=True) as status:
        bar = st.progress(0.0)
        for ul in underliers:
            for day in selected_days:
                done = len(results) + len(skipped)
                bar.progress(done / max(total, 1), text=f"{ul} · {day:%b %d}   ({done}/{total})")
                try:
                    results.append(engine.run_day(day, ul))
                except ValueError:
                    skipped.append((ul, day))
        bar.progress(1.0, text="Done")
        if not results:
            status.update(label="No results", state="error")
            st.error("No results produced. Check the underlier and date selection.")
            return
        meta = engine.describe() | dict(strategy=spec_.label, strategy_key=spec_.key,
                                        params=spec_.resolve(**params), allow_short=spec_.cls.allow_short,
                                        unlimited_risk=spec_.cls.unlimited_risk)
        result = BacktestResult(results, unit=provider.unit, meta=meta)
        data = build_report_data(result)
        st.session_state["run_counter"] += 1
        n = st.session_state["run_counter"]
        run = dict(id=n, label=f"#{n} {spec_.label}", result=result, data=data, provider=provider,
                   provider_key=provider_key, created=datetime.now(), skipped=skipped,
                   cli=_cli_command(strategy_key_, params, src, underliers,
                                    selected_days[0], selected_days[-1], execution, risk))
        st.session_state["runs"] = ([run] + st.session_state["runs"])[:MAX_RUNS]
        st.session_state["active_run"] = n
        status.update(label=f"{spec_.label}: done", state="complete", expanded=False)
    if skipped:
        st.session_state["_flash"] = f"Skipped {len(skipped)} day/underlier combinations with missing data."
    # Re-render from the top: otherwise the progress block stays above the results and the
    # page keeps its scroll offset, hiding the title and header.
    st.session_state["_scroll_top"] = True
    st.rerun()


quick = st.session_state.pop("_quick", None)
if quick and can_run:
    _run(quick[0], specs[quick[0]].resolve(**quick[1]), PRESETS[quick[2]], risk_ui)
elif run_clicked and can_run:
    _run(strategy_key, strategy_params, execution_ui, risk_ui)


# ---------------------------------------------------------------------------
# landing
# ---------------------------------------------------------------------------

def _go(example) -> None:
    st.session_state["_quick"] = (example[1], example[2], example[3])


def landing() -> None:
    st.title("Strategy Backtester")
    st.caption("Test options strategies second by second, with realistic fills, costs and risk limits.")
    st.markdown("&nbsp;")
    c = st.columns(3)
    steps = [("1 · Choose data", "Real prices for NSE, NASDAQ, NYSE and more via Yahoo Finance, or a built-in synthetic market that works offline."),
             ("2 · Choose a strategy", "Straddles, strangles and iron condors are built in; add your own in a few lines."),
             ("3 · Run and read", "Plain-language findings, risk, costs, a PDF tearsheet, and a command to reproduce it.")]
    for col, (h, t) in zip(c, steps):
        with col.container(border=True):
            st.markdown(f"**{h}**")
            st.caption(t)

    st.subheader("Try an example")
    st.caption(f"Runs on the data currently selected ({provider.name}, {len(selected_days)} days)."
               + ("" if provider.name == "synthetic" else " Option prices are modelled; the underlying is real."))
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        with col.container(border=True):
            st.markdown(f"**{ex[0]}**")
            st.caption(ex[4])
            st.button("Run this", key=f"ex_{ex[1]}", on_click=_go, args=(ex,), disabled=not can_run, width="stretch")

    st.subheader("Available strategies")
    st.dataframe(pd.DataFrame([{"Strategy": s.label, "What it does": s.description,
                                "Parameters": ", ".join(p.name for p in s.params) or "—"}
                               for s in specs.values()]), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

def results_view(runs: list) -> None:
    ids = [r["id"] for r in runs]
    active_id = st.session_state.get("active_run", ids[0])
    run = next((r for r in runs if r["id"] == active_id), runs[0])
    d, result = run["data"], run["result"]
    if flash := st.session_state.pop("_flash", None):
        st.warning(flash)
    label_of = {r["id"]: r["label"] for r in runs}

    head, pick, export = st.columns([5, 3, 1.4], vertical_alignment="bottom")
    head.title(d.strategy_name)
    if len(runs) > 1:
        sel = pick.selectbox("Run", ids, index=ids.index(run["id"]), format_func=label_of.get, key="run_picker")
        if sel != run["id"]:
            st.session_state["active_run"] = sel
            st.rerun()
    with export.popover("Export", width="stretch"):
        st.caption("Reproduce this run from a terminal:")
        st.code(run["cli"], language="bash")
        if st.button("Build PDF report", key=f"pdf_{run['id']}", width="stretch"):
            with st.spinner("Building report…"):
                run["pdf"] = build_pdf_report(result)
        if run.get("pdf"):
            st.download_button("Download PDF", run["pdf"], file_name=f"report_{d.strategy_name.replace(' ', '_').lower()}.pdf",
                               mime="application/pdf", width="stretch", key=f"dlpdf_{run['id']}")
        st.download_button("Download trades (CSV)", result.trades_df.to_csv(index=False),
                           file_name=f"trades_{run['id']}.csv", mime="text/csv", width="stretch", key=f"dlcsv_{run['id']}")
    ex = result.meta.get("execution", {})
    fill = "same-second fills" if ex.get("delay_s") == 0 else f"fills +{ex.get('delay_s', 1)}s"
    costs = bool(ex.get("half_spread_bps") or ex.get("commission_per_contract") or ex.get("commission_bps"))
    st.caption(f"{d.period_label}  ·  {d.n_days} days  ·  {', '.join(d.underliers)}  ·  {run['provider'].name} data  ·  "
               f"{fill}, {'with trading costs' if costs else 'no trading costs'}")

    with st.container(border=True):
        st.markdown("**What happened**")
        for line in d.observations:
            st.markdown(f"- {_esc(line)}")

    u = d.unit
    lo, hi = d.sharpe_ci
    cost = d.fees + d.slippage
    flat = dict(delta_color="off", delta_arrow="off")
    k = st.columns(6)
    k[0].metric("Net PnL", _money(d.net_pnl, u, signed=True),
                delta=_esc(f"gross {_money(d.gross_pnl, u, signed=True)}") if cost else None, **flat)
    k[1].metric("Sharpe", f"{d.sharpe:.2f}", delta=f"95% CI {lo:.1f} to {hi:.1f}" if lo == lo else None,
                help="Annualised, risk-free = 0. Interval is a bootstrap over days.", **flat)
    k[2].metric("Max drawdown", _money(d.max_dd, u), delta=f"longest {d.max_dd_days} days", **flat)
    k[3].metric("Day win rate", f"{d.win_rate:.0%}", delta=f"profit factor {d.profit_factor:.2f}", **flat)
    k[4].metric("Trades", f"{d.n_trades:,}", delta=f"{d.n_trades / max(d.n_days, 1):,.0f} per day", **flat)
    k[5].metric("Costs paid", _money(cost, u),
                delta=_esc(f"fees {T.fcompact(d.fees, u)} + slippage {T.fcompact(d.slippage, u)}"), **flat)

    names = ["Overview", "Risk", "Behaviour", "Costs", "Days & trades", "Drill-down"] + (["Compare"] if len(runs) > 1 else [])
    tabs = st.tabs(names)

    with tabs[0]:
        st.markdown("##### Cumulative net PnL")
        st.plotly_chart(charts.equity(d), theme="streamlit", width="stretch")
        a, b = st.columns(2)
        with a:
            st.markdown("##### Drawdown from running peak")
            st.plotly_chart(charts.drawdown(d), theme="streamlit", width="stretch")
        with b:
            st.markdown("##### Daily net PnL")
            st.plotly_chart(charts.daily_bars(d), theme="streamlit", width="stretch")
        st.markdown("##### Performance statistics")
        st.dataframe(pd.DataFrame(d.stats_rows, columns=["Metric"] + d.stats_cols).set_index("Metric"), width="stretch")

    with tabs[1]:
        a, b = st.columns(2)
        with a:
            st.markdown("##### Risk metrics")
            st.caption(d.risk_note)
            st.dataframe(pd.DataFrame(d.risk_rows, columns=["Metric"] + d.stats_cols).set_index("Metric"), width="stretch")
            st.markdown("##### Largest drawdown episodes")
            if d.drawdown_rows:
                st.dataframe(pd.DataFrame(d.drawdown_rows, columns=["Series", "Trough", "Depth", "Recovered"]),
                             hide_index=True, width="stretch")
            else:
                st.caption("No drawdowns.")
        with b:
            st.markdown("##### Sharpe ratio with 95% interval")
            st.plotly_chart(charts.sharpe_ci(d), theme="streamlit", width="stretch")
            st.markdown("##### Distribution of daily PnL")
            st.plotly_chart(charts.histogram(d), theme="streamlit", width="stretch")

    with tabs[2]:
        a, b = st.columns(2)
        with a:
            st.markdown("##### Average intraday PnL path")
            st.plotly_chart(charts.intraday(d), theme="streamlit", width="stretch")
            st.markdown("##### Average PnL by weekday")
            st.plotly_chart(charts.weekday(d), theme="streamlit", width="stretch")
        with b:
            st.markdown("##### Morning vs afternoon")
            st.plotly_chart(charts.am_pm(d), theme="streamlit", width="stretch")
            st.markdown("##### Expiry day vs other days")
            st.plotly_chart(charts.expiry(d), theme="streamlit", width="stretch")

    with tabs[3]:
        a, b = st.columns(2)
        with a:
            st.markdown("##### From gross to net PnL")
            st.plotly_chart(charts.cost_waterfall(d), theme="streamlit", width="stretch")
        with b:
            st.markdown("##### Sensitivity to extra costs")
            st.plotly_chart(charts.cost_sensitivity(d), theme="streamlit", width="stretch")
        a, b = st.columns(2)
        with a:
            st.markdown("##### Cost breakdown")
            st.dataframe(pd.DataFrame(d.cost_rows, columns=["", "Gross", "Slippage", "Fees", "Net"]).set_index(""),
                         width="stretch")
            st.caption("Gross = mid-price fills with no fees. Slippage = spread paid versus mid. "
                       "Both are already inside net PnL.")
        with b:
            st.markdown("##### Trading activity")
            st.dataframe(pd.DataFrame(d.activity_rows, columns=["", "Value"]).set_index(""), width="stretch")

    with tabs[4]:
        st.markdown("##### Daily net PnL")
        frame = d.daily.copy()
        if len(d.underliers) > 1:
            frame["Combined"] = d.combined
        frame["Cumulative"] = d.cum_combined
        cols = [c for c in frame.columns if c != "Cumulative"]
        bound = max(float(frame[cols].abs().to_numpy().max()), 1e-9)
        st.dataframe(frame.style.background_gradient(cmap="RdBu", vmin=-bound, vmax=bound, subset=cols, axis=None)
                     .format("{:,.0f}"), width="stretch", height=min(40 + 35 * len(frame), 520))
        st.markdown("##### Trade log")
        trades = result.trades_df
        if trades.empty:
            st.caption("No trades.")
        else:
            uls = sorted(trades["underlier"].unique())
            c1, c2, c3 = st.columns(3)
            f_ul = c1.multiselect("Underlier", uls, default=uls, key="f_ul")
            f_side = c2.multiselect("Side", ["BUY", "SELL"], default=["BUY", "SELL"], key="f_side")
            f_date = c3.selectbox("Date", ["All"] + sorted(trades["date"].astype(str).unique()), key="f_date")
            view = trades[trades["underlier"].isin(f_ul) & trades["side"].isin(f_side)]
            if f_date != "All":
                view = view[view["date"].astype(str) == f_date]
            st.caption(f"Showing {min(len(view), 2000):,} of {len(view):,} matching trades ({len(trades):,} total)")
            st.dataframe(view.head(2000), hide_index=True, width="stretch", height=380,
                         column_config={"price": st.column_config.NumberColumn(format="%.2f"),
                                        "fee": st.column_config.NumberColumn(format="%.2f"),
                                        "slippage": st.column_config.NumberColumn(format="%.2f")})

    with tabs[5]:
        c1, c2 = st.columns(2)
        sel_date = c1.selectbox("Date", sorted({x.date for x in result.day_results}), key="dd_date")
        sel_ul = c2.selectbox("Underlier", sorted({x.underlier for x in result.day_results}), key="dd_ul")
        dr = next((x for x in result.day_results if x.date == sel_date and x.underlier == sel_ul), None)
        if dr is None:
            st.warning("No result for this combination.")
        else:
            m = st.columns(5)
            m[0].metric("Net PnL", _money(dr.final_pnl, u, signed=True))
            m[1].metric("Trades", len(dr.trades))
            m[2].metric("Peak", _money(dr.mtm.max(), u, signed=True))
            m[3].metric("Trough", _money(dr.mtm.min(), u, signed=True))
            m[4].metric("Costs", _money(dr.fees + dr.slippage, u))
            if dr.rejections:
                st.warning(f"{len(dr.rejections)} orders rejected: " + "; ".join(sorted({r.reason for r in dr.rejections})))
            underlying = _underlying(run["provider"], run["provider_key"], dr.date, dr.underlier)
            a, b = st.columns(2)
            with a:
                st.markdown("##### Net mark-to-market")
                st.plotly_chart(charts.day_mtm(dr, u), theme="streamlit", width="stretch")
                st.markdown("##### Underlying and strike held")
                st.plotly_chart(charts.day_price_strike(dr, underlying, u), theme="streamlit", width="stretch")
            with b:
                st.markdown("##### MTM vs realised")
                st.plotly_chart(charts.day_realized(dr, u), theme="streamlit", width="stretch")
                if not dr.holdings_log.empty:
                    st.markdown("##### Option prices of the position")
                    st.plotly_chart(charts.day_legs(dr, u), theme="streamlit", width="stretch")

    if len(runs) > 1:
        with tabs[6]:
            picks = st.multiselect("Runs to compare", ids, default=ids[:3], format_func=label_of.get, key="cmp")
            chosen = [r for r in runs if r["id"] in picks]
            if chosen:
                if len({r["provider_key"] for r in chosen}) > 1:
                    st.warning("These runs use different market data; compare with care.")
                st.plotly_chart(charts.compare_equity(chosen), theme="streamlit", width="stretch")
                rows = []
                for r in chosen:
                    x = r["data"]
                    rows.append({"Run": r["label"], "Net PnL": _money(x.net_pnl, x.unit, signed=True),
                                 "Sharpe": f"{x.sharpe:.2f}", "Max drawdown": _money(x.max_dd, x.unit),
                                 "Day win rate": f"{x.win_rate:.0%}", "Trades": f"{x.n_trades:,}",
                                 "Costs": _money(x.fees + x.slippage, x.unit),
                                 "Execution": dict(x.config_rows).get("Execution", "")})
                st.dataframe(pd.DataFrame(rows).set_index("Run"), width="stretch")
    if len(runs) > 0 and st.sidebar.button("Clear run history", key="clear"):
        st.session_state["runs"] = []
        st.rerun()


runs = st.session_state["runs"]
if runs:
    results_view(runs)
else:
    landing()

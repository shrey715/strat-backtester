# src/stratbt/metrics.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from stratbt.engine import BacktestResult

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

def _daily_pnl(result: BacktestResult, underlier: str) -> pd.Series:
    """
    Parameters
    ----------
    result : BacktestResult
    underlier : str

    Returns
    -------
    pd.Series
        Daily PnL for one underlier, indexed by date, with NaN days dropped.
    """
    return result.daily_pnl[underlier].dropna()

def _round_trips(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair each BUY with its matching SELL for the same instrument on the same
    day, using FIFO order (first BUY matches first SELL, etc.) — the
    standard lot-matching convention, and exactly correct for strategies
    that never scale into a position before closing it. Works for shorts
    too (the SELL simply comes first); pnl = (sell - buy) * qty either way.

    Parameters
    ----------
    trades_df : pd.DataFrame
        As returned by BacktestResult.trades_df.

    Returns
    -------
    pd.DataFrame
        One row per round trip: date, underlier, instrument, buy_time,
        sell_time, buy_price, sell_price, qty, pnl, holding_seconds. Empty
        (no columns) if there are no complete round trips.
    """
    rows = []
    for (d, underlier, instrument), group in trades_df.groupby(["date", "underlier", "instrument"]):
        buys  = group[group["side"] == "BUY"].reset_index(drop=True)
        sells = group[group["side"] == "SELL"].reset_index(drop=True)
        for i in range(min(len(buys), len(sells))):
            buy  = buys.iloc[i]
            sell = sells.iloc[i]
            rows.append({
                "date":            d,
                "underlier":       underlier,
                "instrument":      instrument,
                "buy_time":        buy["timestamp"],
                "sell_time":       sell["timestamp"],
                "buy_price":       buy["price"],
                "sell_price":      sell["price"],
                "qty":             buy["qty"],
                "pnl":             (sell["price"] - buy["price"]) * buy["qty"],
                "holding_seconds": abs((sell["timestamp"] - buy["timestamp"]).total_seconds()),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Scalar metric functions — each returns a single number or small table
# ---------------------------------------------------------------------------

def total_pnl(result: BacktestResult) -> pd.Series:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Total PnL per underlier, summed across all days.
    """
    return result.daily_pnl.sum()

def mean_daily_pnl(result: BacktestResult) -> pd.Series:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Average daily PnL per underlier.
    """
    return result.daily_pnl.mean()

def std_daily_pnl(result: BacktestResult) -> pd.Series:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Standard deviation of daily PnL per underlier.
    """
    return result.daily_pnl.std()

def sharpe_ratio(result: BacktestResult) -> pd.Series:
    """
    Annualized Sharpe ratio, assuming zero risk-free rate:
    Sharpe = (mean_daily / std_daily) * sqrt(252). sqrt(252) annualizes
    because the mean scales linearly with time but the std scales with
    sqrt(time).

    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Per underlier. Note: with only ~21 trading days per month, this
        estimate carries substantial sampling noise — see
        sharpe_confidence_interval() for a bootstrap CI that quantifies it.
    """
    mean = result.daily_pnl.mean()
    std  = result.daily_pnl.std()
    return (mean / std) * np.sqrt(252)

def sortino_ratio(result: BacktestResult) -> pd.Series:
    """
    Like Sharpe but normalizes by downside volatility only, so it doesn't
    penalize upside variance: Sortino = (mean_daily / downside_std) * sqrt(252).

    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Per underlier. downside_std is the std of negative-PnL days only
        (ddof=1, denominator = count of negative days) — a common
        simplified variant, not the strict textbook semi-deviation
        (which normalizes by the full sample size instead).
    """
    mean         = result.daily_pnl.mean()
    negative_days = result.daily_pnl[result.daily_pnl < 0]
    downside_std = negative_days.std()
    return (mean / downside_std) * np.sqrt(252)

def rolling_sharpe(
    result: BacktestResult,
    window: int = 5,
) -> pd.DataFrame:
    """
    Sharpe ratio recomputed over a trailing window of trading days, to show
    whether performance is stable or deteriorating rather than a single
    whole-period number.

    Parameters
    ----------
    result : BacktestResult
    window : int, default 5
        Trailing window size in trading days.

    Returns
    -------
    pd.DataFrame
        Same shape as daily_pnl; the first (window - 1) rows per underlier
        are NaN (not enough history yet for a full window).
    """
    daily = result.daily_pnl.fillna(0)
    mean  = daily.rolling(window).mean()
    std   = daily.rolling(window).std()
    return (mean / std) * np.sqrt(252)

def calmar_ratio(result: BacktestResult) -> pd.Series:
    """
    Annualized return per unit of max drawdown: Calmar = (mean_daily * 252)
    / max_drawdown. Measures return earned per unit of worst-case loss.

    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Per underlier. NaN where max_drawdown is zero (no drawdown at all).
    """
    annualized = result.daily_pnl.mean() * 252
    mdd        = max_drawdown(result)
    return annualized / mdd.replace(0, np.nan)

def max_drawdown(result: BacktestResult) -> pd.Series:
    """
    The largest peak-to-trough decline in cumulative PnL: how much you'd
    have lost if you'd started at the worst possible moment.

    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Per underlier, as a positive magnitude (peak - trough).
    """
    cum  = result.daily_pnl.fillna(0).cumsum()
    peak = cum.cummax()
    return (peak - cum).max()

def max_drawdown_duration(result: BacktestResult) -> pd.Series:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Per underlier: the longest run of consecutive days spent below the
        running equity peak. A long duration means the strategy is slow to
        recover from losses, independent of how deep the drawdown got.
    """
    results = {}
    for col in result.daily_pnl.columns:
        cum      = result.daily_pnl[col].fillna(0).cumsum()
        peak     = cum.cummax()
        in_dd    = (cum < peak).astype(int)
        # count consecutive days below peak
        groups   = (in_dd != in_dd.shift()).cumsum()
        duration = in_dd.groupby(groups).sum()
        results[col] = int(duration.max())
    return pd.Series(results)

def win_rate(result: BacktestResult) -> pd.Series:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Fraction of trading days with strictly positive PnL, per underlier.
    """
    return (result.daily_pnl > 0).mean()

def profit_factor(result: BacktestResult) -> pd.Series:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Per underlier: sum of winning days / sum(abs(losing days)). > 1 is
        profitable overall (2.0 = gained 2 points for every 1 lost).
        +inf if there are no losing days at all.
    """
    results = {}
    for col in result.daily_pnl.columns:
        pnl    = result.daily_pnl[col].dropna()
        gains  = pnl[pnl > 0].sum()
        losses = pnl[pnl < 0].abs().sum()
        results[col] = gains / losses if losses > 0 else float("inf")
    return pd.Series(results)

def avg_holding_time(result: BacktestResult) -> pd.Series:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Mean holding time per round trip, in seconds, per underlier. Empty
        Series if there are no complete round trips. For a high-frequency
        rolling strategy this will be very short.
    """
    rt = _round_trips(result.trades_df)
    if rt.empty:
        return pd.Series(dtype=float)
    return rt.groupby("underlier")["holding_seconds"].mean()

def rolls_per_day(result: BacktestResult) -> pd.DataFrame:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.DataFrame
        One row per (date, underlier): date, underlier, trades, rolls. A
        roll is a strike change between two CONSECUTIVE (exactly 1 second
        apart) holdings-log rows — a re-entry after going flat is not
        counted as a roll, even if the new strike differs from the last one
        held. Correct for any strategy, including ones that go flat mid-day.
    """
    rows = []
    for dr in result.day_results:
        if dr.holdings_log.empty:
            n_rolls = 0
        else:
            log = dr.holdings_log.reset_index()
            time_gap = log["timestamp"].diff().dt.total_seconds()
            is_roll = (log["strike"] != log["strike"].shift(1)) & (time_gap == 1)
            n_rolls = int(is_roll.sum())

        rows.append({
            "date":      dr.date,
            "underlier": dr.underlier,
            "trades":    len(dr.trades),
            "rolls":     n_rolls,
        })
    return pd.DataFrame(rows)

def cost_sensitivity(
    result:    BacktestResult,
    bps_range: list[float] | None = None,
) -> pd.DataFrame:
    """
    PnL impact of transaction costs at several assumed cost levels, since
    the backtest itself runs at zero cost.

    Parameters
    ----------
    result : BacktestResult
    bps_range : list[float], optional
        Cost levels to evaluate, in basis points of option premium per
        transaction, charged on top of any fees already in the results. Defaults to [0, 1, 2, 5, 10].

    Returns
    -------
    pd.DataFrame
        One row per bps level: cost_bps, total_notional, cost_drag,
        gross_pnl, net_pnl. cost_drag = total_notional * bps / 10,000,
        applied once per transaction (both entry and exit legs are each
        charged, matching how total_notional sums over every trade row).
    """
    if bps_range is None:
        bps_range = [0, 1, 2, 5, 10]

    total_notional = result.notional_df["notional"].sum()
    gross          = result.daily_pnl.sum().sum()

    rows = []
    for bps in bps_range:
        drag = total_notional * bps / 10_000
        rows.append({
            "cost_bps":       bps,
            "total_notional": round(total_notional, 2),
            "cost_drag":      round(drag, 2),
            "gross_pnl":      round(gross, 2),
            "net_pnl":        round(gross - drag, 2),
        })
    return pd.DataFrame(rows)

def cost_breakdown(result: BacktestResult) -> pd.DataFrame:
    """
    Where the money went, per underlier plus a Total row, in the result's
    unit: gross PnL (what mid-price fills with no fees would have made),
    minus slippage (spread paid), minus fees, equals net PnL.

    Returns
    -------
    pd.DataFrame
        Index = underlier (+ "Total"); columns gross, slippage, fees, net.
    """
    rows = {}
    for dr in result.day_results:
        r = rows.setdefault(dr.underlier, {"net": 0.0, "slippage": 0.0, "fees": 0.0})
        r["net"] += dr.final_pnl
        r["slippage"] += dr.slippage
        r["fees"] += dr.fees
    if not rows:
        return pd.DataFrame(columns=["gross", "slippage", "fees", "net"])
    df = pd.DataFrame(rows).T
    df["gross"] = df["net"] + df["slippage"] + df["fees"]
    df = df[["gross", "slippage", "fees", "net"]]
    df.loc["Total"] = df.sum()
    return df

def summary_table(result: BacktestResult) -> pd.DataFrame:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.DataFrame
        One row per underlier (indexed by underlier), with total_pnl,
        mean_daily_pnl, std_daily_pnl, best_day, worst_day, sharpe, sortino,
        calmar, max_drawdown, max_dd_duration, win_rate, profit_factor,
        avg_holding_secs, total_trades. Intended as the first thing shown
        in a results notebook or report.
    """
    rt = _round_trips(result.trades_df)

    rows = []
    for col in result.daily_pnl.columns:
        pnl      = _daily_pnl(result, col)
        rt_col   = rt[rt["underlier"] == col] if not rt.empty else pd.DataFrame()
        avg_hold = rt_col["holding_seconds"].mean() if not rt_col.empty else 0.0

        rows.append({
            "underlier":           col,
            "total_pnl":           round(pnl.sum(), 2),
            "mean_daily_pnl":      round(pnl.mean(), 2),
            "std_daily_pnl":       round(pnl.std(), 2),
            "best_day":            round(pnl.max(), 2),
            "worst_day":           round(pnl.min(), 2),
            "sharpe":              round(sharpe_ratio(result)[col], 3),
            "sortino":             round(sortino_ratio(result)[col], 3),
            "calmar":              round(calmar_ratio(result)[col], 3),
            "max_drawdown":        round(max_drawdown(result)[col], 2),
            "max_dd_duration":     max_drawdown_duration(result)[col],
            "win_rate":            f"{100 * (pnl > 0).mean():.1f}%",
            "profit_factor":       round(profit_factor(result)[col], 2),
            "avg_holding_secs":    round(avg_hold, 1),
            "total_trades":        len(result.trades_df[result.trades_df["underlier"] == col]),
        })
    return pd.DataFrame(rows).set_index("underlier")

# ---------------------------------------------------------------------------
# Plot functions — each returns a Figure
# ---------------------------------------------------------------------------
def plot_cumulative_pnl(result: BacktestResult) -> Figure:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    Figure
        Cumulative PnL over the month: one line per underlier plus a
        combined (dashed) line — the combined line is what a portfolio
        holding all underliers together would show.
    """
    daily    = result.daily_pnl.fillna(0)
    cum      = daily.cumsum()
    combined = cum.sum(axis=1)

    fig, ax = plt.subplots(figsize=(13, 5))
    for col in cum.columns:
        sns.lineplot(x=cum.index, y=cum[col], ax=ax, label=col, linewidth=2)
    sns.lineplot(x=combined.index, y=combined.values, ax=ax,
                 label="Combined", linewidth=2.5, linestyle="--", color="black")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Cumulative PnL")
    ax.set_ylabel("PnL")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_daily_pnl(result: BacktestResult) -> Figure:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    Figure
        Daily PnL bars, one color per underlier, side by side.
    """
    daily = (
        result.daily_pnl
        .reset_index()
        .melt(id_vars="date", var_name="underlier", value_name="pnl")
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(data=daily, x="date", y="pnl", hue="underlier", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Daily PnL by Underlier")
    ax.set_ylabel("PnL")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig

def plot_drawdown(result: BacktestResult) -> Figure:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    Figure
        Drawdown per underlier over the month: drawdown at day D =
        running_peak_up_to_D - cumulative_pnl_at_D, i.e. how far below the
        previous best you are at each point.
    """
    daily = result.daily_pnl.fillna(0)
    fig, ax = plt.subplots(figsize=(13, 5))
    for col in daily.columns:
        cum = daily[col].cumsum()
        dd  = -(cum.cummax() - cum)
        sns.lineplot(x=cum.index, y=dd.values, ax=ax, label=col, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Drawdown over the month")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_intraday_mtm(
    result:       BacktestResult,
    trading_date: str | None = None,
    underlier:    str | None = None,
) -> Figure:
    """
    Parameters
    ----------
    result : BacktestResult
    trading_date : str, optional
        ISO format (matches Python's default date.__str__()). Defaults to
        the first day in the result.
    underlier : str, optional
        Defaults to the first underlier in the result.

    Returns
    -------
    Figure
        Per-second MTM curve for one sample day, with profitable seconds
        shaded green and loss seconds shaded red — shows the intraday
        choppiness of a rolling straddle.

    Raises
    ------
    ValueError
        If no result exists for that (trading_date, underlier) pair.
    """
    first = result.day_results[0] if result.day_results else None
    trading_date = trading_date or (str(first.date) if first else "")
    underlier = underlier or (first.underlier if first else "")
    dr = next(
        (d for d in result.day_results
         if str(d.date) == trading_date and d.underlier == underlier),
        None
    )
    if dr is None:
        raise ValueError(f"No result for {underlier} on {trading_date}")
    fig, ax = plt.subplots(figsize=(13, 5))
    y = dr.mtm.to_numpy(dtype=float)
    x = dr.mtm.index.to_numpy()
    sns.lineplot(x=x, y=y, ax=ax, linewidth=1.2, color="purple")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.fill_between(x, y, 0, where=(y >= 0).tolist(), alpha=0.2, color="green")
    ax.fill_between(x, y, 0, where=(y < 0).tolist(), alpha=0.2, color="red")
    ax.set_title(f"Intraday MTM — {underlier} {trading_date} (final PnL: {dr.final_pnl:.2f})")
    ax.set_ylabel("MTM PnL")
    ax.set_xlabel("Time")
    plt.tight_layout()
    return fig

def plot_rolls_per_day(result: BacktestResult) -> Figure:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    Figure
        Roll frequency per day per underlier — shows how active the
        strategy is day to day.
    """
    df = rolls_per_day(result)
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(data=df, x="date", y="rolls", hue="underlier", ax=ax)
    ax.set_title("Rolls per Day")
    ax.set_ylabel("Number of rolls")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig

def plot_holdings(
    result: BacktestResult,
    trading_date: str | None = None,
    underlier: str | None = None,
) -> Figure:
    """
    Parameters
    ----------
    result : BacktestResult
    trading_date : str, optional
        ISO format (matches Python's default date.__str__()). Defaults to
        the first day in the result.
    underlier : str, optional
        Defaults to the first underlier in the result.

    Returns
    -------
    Figure
        Strike held at every second of a sample day, as a step chart. Flat
        periods where nothing is held appear as gaps — makes rolling
        behaviour visually obvious.

    Raises
    ------
    ValueError
        If no result exists for that (trading_date, underlier) pair, or if
        it exists but no position was ever held that day.
    """
    first = result.day_results[0] if result.day_results else None
    trading_date = trading_date or (str(first.date) if first else "")
    underlier = underlier or (first.underlier if first else "")
    dr = next(
        (d for d in result.day_results
         if str(d.date) == trading_date and d.underlier == underlier),
        None
    )
    if dr is None:
        raise ValueError(f"No result for {underlier} on {trading_date}")
    if dr.holdings_log.empty:
        raise ValueError(f"No holdings recorded for {underlier} on {trading_date}")
    log = dr.holdings_log
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.step(log.index, log["strike"], where="post",linewidth=1.5, color="steelblue")
    ax.set_title(f"Strike held per second — {underlier} {trading_date}")
    ax.set_ylabel("Strike")
    ax.set_xlabel("Time")
    visited = sorted(log["strike"].dropna().unique().astype(int))
    ax.set_yticks(visited)
    plt.tight_layout()
    return fig

def plot_rolling_sharpe(result: BacktestResult, window: int = 5) -> Figure:
    """
    Parameters
    ----------
    result : BacktestResult
    window : int, default 5
        Trailing window size in trading days — see rolling_sharpe().

    Returns
    -------
    Figure
    """
    rs   = rolling_sharpe(result, window)
    fig, ax = plt.subplots(figsize=(13, 4))
    for col in rs.columns:
        sns.lineplot(x=rs.index, y=rs[col], ax=ax, label=col, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(f"Rolling {window}-day Sharpe ratio")
    ax.set_ylabel("Sharpe")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_cost_sensitivity(result: BacktestResult) -> Figure:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    Figure
        Net PnL under different transaction cost assumptions — see
        cost_sensitivity(). The leftmost bar (0 bps) is what the backtest
        itself reports; every bar to the right estimates real trading.
    """
    df = cost_sensitivity(result)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["net_pnl"]]
    ax.bar(df["cost_bps"].astype(str), df["net_pnl"], color=colors, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Net PnL vs Transaction Cost Assumption")
    ax.set_xlabel("Cost (basis points per trade)")
    ax.set_ylabel("Net PnL")
    for i, (bps, val) in enumerate(zip(df["cost_bps"], df["net_pnl"])):
        ax.text(i, val + (5 if val >= 0 else -15), f"{val:.0f}", ha="center", fontsize=9)
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Helper functions used by the Streamlit app / PDF report
# ---------------------------------------------------------------------------

def round_trips_by_leg(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    trades_df : pd.DataFrame
        As returned by BacktestResult.trades_df.

    Returns
    -------
    pd.DataFrame
        _round_trips() output with an added "leg" column ("CE" or "PE"),
        extracted from the instrument filename. Empty if there are no
        complete round trips.
    """
    rt = _round_trips(trades_df)
    if rt.empty:
        return rt
    rt["leg"] = rt["instrument"].str.extract(r"(CE|PE)\.csv$")[0]
    return rt

def morning_afternoon_split(day_results: list) -> pd.DataFrame:
    """
    Parameters
    ----------
    day_results : list[DayResult]

    Returns
    -------
    pd.DataFrame
        One row per (date, underlier): date, underlier, morning_pnl
        (session open to 12:00:00), afternoon_pnl (12:00:00 to session
        close). The two always sum to that day's final_pnl.
    """
    rows = []
    for dr in day_results:
        noon_ts = pd.Timestamp(f"{dr.date} 12:00:00")
        if noon_ts not in dr.mtm.index:
            noon_ts = dr.mtm.index[dr.mtm.index.searchsorted(noon_ts) - 1]
        morning   = float(dr.mtm.at[noon_ts] - dr.mtm.iloc[0])
        afternoon = float(dr.mtm.iloc[-1] - dr.mtm.at[noon_ts])
        rows.append({
            "date":          dr.date,
            "underlier":     dr.underlier,
            "morning_pnl":   round(morning, 2),
            "afternoon_pnl": round(afternoon, 2),
        })
    return pd.DataFrame(rows)

def pnl_by_weekday(result: "BacktestResult") -> pd.DataFrame:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.DataFrame
        Average daily PnL grouped by weekday and underlier: weekday
        (Monday-Friday, ordered), underlier, avg_pnl, count.
    """
    daily = result.daily_pnl.reset_index().melt(
        id_vars="date", var_name="underlier", value_name="pnl"
    )
    daily["weekday"] = pd.to_datetime(daily["date"]).dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    summary = (
        daily.groupby(["weekday", "underlier"])["pnl"]
        .agg(avg_pnl="mean", count="count")
        .reset_index()
    )
    summary["weekday"] = pd.Categorical(summary["weekday"], categories=order, ordered=True)
    return summary.sort_values("weekday").reset_index(drop=True)

def expiry_day_pnl(result: "BacktestResult") -> pd.DataFrame:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.DataFrame
        Average PnL split by whether the day traded its own expiry:
        underlier, day_type ("Expiry day" / "Non-expiry"), avg_pnl, count.
        Days with zero trades contribute no row (no expiry/non-expiry
        classification is possible without a trade to check). Empty (with
        these columns) if no day in the result has any trades at all.
    """
    trades = result.trades_df
    rows = []
    for dr in result.day_results:
        day_trades = trades[
            (trades["date"] == dr.date) & (trades["underlier"] == dr.underlier)
        ]
        if day_trades.empty:
            continue
        is_expiry = (day_trades["expiry"] == dr.date).any()
        rows.append({
            "date":      dr.date,
            "underlier": dr.underlier,
            "day_type":  "Expiry day" if is_expiry else "Non-expiry",
            "pnl":       dr.final_pnl,
        })
    if not rows:
        return pd.DataFrame(columns=["underlier", "day_type", "avg_pnl", "count"])
    df = pd.DataFrame(rows)
    return (
        df.groupby(["underlier", "day_type"])["pnl"]
        .agg(avg_pnl="mean", count="count")
        .reset_index()
    )

def avg_intraday_mtm(result: "BacktestResult") -> pd.DataFrame:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.DataFrame
        Element-wise mean of every day's MTM curve, aligned by time-of-day
        (not date) — one column per underlier, index = time. Every trading
        day shares the same 09:15:00-15:29:59 grid, so alignment by `.time`
        is exact with no interpolation needed.
    """
    by_ul: dict[str, list[pd.Series]] = {}
    for dr in result.day_results:
        s = dr.mtm.copy()
        s.index = pd.DatetimeIndex(s.index).time
        by_ul.setdefault(dr.underlier, []).append(s)
    combined = {}
    for ul, series_list in by_ul.items():
        stacked      = pd.concat(series_list, axis=1)
        combined[ul] = stacked.mean(axis=1)
    return pd.DataFrame(combined)

def notional_per_day(result: "BacktestResult") -> pd.DataFrame:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.DataFrame
        Total notional (price * qty * multiplier, summed over every trade — both entry
        and exit legs) traded per day per underlier: date, underlier, notional.
    """
    trades = result.notional_df
    return (
        trades.groupby(["date", "underlier"])["notional"]
        .sum()
        .reset_index()
    )

def breakeven_bps(result: "BacktestResult") -> float:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    float
        The transaction cost, in basis points per trade, at which net PnL
        (gross - notional * bps / 10,000) would equal zero. Negative if the
        strategy is already loss-making at zero cost (i.e. it would need a
        subsidy, not a cost cut, to break even). 0.0 if there's no traded
        notional to apply a cost to.
    """
    total_notional = result.notional_df["notional"].sum()
    gross_pnl      = result.daily_pnl.sum().sum()
    if total_notional == 0:
        return 0.0
    return round(gross_pnl * 10_000 / total_notional, 4)

# ---------------------------------------------------------------------------
# Tail risk and distribution shape
# ---------------------------------------------------------------------------

def value_at_risk(result: BacktestResult, confidence: float = 0.95) -> pd.Series:
    """
    Historical (empirical) VaR: the (1 - confidence) quantile of the daily
    PnL distribution.

    Parameters
    ----------
    result : BacktestResult
    confidence : float, default 0.95

    Returns
    -------
    pd.Series
        Per underlier, reported as a positive loss magnitude (the raw
        quantile is negated) — matches the sign convention of
        max_drawdown(), where larger means worse.
    """
    return -result.daily_pnl.quantile(1 - confidence)

def conditional_value_at_risk(result: BacktestResult, confidence: float = 0.95) -> pd.Series:
    """
    CVaR / Expected Shortfall: the mean PnL on days at or beyond the VaR
    threshold — captures how bad the tail is on average, not just where it
    starts.

    Parameters
    ----------
    result : BacktestResult
    confidence : float, default 0.95

    Returns
    -------
    pd.Series
        Per underlier, positive loss magnitude (same sign convention as
        value_at_risk). Falls back to the VaR threshold itself if no day
        actually falls at or beyond it.
    """
    threshold = result.daily_pnl.quantile(1 - confidence)
    results = {}
    for col in result.daily_pnl.columns:
        pnl  = result.daily_pnl[col].dropna()
        tail = pnl[pnl <= threshold[col]]
        results[col] = -tail.mean() if not tail.empty else -threshold[col]
    return pd.Series(results)

def pnl_skewness(result: BacktestResult) -> pd.Series:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Skewness of daily PnL per underlier. Negative means a longer left
        tail (occasional large losses); positive means a longer right tail.
    """
    return result.daily_pnl.skew()

def pnl_kurtosis(result: BacktestResult) -> pd.Series:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Excess kurtosis of daily PnL per underlier (0 = normal
        distribution's tail weight; positive means fatter tails than normal).
    """
    return result.daily_pnl.kurt()

def ulcer_index(result: BacktestResult) -> pd.Series:
    """
    sqrt(mean squared drawdown) — unlike max_drawdown, penalizes time spent
    underwater as well as depth, since every day in a drawdown contributes
    to the average, not just the single worst day.

    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Per underlier, in the same points-based units as the rest of this
        module's drawdown metrics (not the percentage-of-equity convention
        some textbook definitions use — kept consistent with max_drawdown/
        calmar_ratio, which are also points-based here).
    """
    results = {}
    for col in result.daily_pnl.columns:
        cum = result.daily_pnl[col].fillna(0).cumsum()
        dd  = cum.cummax() - cum
        results[col] = np.sqrt((dd ** 2).mean())
    return pd.Series(results)

def recovery_factor(result: BacktestResult) -> pd.Series:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.Series
        Total PnL per unit of max drawdown, per underlier — unannualized,
        unlike calmar_ratio which uses annualized return in the numerator.
        NaN where max_drawdown is zero.
    """
    mdd = max_drawdown(result)
    return result.daily_pnl.sum() / mdd.replace(0, np.nan)

def omega_ratio(result: BacktestResult, threshold: float = 0.0) -> pd.Series:
    """
    Probability-weighted gains vs losses relative to a threshold return:
    Omega = sum(max(pnl - threshold, 0)) / sum(max(threshold - pnl, 0)).

    Parameters
    ----------
    result : BacktestResult
    threshold : float, default 0.0
        Minimum acceptable daily PnL. At the default of 0, this is
        mathematically identical to profit_factor() — Omega's value over
        profit_factor is that threshold can be set to something other than
        breakeven.

    Returns
    -------
    pd.Series
        Per underlier. +inf if no day falls below the threshold.
    """
    results = {}
    for col in result.daily_pnl.columns:
        pnl    = result.daily_pnl[col].dropna()
        gains  = (pnl - threshold).clip(lower=0).sum()
        losses = (threshold - pnl).clip(lower=0).sum()
        results[col] = gains / losses if losses > 0 else float("inf")
    return pd.Series(results)

def sharpe_confidence_interval(
    result:     BacktestResult,
    n_boot:     int = 2000,
    confidence: float = 0.95,
    seed:       int = 42,
) -> pd.DataFrame:
    """
    Bootstrap confidence interval on the annualized Sharpe ratio, to
    quantify how much sampling noise a ~21-day estimate actually carries
    (see the caveat on sharpe_ratio()).

    Parameters
    ----------
    result : BacktestResult
    n_boot : int, default 2000
        Number of bootstrap resamples.
    confidence : float, default 0.95
        Confidence level for the interval.
    seed : int, default 42
        RNG seed, for reproducible results across runs.

    Returns
    -------
    pd.DataFrame
        One row per underlier (indexed by underlier): sharpe (point
        estimate), ci_low, ci_high. Resamples with zero variance (all
        identical values, possible with very few days) are excluded from
        the interval rather than propagating inf/NaN into it; if every
        resample is degenerate, ci_low/ci_high are NaN.
    """
    rng   = np.random.default_rng(seed)
    alpha = 1 - confidence
    rows  = []
    for col in result.daily_pnl.columns:
        pnl  = result.daily_pnl[col].dropna().to_numpy()
        boot = np.empty(n_boot)
        for i in range(n_boot):
            sample  = rng.choice(pnl, size=len(pnl), replace=True)
            s_std   = sample.std(ddof=1)
            boot[i] = (sample.mean() / s_std) * np.sqrt(252) if s_std > 0 else np.nan
        point_std = pnl.std(ddof=1)
        valid     = boot[~np.isnan(boot)]
        rows.append({
            "underlier": col,
            "sharpe":    (pnl.mean() / point_std) * np.sqrt(252) if point_std > 0 else np.nan,
            "ci_low":    np.quantile(valid, alpha / 2)     if valid.size else np.nan,
            "ci_high":   np.quantile(valid, 1 - alpha / 2) if valid.size else np.nan,
        })
    return pd.DataFrame(rows).set_index("underlier")

def risk_table(result: BacktestResult) -> pd.DataFrame:
    """
    Parameters
    ----------
    result : BacktestResult

    Returns
    -------
    pd.DataFrame
        One row per underlier (indexed by underlier): var_95, cvar_95,
        skew, kurtosis, ulcer_index, recovery_factor, omega. Empty (with
        these columns) if the result has no underliers at all.
    """
    rows = []
    var  = value_at_risk(result)
    cvar = conditional_value_at_risk(result)
    skew = pnl_skewness(result)
    kurt = pnl_kurtosis(result)
    ui   = ulcer_index(result)
    rf   = recovery_factor(result)
    om   = omega_ratio(result)
    for col in result.daily_pnl.columns:
        rows.append({
            "underlier":       col,
            "var_95":          round(var[col], 2),
            "cvar_95":         round(cvar[col], 2),
            "skew":            round(skew[col], 3),
            "kurtosis":        round(kurt[col], 3),
            "ulcer_index":     round(ui[col], 2),
            "recovery_factor": round(rf[col], 3),
            "omega":           round(om[col], 3),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "underlier", "var_95", "cvar_95", "skew", "kurtosis",
            "ulcer_index", "recovery_factor", "omega",
        ]).set_index("underlier")
    return pd.DataFrame(rows).set_index("underlier")

def drawdown_table(result: BacktestResult, top_n: int = 5) -> pd.DataFrame:
    """
    Parameters
    ----------
    result : BacktestResult
    top_n : int, default 5
        Number of episodes to keep per underlier, ranked by depth.

    Returns
    -------
    pd.DataFrame
        Up to top_n drawdown episodes per underlier: underlier, trough_date
        (the worst day of the episode), depth, recovery_date (the first day
        cumulative PnL reclaims the pre-drawdown peak, or None if the
        episode hasn't recovered by the end of the result). Empty if the
        cumulative PnL curve never dips below its running peak.
    """
    rows = []
    for col in result.daily_pnl.columns:
        cum    = result.daily_pnl[col].fillna(0).cumsum()
        dd     = cum.cummax() - cum
        in_dd  = dd > 0
        groups = (in_dd != in_dd.shift()).cumsum()

        for _, idx in dd[in_dd].groupby(groups[in_dd]).groups.items():
            episode     = dd.loc[idx]
            trough_date = episode.idxmax()
            end_pos     = cum.index.get_loc(idx[-1])
            recovery_date = cum.index[end_pos + 1] if end_pos + 1 < len(cum) else None
            rows.append({
                "underlier":     col,
                "trough_date":   trough_date,
                "depth":         round(episode.max(), 2),
                "recovery_date": recovery_date,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.sort_values("depth", ascending=False)
        .groupby("underlier", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )

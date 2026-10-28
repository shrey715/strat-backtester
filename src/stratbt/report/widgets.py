# streamlit_app/report/widgets.py
"""
Render stage of the report pipeline.

This module owns HOW each element looks; pages.py owns WHERE it goes.
The contract with pages.py:
  • render_table(fig, rect, ...)  — rect is the TABLE BODY only (no title).
    The title is drawn via ax.set_title() which matplotlib places above the
    axes rect automatically.  pages.py reserves TITLE_CLEAR_IN inches for it.
  • Every chart function takes a pre-sized Axes; pages.py is responsible for
    computing the correct height before calling add_axes / add_gridspec.

Date-label note: X-axis dates are formatted as "Nov 09" (7 chars) instead of
"2022-11-09" (10 chars) so they don't collide when rotated 20°.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.dates import DateFormatter, AutoDateLocator

from stratbt.palette import SERIES_COLORS as _SERIES_COLORS

PAGE_SIZE = (11.0, 8.5)
INK       = "#2c3e50"
MUTED     = "#888888"
GRID      = "#cccccc"

SERIES_COLORS  = _SERIES_COLORS
POSITIVE_COLOR = "#2ca02c"
NEGATIVE_COLOR = "#d62728"

ROW_HEIGHT_IN  = 0.24   # per data/header row, at fontsize 8
TITLE_CLEAR_IN = 0.38   # vertical space a titled element needs above its body axes


# ---------------------------------------------------------------------------
# Page factories
# ---------------------------------------------------------------------------

def new_page() -> plt.Figure:
    return plt.figure(figsize=PAGE_SIZE)


def chrome(fig: plt.Figure, section: str, page_num: int,
           total_pages: int, generated_at: str,
           strategy_name: str = "Options Backtest") -> None:
    """Page header and footer chrome drawn once per figure."""
    # Left: strategy name — the primary identifier across all pages
    fig.text(0.05, 0.965, strategy_name,
             fontsize=8, weight="bold", color=INK, va="center")
    # Right: section label
    fig.text(0.95, 0.965, section.upper(),
             fontsize=8, color=MUTED, ha="right", va="center", family="monospace")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.95, 0.95],
                              transform=fig.transFigure,
                              color=GRID, linewidth=0.8))
    fig.text(0.05, 0.015, f"Generated {generated_at}", fontsize=7, color="#aaaaaa")
    fig.text(0.95, 0.015, f"Page {page_num} / {total_pages}",
             fontsize=7, color="#aaaaaa", ha="right")



def kpi_strip(fig: plt.Figure, kpis: list[tuple[str, str]],
              y: float, height: float = 0.10) -> None:
    """Render a row of KPI tiles across the top of a section."""
    n    = len(kpis)
    gap  = 0.015
    tile = (0.9 - gap * (n - 1)) / n
    for i, (label, value) in enumerate(kpis):
        x = 0.05 + i * (tile + gap)
        fig.add_artist(plt.Rectangle(
            (x, y), tile, height, transform=fig.transFigure,
            facecolor="#f4f6f8", edgecolor="#d0d8e0", linewidth=0.9,
        ))
        fig.text(x + tile / 2, y + height - 0.028, value,
                 transform=fig.transFigure,
                 fontsize=14, weight="bold", ha="center", va="top", color=INK)
        fig.text(x + tile / 2, y + 0.014, label.upper(),
                 transform=fig.transFigure,
                 fontsize=6.5, ha="center", va="bottom", color="#9aabb8")


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def _format_cell(v, precision: int = 2, none_label: str = "—") -> str:
    """
    Format a single cell value for display in a report table.

    Parameters
    ----------
    v : Any
        The raw cell value (float, int, date, string, or None/NaN).
    precision : int, default 2
        Decimal places used when formatting floats.
    none_label : str, default "—"
        String shown for None/NaN/NaT. Callers override this per-column
        when None carries a specific meaning (e.g. "Ongoing" for an
        unresolved drawdown's recovery_date, versus "—" for genuinely
        missing data elsewhere).

    Returns
    -------
    str
        A display-ready string. Floats get thousands separators and
        `precision` decimals; ints get thousands separators only; anything
        else is stringified as-is.
    """
    if v is None:
        return none_label
    if isinstance(v, float):
        if pd.isna(v):
            return none_label
        return f"{v:,.{precision}f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    # dates, strings, etc.
    s = str(v)
    return none_label if s in ("nan", "None", "NaT") else s


def render_table(fig: plt.Figure,
                 rect: tuple[float, float, float, float],
                 df: pd.DataFrame,
                 title: str = "",
                 index_label: str = "",
                 precision: int = 2,
                 none_labels: dict[str, str] | None = None) -> None:
    """
    Render a DataFrame as a formatted table inside a figure.

    Parameters
    ----------
    fig : plt.Figure
        The page figure to draw onto.
    rect : tuple[float, float, float, float]
        (left, bottom, width, height) in figure fractions. Covers the table
        BODY only — the title floats above the axes rect via ax.set_title(),
        so callers must reserve TITLE_CLEAR_IN separately (see pages.py).
    df : pd.DataFrame
        Data to render. The index becomes an ordinary first column rather
        than matplotlib's built-in rowLabels, which render outside the
        axes' left edge and get clipped by the page margin (verified
        empirically — never used here).
    title : str, default ""
        Table title, drawn above the axes. Omitted if empty.
    index_label : str, default ""
        Header text for the index-turned-first-column.
    precision : int, default 2
        Decimal places for float cells.
    none_labels : dict[str, str], optional
        Per-column override for how None/NaN renders, keyed by column name
        (e.g. {"recovery_date": "Ongoing"}). Columns not listed use "—".

    Returns
    -------
    None
        Draws directly onto `fig`; nothing is returned.

    Column widths are computed proportional to each column's longest
    string so no cell ever bleeds into a neighbour.
    """
    none_labels = none_labels or {}
    left, bottom, width, height = rect
    ax = fig.add_axes([left, bottom, width, height])
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, weight="bold", loc="left", pad=12)

    if df.empty:
        ax.text(0.0, 0.85, "No data for this run.",
                fontsize=9, color=MUTED, transform=ax.transAxes)
        return

    header = [index_label or (df.index.name or "")] + [str(c) for c in df.columns]
    body   = [
        [str(idx)] + [
            _format_cell(v, precision=precision, none_label=none_labels.get(col, "—"))
            for col, v in zip(df.columns, row)
        ]
        for idx, row in zip(df.index, df.values)
    ]

    def _clen(s: str) -> int:
        return max(len(s), 3)

    col_chars = [
        max(_clen(header[i]), max((_clen(r[i]) for r in body), default=0))
        for i in range(len(header))
    ]
    total     = sum(col_chars)
    col_widths = [c / total for c in col_chars]

    tbl = ax.table(cellText=body, colLabels=header, colWidths=col_widths,
                   cellLoc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d8dde4")
        if row == 0:                       # header
            cell.set_facecolor(INK)
            cell.set_text_props(color="white", weight="bold", fontsize=7.5)
        elif col == 0:                     # row-label column
            cell.set_facecolor("#edf0f4")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:                 # alternating zebra stripe
            cell.set_facecolor("#f7f9fb")


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _style_date_axis(ax: plt.Axes) -> None:
    """Format x-axis dates as 'Nov 09' and thin out ticks so they never crowd."""
    try:
        ax.xaxis.set_major_locator(AutoDateLocator(minticks=4, maxticks=8))
        ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
    except Exception:
        pass   # non-datetime index — leave as-is
    ax.tick_params(labelsize=7, axis="x", rotation=20)


def _chart_style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.axhline(0, color=GRID, linewidth=0.8, linestyle=":")
    ax.set_title(title, fontsize=12, weight="bold", loc="left", pad=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#d0d8e0")
    ax.spines["bottom"].set_color("#d0d8e0")
    # leave a small bottom margin so the zero hline isn't clipped
    ylo, yhi = ax.get_ylim()
    pad = (yhi - ylo) * 0.04
    ax.set_ylim(ylo - pad, yhi + pad)


def render_line_chart(ax: plt.Axes,
                      series: dict[str, pd.Series],
                      title: str,
                      ylabel: str,
                      combined: pd.Series | None = None) -> None:
    for name, s in series.items():
        ax.plot(s.index, s.values, label=name, linewidth=2,
                color=SERIES_COLORS.get(name, "#888"))
    if combined is not None:
        ax.plot(combined.index, combined.values, label="Combined",
                linewidth=2.5, linestyle="--", color=INK)
    _style_date_axis(ax)
    _chart_style(ax, title, ylabel)
    ax.legend(fontsize=7, loc="upper left", frameon=False)


def render_bar_chart(ax: plt.Axes,
                     df: pd.DataFrame,
                     x: str, y: str, hue: str,
                     title: str, ylabel: str) -> None:
    groups = df[hue].unique()
    xvals  = sorted(df[x].unique())
    width  = 0.8 / max(len(groups), 1)
    for i, g in enumerate(groups):
        sub = df[df[hue] == g].set_index(x).reindex(xvals)
        pos = np.arange(len(xvals)) + i * width
        ax.bar(pos, sub[y].fillna(0).values, width=width,
               label=str(g), color=SERIES_COLORS.get(g, "#888"), alpha=0.85)
    # x-tick labels: "Nov 09" style from date objects
    try:
        import pandas as _pd
        labels = [_pd.Timestamp(v).strftime("%b %d") for v in xvals]
        ax.set_xticks(np.arange(len(xvals)) + width * (len(groups) - 1) / 2)
        ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    except Exception:
        ax.set_xticks([])
    ax.axhline(0, color=GRID, linewidth=0.8, linestyle=":")
    ax.set_title(title, fontsize=12, weight="bold", loc="left", pad=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#d0d8e0")
    ax.spines["bottom"].set_color("#d0d8e0")
    ax.legend(fontsize=7, loc="upper left", frameon=False)


def render_drawdown_chart(ax: plt.Axes, daily: pd.DataFrame) -> None:
    for col in daily.columns:
        cum   = daily[col].cumsum()
        dd    = -(cum.cummax() - cum)
        color = SERIES_COLORS.get(col, "#888")
        ax.fill_between(cum.index, dd.values, 0, alpha=0.12, color=color)
        ax.plot(cum.index, dd.values, linewidth=1.8, color=color, label=col)
    ax.axhline(0, color=GRID, linewidth=0.8, linestyle=":")
    ax.set_title("Drawdown", fontsize=12, weight="bold", loc="left", pad=10)
    ax.set_ylabel("Drawdown", fontsize=8)
    _style_date_axis(ax)
    ax.tick_params(labelsize=7, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#d0d8e0")
    ax.spines["bottom"].set_color("#d0d8e0")
    ax.legend(fontsize=7, loc="lower left", frameon=False)


def render_cost_chart(ax: plt.Axes, cs: pd.DataFrame) -> None:
    colors = [NEGATIVE_COLOR if v < 0 else POSITIVE_COLOR for v in cs["net_pnl"]]
    bars = ax.bar(cs["cost_bps"].astype(str), cs["net_pnl"],
                  color=colors, width=0.5, alpha=0.85)
    # annotate each bar with its value
    for bar, val in zip(bars, cs["net_pnl"]):
        y_off = 4 if val >= 0 else -14
        ax.annotate(
            f"{val:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, y_off), textcoords="offset points",
            ha="center", fontsize=8, color="#444444",
        )
    ax.axhline(0, color=GRID, linewidth=0.8, linestyle=":")
    ax.set_title("Net PnL vs Transaction Cost Assumption",
                 fontsize=12, weight="bold", loc="left", pad=10)
    ax.set_xlabel("Cost (bps per trade)", fontsize=8)
    ax.set_ylabel("Net PnL", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#d0d8e0")
    ax.spines["bottom"].set_color("#d0d8e0")
    # y-axis headroom so annotations don't clip
    ylo, yhi = ax.get_ylim()
    pad = (yhi - ylo) * 0.12
    ax.set_ylim(ylo - pad, yhi + pad)


# ---------------------------------------------------------------------------
# New: Statistical validity charts
# ---------------------------------------------------------------------------

def render_pnl_histogram(
    ax: plt.Axes,
    daily_pnl: pd.DataFrame,
    var_series: pd.Series,
    cvar_series: pd.Series,
) -> None:
    """
    Histogram of daily PnL per underlier with VaR95 and CVaR95 vertical lines.
    Shows the full return distribution — skew, kurtosis, and tail risk at a glance.
    """
    for col in daily_pnl.columns:
        color = SERIES_COLORS.get(col, "#888")
        ax.hist(daily_pnl[col].dropna(), bins=12, alpha=0.55, color=color,
                label=col, edgecolor="white", linewidth=0.5)

    # Draw VaR/CVaR lines *after* hist so ylim is set properly
    for col in daily_pnl.columns:
        color    = SERIES_COLORS.get(col, "#888")
        var_val  = -var_series[col]    # var_series stores loss as positive
        cvar_val = -cvar_series[col]
        ax.axvline(var_val,  color=color, linewidth=1.4, linestyle="--")
        ax.axvline(cvar_val, color=color, linewidth=1.0, linestyle="-.", alpha=0.7)
        # label just above the x-axis at 5% of current y-height
        ylo, yhi = ax.get_ylim()
        ax.text(var_val, ylo + (yhi - ylo) * 0.04,
                f" {col}\n VaR\n {var_val:.0f}",
                fontsize=6, color=color, va="bottom")

    ax.axvline(0, color=GRID, linewidth=0.9, linestyle=":")
    ax.set_title("Daily PnL Distribution  (dashed = VaR 95 / dash-dot = CVaR 95)",
                 fontsize=11, weight="bold", loc="left", pad=10)
    ax.set_xlabel("Daily PnL", fontsize=8)
    ax.set_ylabel("Frequency (days)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#d0d8e0")
    ax.spines["bottom"].set_color("#d0d8e0")
    ax.legend(fontsize=7, frameon=False, loc="upper left")



def render_sharpe_ci(ax: plt.Axes, ci: pd.DataFrame) -> None:
    """
    Bar chart of annualised Sharpe with 95% bootstrap confidence intervals.
    Error bars show [ci_low, ci_high].  A bar crossing zero means the Sharpe
    is not statistically distinguishable from zero at 95% confidence.
    """
    underliers = ci.index.tolist()
    x          = np.arange(len(underliers))
    colors     = [SERIES_COLORS.get(ul, "#888") for ul in underliers]
    sharpes    = ci["sharpe"].values
    err_lo     = sharpes - ci["ci_low"].values
    err_hi     = ci["ci_high"].values - sharpes

    bars = ax.bar(x, sharpes, color=colors, alpha=0.80, width=0.5,
                  yerr=[err_lo, err_hi],
                  error_kw=dict(ecolor="#555555", capsize=6, capthick=1.2,
                                elinewidth=1.2))
    ax.axhline(0, color=GRID, linewidth=0.9, linestyle=":")

    # annotate bars with point estimate
    for i, (bar, val) in enumerate(zip(bars, sharpes)):
        y_off = 4 if val >= 0 else -14
        ax.text(bar.get_x() + bar.get_width() / 2, val + (err_hi[i] if val >= 0 else -err_lo[i]),
                f"{val:.2f}", ha="center", fontsize=8, va="bottom" if val >= 0 else "top",
                color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(underliers, fontsize=9)
    ax.set_title("Sharpe Ratio — Point Estimate with 95% Bootstrap CI  (n = 2,000 resamples)",
                 fontsize=11, weight="bold", loc="left", pad=10)
    ax.set_ylabel("Sharpe (annualised)", fontsize=8)
    ax.tick_params(labelsize=7, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#d0d8e0")
    ax.spines["bottom"].set_color("#d0d8e0")
    ylo, yhi = ax.get_ylim()
    pad = (yhi - ylo) * 0.15
    ax.set_ylim(ylo - pad, yhi + pad)


# ---------------------------------------------------------------------------
# New: Pattern analysis charts
# ---------------------------------------------------------------------------

def render_weekday_bars(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Average PnL by day of week, underliers side by side.
    df cols: weekday, underlier, avg_pnl, count
    """
    order   = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    uls     = sorted(df["underlier"].unique())
    n       = len(uls)
    width   = 0.7 / max(n, 1)
    x       = np.arange(len(order))

    for i, ul in enumerate(uls):
        sub    = df[df["underlier"] == ul].set_index("weekday")["avg_pnl"]
        vals   = [sub.get(d, 0.0) for d in order]
        colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in vals]
        ax.bar(x + i * width, vals, width=width,
               color=SERIES_COLORS.get(ul, "#888"), alpha=0.82, label=ul)

    ax.axhline(0, color=GRID, linewidth=0.8, linestyle=":")
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels([d[:3] for d in order], fontsize=8)   # "Mon", "Tue" …
    ax.set_title("Average Daily PnL by Day of Week",
                 fontsize=12, weight="bold", loc="left", pad=10)
    ax.set_ylabel("Avg PnL", fontsize=8)
    ax.tick_params(labelsize=7, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#d0d8e0")
    ax.spines["bottom"].set_color("#d0d8e0")
    ax.legend(fontsize=7, frameon=False, loc="upper right")


def render_expiry_bars(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Expiry day vs non-expiry average PnL per underlier.
    df cols: underlier, day_type, avg_pnl, count
    """
    uls      = sorted(df["underlier"].unique())
    types    = ["Expiry day", "Non-expiry"]
    n_types  = len(types)
    width    = 0.35
    x        = np.arange(len(uls))
    type_colors = {"Expiry day": "#e06030", "Non-expiry": "#4a90d9"}

    for i, dt in enumerate(types):
        sub  = df[df["day_type"] == dt].set_index("underlier")
        vals = [sub["avg_pnl"].get(ul, 0.0) for ul in uls]
        cnts = [int(sub["count"].get(ul, 0)) for ul in uls]
        bars = ax.bar(x + i * width, vals, width=width,
                      color=type_colors.get(dt, "#888"), alpha=0.82, label=dt)
        for bar, cnt in zip(bars, cnts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (2 if bar.get_height() >= 0 else -10),
                    f"n={cnt}", ha="center", fontsize=6.5, color="#555")

    ax.axhline(0, color=GRID, linewidth=0.8, linestyle=":")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(uls, fontsize=9)
    ax.set_title("Expiry Day vs Non-Expiry: Average Daily PnL",
                 fontsize=12, weight="bold", loc="left", pad=10)
    ax.set_ylabel("Avg PnL", fontsize=8)
    ax.tick_params(labelsize=7, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#d0d8e0")
    ax.spines["bottom"].set_color("#d0d8e0")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ylo, yhi = ax.get_ylim()
    pad = (yhi - ylo) * 0.18
    ax.set_ylim(ylo - pad, yhi + pad)


def render_morning_afternoon_bars(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Stacked bar: morning PnL + afternoon PnL per day, per underlier.
    Shows intraday timing of where the strategy captures (or loses) PnL.
    """
    for ul in sorted(df["underlier"].unique()):
        sub   = df[df["underlier"] == ul].sort_values("date")
        color = SERIES_COLORS.get(ul, "#888")
        dates = [str(d) for d in sub["date"]]
        x     = np.arange(len(dates))
        ax.bar(x, sub["morning_pnl"].values, width=0.8,
               color=color, alpha=0.85, label=f"{ul} AM")
        ax.bar(x, sub["afternoon_pnl"].values, width=0.8,
               bottom=sub["morning_pnl"].values,
               color=color, alpha=0.45, label=f"{ul} PM",
               hatch="//", edgecolor="white")

    ax.axhline(0, color=GRID, linewidth=0.8, linestyle=":")
    ax.set_title("Morning (solid) vs Afternoon (hatched) PnL per Day",
                 fontsize=11, weight="bold", loc="left", pad=10)
    ax.set_ylabel("PnL", fontsize=8)
    ax.set_xticks([])
    ax.tick_params(labelsize=7, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#d0d8e0")
    ax.spines["bottom"].set_color("#d0d8e0")
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)

# streamlit_app/report/pages.py
"""
Layout stage of the report pipeline.

Each page tracks a top-down "cursor" (y-fraction of the figure, 0=bottom
1=top).  Every element is sized from actual row counts, not guessed
fractions, so nothing can overflow or waste space.

Key sizing rules
────────────────
• _place_table(cursor_top)  renders the table title in the band
  [cursor_top - TITLE_CLEAR_IN, cursor_top], then the table body in
  [cursor_top - TITLE_CLEAR_IN - body_frac, cursor_top - TITLE_CLEAR_IN].
  It returns the body bottom minus GAP (= next element's cursor_top).

• CHART_GAP (> GAP) is used beneath any axes that shows rotated date labels
  so tick text never overlaps the element below.

• When placing a chart above a table, compute the table's top position
  first (cursor_top for the table = CONTENT_BOTTOM + body_frac + title_frac),
  then give the chart everything above that.  Do NOT recompute the table
  height inside the chart layout; that double-counts and causes the overlap.
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from . import widgets as w
from .data import ReportData

CONTENT_TOP    = 0.90   # just below the chrome rule at y=0.95
CONTENT_BOTTOM = 0.06   # just above the chrome footer at y=0.02
GAP            = 0.025  # breathing room between stacked text/table elements
CHART_GAP      = 0.055  # extra clearance below a chart (rotated tick labels) at page bottom
CHART_TO_TITLE_GAP = 0.095 # clearance below a chart if a titled element is directly underneath it


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _in_to_frac(inches: float) -> float:
    return inches / w.PAGE_SIZE[1]


def _body_frac(n_rows: int) -> float:
    """Axes height fraction for n_rows data rows + 1 header row."""
    return _in_to_frac((n_rows + 1) * w.ROW_HEIGHT_IN)


def _title_frac() -> float:
    return _in_to_frac(w.TITLE_CLEAR_IN)


def _table_top_for_bottom(n_rows: int, bottom: float = CONTENT_BOTTOM) -> float:
    """
    Given that we want a table's body bottom to sit at *bottom*,
    return the cursor_top value that _place_table needs.
    cursor_top = body_bottom + body_frac + title_frac
    """
    return bottom + _body_frac(n_rows) + _title_frac()


def _place_table(
    fig: plt.Figure,
    cursor_top: float,
    left: float,
    width: float,
    df: pd.DataFrame,
    title: str,
    index_label: str = "",
    none_labels: dict[str, str] | None = None,
) -> float:
    """
    Render a table anchored at a cursor position and return the next cursor.

    Parameters
    ----------
    fig : plt.Figure
        Page figure to draw onto.
    cursor_top : float
        Figure-fraction y where this element's content block starts. The
        title occupies [cursor_top - title_frac, cursor_top]; the table
        body occupies [cursor_top - title_frac - body_frac, cursor_top - title_frac].
    left, width : float
        Horizontal placement in figure fractions.
    df : pd.DataFrame
        Data to render.
    title : str
        Table title; pass "" to skip reserving title clearance.
    index_label : str, default ""
        Header text for the index-turned-first-column.
    none_labels : dict[str, str], optional
        Forwarded to widgets.render_table — per-column override for how
        None/NaN renders (e.g. {"recovery_date": "Ongoing"}).

    Returns
    -------
    float
        body_bottom - GAP, i.e. the cursor_top the next element should use.
    """
    bf          = _body_frac(len(df))
    tf          = _title_frac() if title else 0.0
    axes_top    = cursor_top - tf
    axes_bottom = axes_top - bf
    w.render_table(fig, (left, axes_bottom, width, bf),
                   df, title=title, index_label=index_label, none_labels=none_labels)
    return axes_bottom - GAP


def _fits(cursor_top: float, df: pd.DataFrame, has_title: bool = True) -> bool:
    """True if *df* can be rendered below *cursor_top* without hitting CONTENT_BOTTOM."""
    need = _body_frac(len(df)) + (_title_frac() if has_title else 0.0)
    return (cursor_top - need) >= CONTENT_BOTTOM


def _fits_chart_and_table(cursor_top: float, df: pd.DataFrame) -> bool:
    """True if both a chart (min height 0.06) and the table can fit below *cursor_top*."""
    need = 0.06 + CHART_TO_TITLE_GAP + _body_frac(len(df)) + _title_frac()
    return (cursor_top - need) >= CONTENT_BOTTOM


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------

# The summary_table has 14 metric columns.  Transposing gives 14 rows × 2
# underlier cols, which is tall but narrow.  Split evenly into left/right
# half-page panels so both panels fit comfortably on the cover.
_SUMMARY_LEFT_METRICS = [
    "total_pnl", "mean_daily_pnl", "std_daily_pnl",
    "best_day", "worst_day", "sharpe", "sortino",
]
_SUMMARY_RIGHT_METRICS = [
    "calmar", "max_drawdown", "max_dd_duration",
    "win_rate", "profit_factor", "avg_holding_secs", "total_trades",
]


def build_cover_page(data: ReportData) -> tuple[plt.Figure, str]:
    fig = w.new_page()

    # ── headline block ────────────────────────────────────────────────────
    # "Options Backtest Report" is the document type — kept small so it
    # doesn't compete with the strategy name, which is what this report is FOR.
    fig.text(0.05, 0.948, "OPTIONS BACKTEST REPORT",
             fontsize=8, weight="bold", color="#9aabb8",
             family="monospace", va="top")

    # Strategy name is the primary headline — large, bold, ink colour
    fig.text(0.05, 0.930, data.strategy_name,
             fontsize=26, weight="bold", color=w.INK, va="top")

    # Period + run count on the line below
    fig.text(0.05, 0.878,
             f"{data.period_label}  \u00b7  {data.n_day_underlier_runs} day-underlier runs",
             fontsize=10, color="#6e8497", va="top")

    # Rule separating the header from the KPI strip
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.862, 0.862],
                              transform=fig.transFigure,
                              color="#d0dae3", linewidth=0.8))

    # ── KPI strip ─────────────────────────────────────────────────────────
    w.kpi_strip(fig, [
        ("Combined PnL",    f"{data.combined_pnl.sum():+,.1f} {data.unit}"),
        ("Total Trades",    f"{data.n_trades:,}"),
        ("Avg Rolls / Day", f"{data.avg_rolls_per_day:.0f}"),
        ("Win Rate",        data.win_rate_label),
    ], y=0.74, height=0.10)

    # ── Summary statistics — two half-page panels side by side ───────────
    st = data.summary_table          # underlier x metric (N rows x 14 cols)

    left_df  = st[_SUMMARY_LEFT_METRICS].T
    right_df = st[_SUMMARY_RIGHT_METRICS].T

    cursor_start = 0.71
    half_w       = 0.435
    gap_mid      = 0.02

    bf = _body_frac(len(left_df))   # both panels have same row count
    tf = _title_frac()

    left_axes_top    = cursor_start - tf
    left_axes_bottom = left_axes_top - bf

    w.render_table(fig, (0.05, left_axes_bottom, half_w, bf),
                   left_df,
                   title="Summary Statistics — Returns & Ratios",
                   index_label="metric")
    w.render_table(fig, (0.05 + half_w + gap_mid, left_axes_bottom, half_w, bf),
                   right_df,
                   title="Summary Statistics — Risk & Activity",
                   index_label="metric")

    cursor = left_axes_bottom - GAP

    # Rule above the break-even line
    fig.add_artist(plt.Line2D([0.05, 0.95], [cursor - 0.005, cursor - 0.005],
                              transform=fig.transFigure,
                              color="#d0dae3", linewidth=0.6))

    # ── Break-even callout ────────────────────────────────────────────────
    be = data.breakeven_bps
    if be >= 0:
        prefix   = "PROFITABLE  |"
        be_text  = f"Break-even transaction cost: {be:.2f} bps per trade."
        be_color = "#1e6e1e"
    else:
        prefix   = "LOSS-MAKING  |"
        be_text  = (f"Needs a {abs(be):.2f} bps/trade subsidy to break even "
                    f"at zero transaction cost.")
        be_color = "#992222"

    fig.text(0.05, cursor - 0.016,
             f"{prefix}  {be_text}",
             fontsize=9, color=be_color, va="top", weight="bold")

    return fig, "Cover"



# ---------------------------------------------------------------------------
# Performance pages
# ---------------------------------------------------------------------------

def build_performance_pages(data: ReportData) -> list[tuple[plt.Figure, str]]:
    pages = []
    daily = data.daily_pnl

    # Page 1: Cumulative PnL + Drawdown chart
    fig = w.new_page()
    gs  = fig.add_gridspec(2, 1, left=0.08, right=0.95,
                           top=CONTENT_TOP, bottom=CONTENT_BOTTOM + 0.04,
                           hspace=0.60)
    w.render_line_chart(
        fig.add_subplot(gs[0]),
        {c: daily[c].cumsum() for c in daily.columns},
        "Cumulative PnL", "PnL",
        combined=data.combined_pnl.cumsum(),
    )
    w.render_drawdown_chart(fig.add_subplot(gs[1]), daily)
    pages.append((fig, "Performance"))

    # Page 2: Daily PnL bar chart
    fig = w.new_page()
    ax  = fig.add_axes((0.08, CONTENT_BOTTOM + CHART_GAP,
                        0.87, CONTENT_TOP - CONTENT_BOTTOM - CHART_GAP))
    melted = daily.reset_index().melt(
        id_vars="date", var_name="underlier", value_name="pnl"
    )
    w.render_bar_chart(ax, melted, "date", "pnl", "underlier",
                       "Daily PnL by Underlier", "PnL")
    pages.append((fig, "Performance"))

    return pages


# ---------------------------------------------------------------------------
# Risk pages
# ---------------------------------------------------------------------------

def build_risk_pages(data: ReportData) -> list[tuple[plt.Figure, str]]:
    """
    Risk page layout (top → bottom):
      1. Risk metrics table
      2. Rolling Sharpe chart — sized to fill whatever is left above …
      3. Largest drawdown table — anchored at the page bottom if it fits,
         otherwise pushed to a fresh second page.

    The key invariant: compute where the drawdown table TOP must be
    (cursor_top = CONTENT_BOTTOM + body_frac + title_frac) FIRST, then
    give the chart everything between there and the risk table cursor.
    Never recalculate the drawdown size inside the chart geometry — that
    was the source of the double-counting overlap bug.
    """
    pages = []
    fig   = w.new_page()

    # 1. Risk metrics table
    cursor = _place_table(
        fig, CONTENT_TOP, 0.05, 0.9, data.risk_table.T,
        title="Risk Metrics — VaR / CVaR / Distribution Shape / Ulcer Index",
        index_label="metric",
    )

    dd = data.drawdown_table

    if _fits_chart_and_table(cursor, dd):
        # ── both chart and drawdown table fit on this page ────────────────
        # Anchor the drawdown table at CONTENT_BOTTOM.
        # dd_cursor_top is the cursor_top argument _place_table expects,
        # i.e. the very top of the title band.
        dd_cursor_top = _table_top_for_bottom(len(dd), CONTENT_BOTTOM)

        # Chart fills [chart_bottom, chart_top] between the two tables.
        # chart_bottom must clear the rotated tick labels of the chart itself
        # and the title of the table below; we add CHART_TO_TITLE_GAP above the drawdown title band.
        chart_top    = cursor - GAP
        chart_bottom = dd_cursor_top + CHART_TO_TITLE_GAP

        chart_h = chart_top - chart_bottom
        if chart_h > 0.06:
            ax = fig.add_axes((0.08, chart_bottom, 0.87, chart_h))
            w.render_line_chart(
                ax,
                {c: data.rolling_sharpe[c] for c in data.rolling_sharpe.columns},
                "Rolling 5-Day Sharpe Ratio", "Sharpe (annualized)",
            )

        _place_table(fig, dd_cursor_top, 0.05, 0.9, dd,
                     title="Largest Drawdown Episodes", index_label="",
                     none_labels={"recovery_date": "Ongoing"})
        pages.append((fig, "Risk"))

    else:
        # ── drawdown table doesn't fit; chart fills this page, table on p2 ─
        chart_top    = cursor - GAP
        chart_bottom = CONTENT_BOTTOM
        chart_h      = chart_top - chart_bottom
        if chart_h > 0.06:
            ax = fig.add_axes((0.08, chart_bottom, 0.87, chart_h))
            w.render_line_chart(
                ax,
                {c: data.rolling_sharpe[c] for c in data.rolling_sharpe.columns},
                "Rolling 5-Day Sharpe Ratio", "Sharpe (annualized)",
            )
        pages.append((fig, "Risk"))

        fig2 = w.new_page()
        _place_table(fig2, CONTENT_TOP, 0.05, 0.9, dd,
                     title="Largest Drawdown Episodes", index_label="",
                     none_labels={"recovery_date": "Ongoing"})
        pages.append((fig2, "Risk"))

    return pages


# ---------------------------------------------------------------------------
# Activity page
# ---------------------------------------------------------------------------

def build_activity_page(data: ReportData) -> tuple[plt.Figure, str]:
    fig = w.new_page()

    if data.total_rolls == 0:
        ax = fig.add_axes((0.05, CONTENT_BOTTOM,
                           0.9, CONTENT_TOP - CONTENT_BOTTOM))
        ax.axis("off")
        ax.set_title("Roll Activity", fontsize=12, weight="bold", loc="left", pad=10)
        ax.text(0.0, 0.85,
                "No rolls — this strategy holds a fixed position without rolling.",
                fontsize=10, color="#666666", transform=ax.transAxes)
    else:
        ax = fig.add_axes((0.08, CONTENT_BOTTOM + CHART_GAP,
                           0.87, CONTENT_TOP - CONTENT_BOTTOM - CHART_GAP))
        w.render_bar_chart(ax, data.rolls_per_day, "date", "rolls", "underlier",
                           "Rolls per Day", "Number of rolls")
    return fig, "Activity"


# ---------------------------------------------------------------------------
# Cost sensitivity page
# ---------------------------------------------------------------------------

def build_cost_page(data: ReportData) -> tuple[plt.Figure, str]:
    """
    Layout (bottom → top):
      • Cost sensitivity table at page bottom
      • Bar chart fills remaining space above it
    """
    fig = w.new_page()
    cs  = data.cost_sensitivity

    # Table is anchored at CONTENT_BOTTOM
    table_body_frac  = _body_frac(len(cs))
    table_title_frac = _title_frac()

    # Axes rect for the table body starts just above the footer + title band
    table_body_bottom = CONTENT_BOTTOM + table_title_frac

    # Chart sits above the table; its bottom clears rotated tick labels & table title
    chart_bottom = table_body_bottom + table_body_frac + CHART_TO_TITLE_GAP
    chart_h      = CONTENT_TOP - chart_bottom

    if chart_h > 0.06:
        ax = fig.add_axes((0.08, chart_bottom, 0.87, chart_h))
        w.render_cost_chart(ax, cs)

    # Render the table (title is placed by ax.set_title inside render_table,
    # which sits above the axes rect automatically via matplotlib's title pad)
    w.render_table(
        fig,
        (0.05, table_body_bottom, 0.9, table_body_frac),
        cs,
        title="Cost Sensitivity — Net PnL at Different Transaction Cost Assumptions",
        index_label="",
    )

    return fig, "Costs"


# ---------------------------------------------------------------------------
# Pattern analysis page
# ---------------------------------------------------------------------------

def build_pattern_page(data: ReportData) -> tuple[plt.Figure, str]:
    """
    Three-panel page: expiry vs non-expiry (top-left), day-of-week (top-right),
    AM vs PM split (bottom full-width).

    These three views together answer: *when* does the strategy win or lose?
    — Which day type?  Which weekday?  Which half of the session?
    """
    fig = w.new_page()

    # ── top row: 2 charts side by side ───────────────────────────────────
    # The top row of charts have x-axis tick labels. The bottom chart has a title.
    # We must separate them by CHART_TO_TITLE_GAP.
    avail    = CONTENT_TOP - CONTENT_BOTTOM - CHART_TO_TITLE_GAP
    top_h    = avail * 0.46
    bot_h    = avail * 0.54
    top_y    = CONTENT_TOP - top_h

    gap_mid  = 0.025
    half_w   = (0.87 - gap_mid) / 2

    ax_expiry  = fig.add_axes((0.08, top_y, half_w, top_h - 0.01))
    ax_weekday = fig.add_axes((0.08 + half_w + gap_mid, top_y, half_w, top_h - 0.01))

    if not data.expiry_day.empty:
        w.render_expiry_bars(ax_expiry, data.expiry_day)
    else:
        ax_expiry.axis("off")
        ax_expiry.text(0.5, 0.5, "No expiry data", ha="center",
                       transform=ax_expiry.transAxes, fontsize=10, color=w.MUTED)

    if not data.pnl_by_weekday.empty:
        w.render_weekday_bars(ax_weekday, data.pnl_by_weekday)
    else:
        ax_weekday.axis("off")

    # ── bottom: AM vs PM full width ───────────────────────────────────────
    if bot_h > 0.06 and not data.morning_afternoon.empty:
        ax_ampm = fig.add_axes((0.08, CONTENT_BOTTOM, 0.87, bot_h))
        w.render_morning_afternoon_bars(ax_ampm, data.morning_afternoon)

    return fig, "Analysis"


# ---------------------------------------------------------------------------
# Statistical validity page
# ---------------------------------------------------------------------------

def build_statistics_page(data: ReportData) -> tuple[plt.Figure, str]:
    """
    Two-panel page: PnL distribution histogram with VaR/CVaR (top half)
    and Sharpe bootstrap CI bar chart (bottom half).

    Together they answer: *how reliable* is the strategy's performance?
    — Is the Sharpe statistically different from zero?
    — How fat are the tails?
    """
    fig = w.new_page()

    # ── top: PnL histogram ────────────────────────────────────────────────
    # Top chart has x-axis labels. Bottom chart has a title.
    # We must separate their axes edges by CHART_TO_TITLE_GAP.
    avail  = CONTENT_TOP - CONTENT_BOTTOM - CHART_TO_TITLE_GAP
    hist_h = avail / 2
    ci_h   = avail / 2
    mid    = CONTENT_BOTTOM + ci_h + CHART_TO_TITLE_GAP

    if hist_h > 0.06:
        ax_hist = fig.add_axes((0.08, mid, 0.87, hist_h))
        w.render_pnl_histogram(ax_hist, data.daily_pnl,
                               data.var_series, data.cvar_series)

    # ── bottom: Sharpe CI ─────────────────────────────────────────────────
    if ci_h > 0.06 and not data.sharpe_ci.empty:
        ax_ci = fig.add_axes((0.08, CONTENT_BOTTOM, 0.87, ci_h))
        w.render_sharpe_ci(ax_ci, data.sharpe_ci)

    return fig, "Statistics"


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_all_pages(data: ReportData) -> list[tuple[plt.Figure, str]]:
    """Return every (figure, section_label) pair in presentation order."""
    pages  = [build_cover_page(data)]
    pages += build_performance_pages(data)
    pages += build_risk_pages(data)
    pages.append(build_activity_page(data))
    pages.append(build_cost_page(data))
    pages.append(build_pattern_page(data))
    pages.append(build_statistics_page(data))
    return pages


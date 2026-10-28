# src/stratbt/report/charts.py
"""Axes-level chart functions for the tearsheet. Each draws into a given Axes."""
from __future__ import annotations

import math

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from stratbt.report import theme as T


def _base(ax, ygrid: bool = True, money_unit: str | None = None, xgrid: bool = False) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(T.RULE)
    ax.tick_params(axis="both", length=0, labelsize=7, pad=4)
    if ygrid:
        ax.grid(axis="y", color=T.GRID, linewidth=0.7)
    if xgrid:
        ax.grid(axis="x", color=T.GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    if money_unit is not None:
        ax.yaxis.set_major_formatter(lambda v, _: _compact(v, money_unit))


def _compact(v: float, unit: str) -> str:
    return T.fcompact(v, unit) if unit not in ("pts", "") else T.fcompact(v, "")


def _dates(ax, n: int) -> None:
    loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def _dodge(items: list[tuple[float, str, str]], min_gap: float) -> list[tuple[float, str, str]]:
    """Nudge end-of-line labels apart so they never overlap."""
    items = sorted(items, key=lambda t: t[0])
    out = []
    for y, txt, col in items:
        if out and y - out[-1][0] < min_gap:
            y = out[-1][0] + min_gap
        out.append((y, txt, col))
    return out


def equity(ax, data) -> None:
    """Cumulative net PnL: one line per underlier, combined in ink, direct end labels."""
    _base(ax, money_unit=data.unit)
    names = data.underliers
    if not len(data.cum):
        return
    start = data.cum.index[0] - pd.Timedelta(days=1)
    series = [(u, data.cum[u]) for u in names]
    if len(names) > 1:
        series.append(("Combined", data.cum_combined))
    ends = []
    for name, s in series:
        s = pd.concat([pd.Series([0.0], index=[start]), s])
        col = T.series_color(name, names)
        ax.plot(pd.to_datetime(s.index), s.values, color=col, linewidth=2.4 if name == "Combined" else 1.8,
                solid_capstyle="round", linestyle="-", zorder=3)
        ends.append((float(s.values[-1]), name, col))
    ax.axhline(0, color=T.INK_2, linewidth=0.8, zorder=2)
    lo, hi = ax.get_ylim()
    for y, name, col in _dodge([(y, f"{n}  {T.fmoney(y, data.unit, signed=True)}", c) for y, n, c in ends],
                               (hi - lo) * 0.075):
        ax.annotate(name, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=T.INK_2, annotation_clip=False)
    ax.set_xlim(pd.to_datetime(start), pd.to_datetime(data.cum.index[-1]))
    _dates(ax, len(data.cum))


def drawdown(ax, data) -> None:
    _base(ax, money_unit=data.unit)
    if not len(data.drawdown):
        return
    x = pd.to_datetime(data.drawdown.index)
    ax.fill_between(x, data.drawdown.values, 0, color=T.LOSS, alpha=0.16, linewidth=0)
    ax.plot(x, data.drawdown.values, color=T.LOSS, linewidth=1.4)
    ax.axhline(0, color=T.INK_2, linewidth=0.8)
    if data.drawdown.min() < 0:
        i = int(np.argmin(data.drawdown.values))
        ax.annotate(T.fmoney(data.drawdown.values[i], data.unit), xy=(x[i], data.drawdown.values[i]),
                    xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=7.5, color=T.INK_2)
        ax.margins(y=0.25)
    ax.set_ylim(top=0)
    _dates(ax, len(x))


def daily_bars(ax, data) -> None:
    _base(ax, money_unit=data.unit)
    if not len(data.combined):
        return
    x = np.arange(len(data.combined))
    v = data.combined.values
    ax.bar(x, v, width=0.72, color=[T.GAIN if a >= 0 else T.LOSS for a in v], zorder=3)
    ax.axhline(0, color=T.INK_2, linewidth=0.8, zorder=4)
    step = max(1, math.ceil(len(x) / 10))
    ax.set_xticks(x[::step])
    ax.set_xticklabels([f"{d:%b %d}" for d in data.combined.index[::step]])
    for i in (int(np.argmax(v)), int(np.argmin(v))):
        if len({int(np.argmax(v)), int(np.argmin(v))}) == 2 or i == int(np.argmax(v)):
            ax.annotate(T.fmoney(v[i], data.unit, signed=True), xy=(x[i], v[i]),
                        xytext=(0, 3 if v[i] >= 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if v[i] >= 0 else "top", fontsize=7, color=T.INK_2)
    ax.margins(y=0.18)


def sharpe_ci(ax, data) -> None:
    """Point estimate with 95% bootstrap interval per series; zero line marked."""
    _base(ax, ygrid=False, xgrid=True)
    df = data.sharpe_ci_df.dropna(how="all")
    if df.empty:
        ax.text(0.5, 0.5, "not enough data", transform=ax.transAxes, ha="center", color=T.MUTED)
        return
    names = list(df.index)
    for i, n in enumerate(names):
        r = df.loc[n]
        col = T.series_color(n, data.underliers)
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=col, linewidth=2.2, solid_capstyle="round")
        ax.plot(r["sharpe"], i, "o", color=col, markersize=7, markeredgecolor=T.SURFACE, markeredgewidth=1.5, zorder=4)
        ax.annotate(f"{T.fnum(r['sharpe'], 1)}  [{T.fnum(r['ci_low'], 1)}, {T.fnum(r['ci_high'], 1)}]",
                    xy=(r["sharpe"], i), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7, color=T.INK_2)
    ax.axvline(0, color=T.INK_2, linewidth=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5, color=T.INK_2)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.invert_yaxis()
    ax.margins(x=0.12)


def histogram(ax, data) -> None:
    _base(ax, money_unit=None)
    v = data.combined.values
    if len(v) < 2:
        return
    bins = int(min(12, max(5, math.sqrt(len(v)) * 1.6)))
    ax.hist(v, bins=bins, color=T.BLUE, alpha=0.85, edgecolor=T.SURFACE, linewidth=1.5, zorder=3)
    ax.axvline(0, color=T.INK_2, linewidth=0.8, zorder=4)
    q = np.quantile(v, 0.05)
    tail = v[v <= q]
    ax.axvline(q, color=T.LOSS, linewidth=1.3, linestyle="--", zorder=4)
    ax.annotate(f"VaR 95%  {T.fmoney(q, data.unit)}", xy=(q, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -2), textcoords="offset points", va="top", fontsize=7, color=T.INK_2)
    if len(tail):
        c = float(tail.mean())
        ax.axvline(c, color=T.LOSS, linewidth=1.3, linestyle=":", zorder=4)
        ax.annotate(f"CVaR  {T.fmoney(c, data.unit)}", xy=(c, 0.88), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", va="top", ha="left", fontsize=7, color=T.INK_2)
    ax.xaxis.set_major_formatter(lambda x, _: _compact(x, data.unit))
    ax.yaxis.set_major_locator(__import__("matplotlib.ticker").ticker.MaxNLocator(integer=True))
    ax.set_ylabel("days", fontsize=7, color=T.MUTED)


def intraday(ax, data) -> None:
    _base(ax, money_unit=data.unit)
    df = data.intraday
    if df.empty:
        return
    names = list(df.columns)
    ends = []
    for u in names:
        col = T.series_color(u, data.underliers)
        ax.plot(df.index, df[u].values, color=col, linewidth=1.6)
        ends.append((float(df[u].dropna().iloc[-1]), u, col))
    ax.axhline(0, color=T.INK_2, linewidth=0.8)
    lo, hi = ax.get_ylim()
    for y, u, col in _dodge(ends, (hi - lo) * 0.09):
        ax.annotate(u, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=T.INK_2, annotation_clip=False)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(df.index[0], df.index[-1])


def _grouped(ax, labels, series: dict[str, list[float]], unit: str, colors: dict[str, str], counts=None) -> None:
    x = np.arange(len(labels))
    n = len(series)
    w = 0.74 / n
    for k, (name, vals) in enumerate(series.items()):
        off = (k - (n - 1) / 2) * w
        ax.bar(x + off, vals, width=w - 0.04, color=colors[name], zorder=3, label=name)
        for xi, v in zip(x + off, vals):
            ax.annotate(T.fmoney(v, unit, signed=True), xy=(xi, v), xytext=(0, 2 if v >= 0 else -2),
                        textcoords="offset points", ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=6.5, color=T.INK_2)
    ax.axhline(0, color=T.INK_2, linewidth=0.8, zorder=4)
    ax.set_xticks(x)
    lab = [f"{l}\nn={c}" for l, c in zip(labels, counts)] if counts is not None else labels
    ax.set_xticklabels(lab, fontsize=7, color=T.INK_2)
    ax.margins(y=0.22)


def am_pm(ax, data) -> None:
    _base(ax, money_unit=data.unit)
    df = data.am_pm
    if df.empty:
        return
    series = {"Morning (09:30–12:45)" if False else "Morning": df["morning_pnl"].tolist(),
              "Afternoon": df["afternoon_pnl"].tolist()}
    colors = {"Morning": T.BLUE, "Afternoon": T.ORANGE}
    _grouped(ax, list(df.index), series, data.unit, colors)
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2, handlelength=1.2)


def _single(ax, labels, values, unit, counts=None) -> None:
    x = np.arange(len(labels))
    ax.bar(x, values, width=0.62, color=[T.GAIN if v >= 0 else T.LOSS for v in values], zorder=3)
    for xi, v in zip(x, values):
        ax.annotate(T.fmoney(v, unit, signed=True), xy=(xi, v), xytext=(0, 2 if v >= 0 else -2),
                    textcoords="offset points", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6.8, color=T.INK_2)
    ax.axhline(0, color=T.INK_2, linewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nn={c}" for l, c in zip(labels, counts)], fontsize=7, color=T.INK_2)
    ax.margins(y=0.22)


def weekday(ax, data) -> None:
    _base(ax, money_unit=data.unit)
    df = data.weekday
    if df.empty:
        return
    _single(ax, [w[:3] for w in df["weekday"]], df["avg_pnl"].tolist(), data.unit, df["count"].astype(int).tolist())


def expiry(ax, data) -> None:
    _base(ax, money_unit=data.unit)
    df = data.expiry
    if df.empty:
        return
    _single(ax, df["day_type"].tolist(), df["avg_pnl"].tolist(), data.unit, df["count"].astype(int).tolist())


def cost_waterfall(ax, data) -> None:
    """Gross -> slippage -> fees -> net, for the Total row."""
    _base(ax, money_unit=data.unit)
    b = data.cost_breakdown
    if b.empty:
        return
    t = b.loc["Total"]
    steps = [("Gross PnL\n(mid prices)", t["gross"], 0.0, "total"),
             ("Slippage", -t["slippage"], t["gross"], "delta"),
             ("Fees", -t["fees"], t["gross"] - t["slippage"], "delta"),
             ("Net PnL", t["net"], 0.0, "total")]
    for i, (lab, v, base, kind) in enumerate(steps):
        if kind == "total":
            col = T.GAIN if v >= 0 else T.LOSS
            ax.bar(i, v, width=0.6, color=col, zorder=3)
            top = v
        else:
            ax.bar(i, v, bottom=base, width=0.6, color=T.MUTED, zorder=3)
            top = base + v
        y_end = v if kind == "total" else base + v
        ax.annotate(T.fmoney(v, data.unit, signed=kind == "delta"), xy=(i, max(y_end, base if kind == "delta" else y_end)),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=T.INK_2)
        if i < len(steps) - 1:
            level = t["gross"] if i == 0 else (t["gross"] - t["slippage"] if i == 1 else t["net"])
            ax.plot([i + 0.3, i + 0.7], [level, level], color=T.MUTED, linewidth=0.8, zorder=2)
    ax.axhline(0, color=T.INK_2, linewidth=0.8, zorder=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([s[0] for s in steps], fontsize=7, color=T.INK_2)
    ax.margins(y=0.2)


def cost_sensitivity(ax, data) -> None:
    """Net PnL as extra proportional cost rises; marks where it crosses zero."""
    _base(ax, money_unit=data.unit)
    cs = data.cost_sens
    if cs.empty:
        return
    ax.plot(cs["cost_bps"], cs["net_pnl"], color=T.BLUE, linewidth=2, marker="o", markersize=4,
            markeredgecolor=T.SURFACE, markeredgewidth=1.2, zorder=3)
    ax.axhline(0, color=T.INK_2, linewidth=0.8, zorder=2)
    be = data.breakeven_bps
    if 0 < be <= cs["cost_bps"].max():
        ax.axvline(be, color=T.MUTED, linewidth=1, linestyle="--")
        ax.annotate(f"break-even {be:.1f} bps", xy=(be, 0), xytext=(5, 6), textcoords="offset points",
                    fontsize=7, color=T.INK_2)
    ax.set_xlabel("additional cost, basis points of premium per fill", fontsize=7, color=T.MUTED)
    ax.margins(y=0.15)

# src/stratbt/report/charts.py
"""
Axes-level chart functions for the tearsheet, drawn with seaborn on matplotlib axes.

Each function draws into a given Axes and never touches pyplot, so reports can be built
from any thread. Seaborn does the statistical drawing (lineplot / barplot / histplot /
rugplot / scatterplot); matplotlib is used only for the things seaborn has no verb for
(waterfall steps, area fills, direct labels).
"""
import math

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from stratbt.report import theme as T


def _base(ax, ygrid: bool = True, money_unit: str | None = None, xgrid: bool = False) -> None:
    sns.despine(ax=ax, left=True, top=True, right=True)
    ax.spines["bottom"].set_color(T.RULE)
    ax.tick_params(axis="both", length=0, labelsize=7, pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if ygrid:
        ax.grid(axis="y", color=T.GRID, linewidth=0.7)
    if xgrid:
        ax.grid(axis="x", color=T.GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    if money_unit is not None:
        ax.yaxis.set_major_formatter(lambda v, _: _compact(v, money_unit))


def _tidy(ax) -> None:
    """Seaborn titles axes from column names after plotting; the page headings already say what each chart is."""
    ax.set_xlabel("")
    ax.set_ylabel("")


def _compact(v: float, unit: str) -> str:
    return T.fcompact(v, unit) if unit not in ("pts", "") else T.fcompact(v, "")


def _dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def _dodge(items: list[tuple[float, str, str]], min_gap: float) -> list[tuple[float, str, str]]:
    """Nudge end-of-line labels apart so they never overlap."""
    out = []
    for y, txt, col in sorted(items, key=lambda t: t[0]):
        if out and y - out[-1][0] < min_gap:
            y = out[-1][0] + min_gap
        out.append((y, txt, col))
    return out


def _palette(names: list[str]) -> dict[str, str]:
    return {n: T.series_color(n) for n in names}


def _sign_palette() -> dict[str, str]:
    return {"gain": T.GAIN, "loss": T.LOSS}


def _label_bars(ax, unit: str, signed: bool = True, size: float = 6.5) -> None:
    for c in ax.containers:
        labels = [T.fmoney(b.get_height(), unit, signed=signed) if b.get_height() else "" for b in c]
        ax.bar_label(c, labels=labels, fontsize=size, color=T.INK_2, padding=2)


# ── performance ──────────────────────────────────────────────────────────

def equity(ax, data) -> None:
    """Cumulative net PnL: one line per underlier, combined in ink, direct end labels."""
    _base(ax, money_unit=data.unit)
    names = data.underliers
    if not len(data.cum):
        return
    start = pd.Timestamp(data.cum.index[0]) - pd.Timedelta(days=1)
    frames, ends = [], []
    series = [(u, data.cum[u]) for u in names]
    for name, s in series:
        s = pd.concat([pd.Series([0.0], index=[start]), s])
        frames.append(pd.DataFrame({"date": pd.to_datetime(s.index), "cum": s.values, "series": name}))
    long = pd.concat(frames)
    sns.lineplot(data=long, x="date", y="cum", hue="series", hue_order=names, palette=_palette(names),
                 linewidth=1.9, legend=False, ax=ax, zorder=3)
    for name, s in series:
        ends.append((float(s.iloc[-1]), name, T.series_color(name)))
    if len(names) > 1:
        c = pd.concat([pd.Series([0.0], index=[start]), data.cum_combined])
        sns.lineplot(x=pd.to_datetime(c.index), y=c.values, color=T.COMBINED, linewidth=2.7, ax=ax, zorder=4)
        ends.append((float(c.values[-1]), "Combined", T.COMBINED))
    ax.axhline(0, color=T.INK_2, linewidth=0.8, zorder=2)
    lo, hi = ax.get_ylim()
    labelled = [(y, f"{n}  {T.fmoney(y, data.unit, signed=True)}", c) for y, n, c in ends]
    for y, txt, _ in _dodge(labelled, (hi - lo) * 0.075):
        ax.annotate(txt, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=T.INK_2, annotation_clip=False)
    ax.set_xlim(start, pd.Timestamp(data.cum.index[-1]))
    _dates(ax)
    _tidy(ax)


def drawdown(ax, data) -> None:
    _base(ax, money_unit=data.unit)
    if not len(data.drawdown):
        return
    x = pd.to_datetime(data.drawdown.index)
    ax.fill_between(x, data.drawdown.values, 0, color=T.LOSS, alpha=0.16, linewidth=0)
    sns.lineplot(x=x, y=data.drawdown.values, color=T.LOSS, linewidth=1.4, ax=ax)
    ax.axhline(0, color=T.INK_2, linewidth=0.8)
    if data.drawdown.min() < 0:
        i = int(np.argmin(data.drawdown.values))
        ax.annotate(T.fmoney(data.drawdown.values[i], data.unit), xy=(x[i], data.drawdown.values[i]),
                    xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=7.5, color=T.INK_2)
        ax.margins(y=0.25)
    ax.set_ylim(top=0)
    _dates(ax)
    _tidy(ax)


def daily_bars(ax, data) -> None:
    _base(ax, money_unit=data.unit)
    if not len(data.combined):
        return
    v = data.combined.values
    df = pd.DataFrame({"i": np.arange(len(v)), "pnl": v, "kind": np.where(v >= 0, "gain", "loss")})
    sns.barplot(data=df, x="i", y="pnl", hue="kind", hue_order=["gain", "loss"], palette=_sign_palette(),
                dodge=False, width=0.72, saturation=1, legend=False, errorbar=None, ax=ax, zorder=3)
    ax.axhline(0, color=T.INK_2, linewidth=0.8, zorder=4)
    step = max(1, math.ceil(len(v) / 10))
    ax.set_xticks(range(0, len(v), step))
    ax.set_xticklabels([f"{d:%b %d}" for d in data.combined.index[::step]])
    ax.set_xlabel("")
    hi, lo = int(np.argmax(v)), int(np.argmin(v))
    for i in {hi, lo}:
        ax.annotate(T.fmoney(v[i], data.unit, signed=True), xy=(i, v[i]), xytext=(0, 3 if v[i] >= 0 else -3),
                    textcoords="offset points", ha="center", va="bottom" if v[i] >= 0 else "top",
                    fontsize=7, color=T.INK_2)
    ax.margins(y=0.18)
    _tidy(ax)


# ── risk ─────────────────────────────────────────────────────────────────

def sharpe_ci(ax, data) -> None:
    """Point estimate with 95% bootstrap interval per series; zero line marked."""
    _base(ax, ygrid=False, xgrid=True)
    df = data.sharpe_ci_df.dropna(how="all")
    if df.empty:
        ax.text(0.5, 0.5, "not enough data", transform=ax.transAxes, ha="center", color=T.MUTED)
        return
    names = list(df.index)
    pos = np.arange(len(names))
    for p, n in zip(pos, names):
        r = df.loc[n]
        ax.plot([r["ci_low"], r["ci_high"]], [p, p], color=T.series_color(n), linewidth=2.4, solid_capstyle="round")
        ax.annotate(f"{T.fnum(r['sharpe'], 1)}  [{T.fnum(r['ci_low'], 1)}, {T.fnum(r['ci_high'], 1)}]",
                    xy=(r["sharpe"], p), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7, color=T.INK_2)
    pts = pd.DataFrame({"sharpe": df["sharpe"].values, "pos": pos, "name": names})
    sns.scatterplot(data=pts, x="sharpe", y="pos", hue="name", palette=_palette(names), s=55, legend=False,
                    edgecolor=T.SURFACE, linewidth=1.5, ax=ax, zorder=4)
    ax.axvline(0, color=T.INK_2, linewidth=0.8)
    ax.set_yticks(pos)
    ax.set_yticklabels(names, fontsize=7.5, color=T.INK_2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.margins(x=0.12)


def histogram(ax, data) -> None:
    _base(ax)
    v = data.combined.values
    if len(v) < 2:
        return
    bins = int(min(12, max(5, math.sqrt(len(v)) * 1.6)))
    sns.histplot(x=v, bins=bins, color=T.BLUE, alpha=0.85, edgecolor=T.SURFACE, linewidth=1.5, ax=ax, zorder=3)
    sns.rugplot(x=v, height=0.035, color=T.INK_2, linewidth=1, ax=ax, zorder=4)   # every actual day, so bin choice can't mislead
    ax.axvline(0, color=T.INK_2, linewidth=0.8, zorder=4)
    q = float(np.quantile(v, 0.05))
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
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _tidy(ax)
    ax.set_ylabel("days", fontsize=7, color=T.MUTED)


# ── behaviour ────────────────────────────────────────────────────────────

def intraday(ax, data) -> None:
    _base(ax, money_unit=data.unit)
    df = data.intraday
    if df.empty:
        return
    names = list(df.columns)
    long = df.reset_index(names=["time"]).melt(id_vars="time", var_name="series", value_name="pnl").dropna()
    sns.lineplot(data=long, x="time", y="pnl", hue="series", hue_order=names, palette=_palette(names),
                 linewidth=1.5, legend=False, ax=ax)
    ends = [(float(df[u].dropna().iloc[-1]), u, T.series_color(u)) for u in names]
    ax.axhline(0, color=T.INK_2, linewidth=0.8)
    lo, hi = ax.get_ylim()
    for y, u, _ in _dodge(ends, (hi - lo) * 0.09):
        ax.annotate(u, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=T.INK_2, annotation_clip=False)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(df.index[0], df.index[-1])
    _tidy(ax)


def am_pm(ax, data) -> None:
    _base(ax, money_unit=data.unit)
    df = data.am_pm
    if df.empty:
        return
    long = (df.rename(columns={"morning_pnl": "Morning", "afternoon_pnl": "Afternoon"})
              .rename_axis("series").reset_index().melt(id_vars="series", var_name="period", value_name="pnl"))
    sns.barplot(data=long, x="series", y="pnl", hue="period", hue_order=["Morning", "Afternoon"],
                palette={"Morning": T.BLUE, "Afternoon": T.ORANGE}, saturation=1, errorbar=None, width=0.74, ax=ax, zorder=3)
    ax.axhline(0, color=T.INK_2, linewidth=0.8, zorder=4)
    _label_bars(ax, data.unit)
    sns.move_legend(ax, "upper right", frameon=False, title=None, fontsize=7, ncol=2, handlelength=1.2)
    _tidy(ax)
    ax.tick_params(axis="x", labelsize=7, labelcolor=T.INK_2)
    ax.margins(y=0.22)


def _single(ax, labels, values, unit, counts) -> None:
    df = pd.DataFrame({"label": [f"{l}\nn={c}" for l, c in zip(labels, counts)], "pnl": values,
                       "kind": np.where(np.array(values) >= 0, "gain", "loss")})
    sns.barplot(data=df, x="label", y="pnl", hue="kind", hue_order=["gain", "loss"], palette=_sign_palette(),
                dodge=False, width=0.62, saturation=1, legend=False, errorbar=None, ax=ax, zorder=3)
    ax.axhline(0, color=T.INK_2, linewidth=0.8, zorder=4)
    _label_bars(ax, unit, size=6.8)
    _tidy(ax)
    ax.tick_params(axis="x", labelsize=7, labelcolor=T.INK_2)
    ax.margins(y=0.22)


def weekday(ax, data) -> None:
    _base(ax, money_unit=data.unit)
    df = data.weekday
    if not df.empty:
        _single(ax, [w[:3] for w in df["weekday"]], df["avg_pnl"].tolist(), data.unit, df["count"].astype(int).tolist())


def expiry(ax, data) -> None:
    _base(ax, money_unit=data.unit)
    df = data.expiry
    if not df.empty:
        _single(ax, df["day_type"].tolist(), df["avg_pnl"].tolist(), data.unit, df["count"].astype(int).tolist())


# ── costs ────────────────────────────────────────────────────────────────

def cost_waterfall(ax, data) -> None:
    """Gross -> slippage -> fees -> net, for the Total row. (Seaborn has no waterfall verb; drawn with bars.)"""
    _base(ax, money_unit=data.unit)
    b = data.cost_breakdown
    if b.empty:
        return
    t = b.loc["Total"]
    steps = [("Gross PnL\n(mid prices)", t["gross"], 0.0, "total"),
             ("Slippage", -t["slippage"], t["gross"], "delta"),
             ("Fees", -t["fees"], t["gross"] - t["slippage"], "delta"),
             ("Net PnL", t["net"], 0.0, "total")]
    levels = [t["gross"], t["gross"] - t["slippage"], t["net"]]
    for i, (_, v, base, kind) in enumerate(steps):
        if kind == "total":
            ax.bar(i, v, width=0.6, color=T.GAIN if v >= 0 else T.LOSS, zorder=3)
            top = v
        else:
            ax.bar(i, v, bottom=base, width=0.6, color=T.MUTED, zorder=3)
            top = base
        ax.annotate(T.fmoney(v, data.unit, signed=kind == "delta"), xy=(i, max(top, top + v if kind == "delta" else top)),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=T.INK_2)
        if i < len(steps) - 1:
            ax.plot([i + 0.3, i + 0.7], [levels[i]] * 2, color=T.MUTED, linewidth=0.8, zorder=2)
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
    sns.lineplot(data=cs, x="cost_bps", y="net_pnl", color=T.BLUE, linewidth=2, marker="o", markersize=5,
                 markeredgecolor=T.SURFACE, markeredgewidth=1.2, ax=ax, zorder=3)
    ax.axhline(0, color=T.INK_2, linewidth=0.8, zorder=2)
    be = data.breakeven_bps
    if 0 < be <= cs["cost_bps"].max():
        ax.axvline(be, color=T.MUTED, linewidth=1, linestyle="--")
        ax.annotate(f"break-even {be:.1f} bps", xy=(be, 0), xytext=(5, 6), textcoords="offset points",
                    fontsize=7, color=T.INK_2)
    _tidy(ax)
    ax.set_xlabel("additional cost, basis points of premium per fill", fontsize=7, color=T.MUTED)
    ax.margins(y=0.15)

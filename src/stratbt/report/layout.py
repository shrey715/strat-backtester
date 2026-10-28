# src/stratbt/report/layout.py
"""
Page layout primitives. All positions are in inches measured from the
top-left corner of the page, so page code reads like a layout spec.
"""
from __future__ import annotations

import textwrap

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from stratbt.report import theme as T


class Page:
    def __init__(self, section: str):
        self.section = section
        # Object API, not pyplot: no GUI backend is ever touched, so building a report is safe from any thread
        # (pyplot would try to open a Tk window and abort the whole process in a server worker).
        self.fig = Figure(figsize=(T.PAGE_W, T.PAGE_H), facecolor=T.SURFACE)

    # ---- coordinates ---------------------------------------------------
    def _fx(self, x: float) -> float:
        return x / T.PAGE_W

    def _fy(self, y: float) -> float:
        return 1.0 - y / T.PAGE_H

    def ax(self, x: float, y: float, w: float, h: float) -> Axes:
        """Axes whose top-left is (x, y) inches and size is w x h inches."""
        return self.fig.add_axes([self._fx(x), self._fy(y + h), w / T.PAGE_W, h / T.PAGE_H])

    # ---- primitives ----------------------------------------------------
    def text(self, x: float, y: float, s: str, size: float = 9, color: str = T.INK,
             weight: str = "normal", ha: str = "left", va: str = "top",
             wrap: float | None = None, line_h: float = 1.35, style: str = "normal") -> float:
        """Draw text; with `wrap` (inches) it is word-wrapped. Returns the y below the text."""
        if wrap is not None:
            chars = max(10, int(wrap * 72 / (size * 0.56)))
            s = "\n".join(textwrap.fill(p, chars) for p in s.split("\n"))
        n_lines = s.count("\n") + 1
        self.fig.text(self._fx(x), self._fy(y), s, fontsize=size, color=color, weight=weight,
                      ha=ha, va=va, linespacing=line_h, fontstyle=style)
        return y + n_lines * size * line_h / 72

    def rule(self, x: float, y: float, w: float, color: str = T.RULE, lw: float = 0.6) -> None:
        self.fig.add_artist(Line2D([self._fx(x), self._fx(x + w)], [self._fy(y)] * 2,
                                   transform=self.fig.transFigure, color=color, linewidth=lw))

    def box(self, x: float, y: float, w: float, h: float, fill: str = T.CARD,
            edge: str | None = None) -> None:
        pad = 0.0
        patch = FancyBboxPatch((self._fx(x), self._fy(y + h)), w / T.PAGE_W, h / T.PAGE_H,
                               boxstyle=f"round,pad={pad},rounding_size=0.01",
                               transform=self.fig.transFigure, facecolor=fill,
                               edgecolor=edge or "none", linewidth=0.6, mutation_aspect=T.PAGE_W / T.PAGE_H)
        self.fig.add_artist(patch)

    def heading(self, x: float, y: float, s: str, w: float | None = None, sub: str | None = None) -> float:
        """Section heading with an optional one-line takeaway below it. Returns next y."""
        yy = self.text(x, y, s, size=10.5, weight="bold")
        if sub:
            yy = self.text(x, yy + 0.01, sub, size=8, color=T.INK_2, wrap=w)
        return yy + 0.05

    def legend(self, x: float, y: float, items: list[tuple[str, str]], gap: float = 0.18) -> None:
        """Horizontal swatch+label legend, left-aligned at (x, y)."""
        cx = x
        for label, color in items:
            self.fig.add_artist(Line2D([self._fx(cx), self._fx(cx + 0.22)], [self._fy(y + 0.06)] * 2,
                                       transform=self.fig.transFigure, color=color, linewidth=2.2,
                                       solid_capstyle="round"))
            self.text(cx + 0.28, y, label, size=7.5, color=T.INK_2)
            cx += 0.28 + 0.065 * len(label) + gap

    # ---- page furniture -----------------------------------------------
    def chrome(self, strategy: str, page_no: int, total: int, generated: str, version: str) -> None:
        self.text(T.MARGIN, 0.32, strategy, size=8, weight="bold")
        self.text(T.PAGE_W - T.MARGIN, 0.32, self.section.upper(), size=7, color=T.MUTED, ha="right")
        self.rule(T.MARGIN, 0.52, T.CONTENT_W)
        self.rule(T.MARGIN, T.PAGE_H - 0.5, T.CONTENT_W)
        self.text(T.MARGIN, T.PAGE_H - 0.42, f"stratbt {version}  ·  generated {generated}",
                  size=7, color=T.MUTED)
        self.text(T.PAGE_W - T.MARGIN, T.PAGE_H - 0.42, f"Page {page_no} of {total}",
                  size=7, color=T.MUTED, ha="right")

    # ---- composite elements --------------------------------------------
    def kpi_cards(self, x: float, y: float, w: float, h: float,
                  cards: list[tuple[str, str, str]], gap: float = 0.14) -> None:
        """Row of (label, value, sub-label) cards filling width w."""
        n = len(cards)
        cw = (w - gap * (n - 1)) / n
        for i, (label, value, sub) in enumerate(cards):
            cx = x + i * (cw + gap)
            self.box(cx, y, cw, h)
            self.text(cx + 0.14, y + 0.13, label.upper(), size=6.5, color=T.MUTED, weight="bold")
            self.text(cx + 0.14, y + 0.36, value, size=15, weight="bold")
            if sub:
                self.text(cx + 0.14, y + h - 0.2, sub, size=6.8, color=T.INK_2)

    def kv(self, x: float, y: float, w: float, rows: list[tuple[str, str]],
           key_w: float = 1.15, size: float = 8, row_h: float = 0.235) -> float:
        """Key/value list with hairlines. Long values wrap. Returns the y below it."""
        for k, v in rows:
            self.rule(x, y, w)
            self.text(x, y + 0.06, k, size=size - 0.5, color=T.MUTED)
            end = self.text(x + key_w, y + 0.055, v, size=size, wrap=w - key_w, line_h=1.25)
            y = max(y + row_h, end + 0.04)
        self.rule(x, y, w)
        return y

    def table(self, x: float, y: float, headers: list[str], rows: list[list[str]],
              col_w: list[float], align: list[str] | None = None, row_h: float = 0.215,
              size: float = 7.8, bold_last: bool = False, bold_rows: set[int] | None = None) -> float:
        """
        Minimal table: muted header over a rule, hairline-separated rows, no fills.
        `align` is per column 'l' or 'r' (default: first left, rest right). Returns y below.
        """
        n = len(headers)
        align = align or (["l"] + ["r"] * (n - 1))
        xs = [x]
        for w in col_w[:-1]:
            xs.append(xs[-1] + w)

        def put(i: int, s: str, yy: float, **kw):
            cx = xs[i] + (col_w[i] - 0.08 if align[i] == "r" else 0.04)
            self.text(cx, yy, s, ha="right" if align[i] == "r" else "left", va="center", **kw)

        for i, h in enumerate(headers):
            put(i, h, y + row_h / 2, size=size - 1, color=T.MUTED, weight="bold")
        y += row_h
        self.rule(x, y, sum(col_w), color=T.INK_2, lw=0.8)
        for r, row in enumerate(rows):
            bold = (bold_last and r == len(rows) - 1) or (bold_rows is not None and r in bold_rows)
            for i, cell in enumerate(row):
                put(i, str(cell), y + row_h / 2, size=size, weight="bold" if bold else "normal",
                    color=T.INK if i else T.INK)
            y += row_h
            self.rule(x, y, sum(col_w), lw=0.4)
        return y

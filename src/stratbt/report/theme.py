# src/stratbt/report/theme.py
"""Design tokens and number formatting for the PDF tearsheet."""
from __future__ import annotations

import math

from stratbt.palette import SERIES_COLORS

# --- page geometry (inches, US-letter landscape) -------------------------
PAGE_W, PAGE_H = 11.0, 8.5
MARGIN = 0.6
CONTENT_W = PAGE_W - 2 * MARGIN

# --- colour tokens (light surface; see the dataviz reference palette) ----
SURFACE = "#ffffff"
CARD = "#f5f4f0"
INK = "#1a1a19"          # primary text
INK_2 = "#52514e"        # secondary text
MUTED = "#8b8a85"        # captions, axis labels
RULE = "#dcdbd6"         # hairlines
GRID = "#ebeae6"         # chart gridlines

BLUE = "#2a78d6"         # categorical slot 1 / gain pole
ORANGE = "#eb6834"       # slot 2
AQUA = "#1baf7a"         # slot 3
GAIN = BLUE              # diverging pair: blue <-> red
LOSS = "#e34948"
COMBINED = INK



def series_color(name: str, names: list[str] | None = None) -> str:
    """Colour for a series, stable by name (shared with the dashboard). `names` is accepted for compatibility."""
    return COMBINED if name == "Combined" else SERIES_COLORS[name]


RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.unicode_minus": True,
    "text.parse_math": False,    # "$" is a currency sign here, not mathtext
    "axes.grid": False,          # seaborn's global style (imported by metrics) must not leak in
    "axes.axisbelow": True,
    "axes.edgecolor": RULE,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,          # embed TrueType: selectable text, crisp at any zoom
}

MINUS = "−"
DASH = "—"


def _bad(v) -> bool:
    return v is None or (isinstance(v, float) and (math.isnan(v)))


def fnum(v, d: int = 2, signed: bool = False) -> str:
    """Plain number with thousands separators; NaN -> dash, inf -> infinity sign."""
    if _bad(v):
        return DASH
    if isinstance(v, float) and math.isinf(v):
        return "∞" if v > 0 else f"{MINUS}∞"
    s = f"{abs(v):,.{d}f}"
    if v < 0 and float(s.replace(",", "")) != 0:
        return MINUS + s
    return ("+" + s) if signed and float(s.replace(",", "")) != 0 else s


def fmoney(v, unit: str = "pts", d: int = 0, signed: bool = False) -> str:
    """Currency-aware amount: $1,234 / ₹1,234 / 1,234 pts."""
    if _bad(v):
        return DASH
    sym = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£"}.get(unit)
    body = fnum(abs(v), d)
    sign = MINUS if v < 0 and body.strip("0.,") else ("+" if signed and body.strip("0.,") else "")
    return f"{sign}{sym}{body}" if sym else f"{sign}{body} {unit}"


def fpct(v, d: int = 0) -> str:
    return DASH if _bad(v) else f"{v * 100:.{d}f}%"


def fduration(seconds) -> str:
    if _bad(seconds):
        return DASH
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def fcompact(v, unit: str = "pts") -> str:
    """Short amount for tight spaces: $10.4k, $1.2M, 950 pts."""
    if _bad(v):
        return DASH
    a = abs(v)
    body = f"{a / 1e6:.1f}M" if a >= 1e6 else f"{a / 1e3:.1f}k" if a >= 1e4 else f"{a:,.0f}"
    sym = {"USD": "$", "INR": "\u20b9", "EUR": "\u20ac", "GBP": "\u00a3"}.get(unit)
    sign = MINUS if v < 0 else ""
    return f"{sign}{sym}{body}" if sym else f"{sign}{body}" + ("" if unit in ("pts", "") else f" {unit}")

# src/stratbt/palette.py
"""
One categorical palette for the dashboard and the PDF.

Colours are the validated light-surface categorical slots (blue, orange,
aqua, violet, magenta, green): adjacent-pair colour-blind separation passes.
A name keeps its colour for the life of the process — dropping a series
never repaints the others.
"""
from itertools import cycle

INK = "#1a1a19"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
_CYCLE = [BLUE, ORANGE, AQUA, "#6250d6", "#e87ba4", "#008300"]


class SeriesColors(dict):
    """underlier -> colour; unseen names get the next palette colour on first use."""

    def __init__(self):
        super().__init__({"SPY": BLUE, "QQQ": ORANGE, "NIFTY": BLUE, "BANKNIFTY": ORANGE, "Combined": INK})
        self._next = cycle(_CYCLE[2:])

    def get(self, key, default=None):
        if key not in self:
            self[key] = next(self._next)
        return self[key]

    def __missing__(self, key):
        return self.get(key)


SERIES_COLORS = SeriesColors()

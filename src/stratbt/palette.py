# src/stratbt/palette.py
from itertools import cycle

INK = "#2c3e50"
_CYCLE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]


class SeriesColors(dict):
    """
    underlier -> colour. Known names keep stable colours; any new name is
    assigned the next palette colour on first use, so charts for arbitrary
    underliers never fall back to the same grey.
    """

    def __init__(self):
        super().__init__({"NIFTY": _CYCLE[0], "BANKNIFTY": _CYCLE[1], "SPY": _CYCLE[0],
                          "QQQ": _CYCLE[1], "Combined": INK})
        self._next = cycle(_CYCLE[2:])

    def get(self, key, default=None):
        if key not in self:
            self[key] = next(self._next)
        return self[key]

    def __missing__(self, key):
        return self.get(key)


SERIES_COLORS = SeriesColors()

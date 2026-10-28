# src/stratbt/providers/nse.py
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd

from stratbt.instruments import Instrument
from stratbt.loader import load_futures_1sec, load_option_1sec
from stratbt.providers.base import DataProvider
from stratbt.universe import Universe


class NSETickProvider(DataProvider):
    """
    NSE index options from a folder of raw tick CSVs laid out as
    `<data_dir>/NSE_YYYYMMDD/{Options,Futures (Continuous)}/...`.
    The underlying price is the front-month continuous future (`<UL>-I.csv`).

    Parameters
    ----------
    data_dir : Path
        Folder holding the `NSE_YYYYMMDD` day directories.
    underliers : list[str], optional
        Defaults to ["NIFTY", "BANKNIFTY"].
    lot_sizes : dict[str, float], optional
        Contract multipliers, e.g. {"NIFTY": 50, "BANKNIFTY": 25}. Without
        it PnL is reported in index points (multiplier 1).
    """

    name = "nse"
    session_open = time(9, 15, 0)
    session_close = time(15, 29, 59)

    def __init__(
        self,
        data_dir: Path | str,
        underliers: list[str] | None = None,
        lot_sizes: dict[str, float] | None = None,
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"NSE data directory not found: {self.data_dir}")
        self._universe = Universe(self.data_dir, underliers)
        self._lot_sizes = dict(lot_sizes or {})
        self.unit = "INR" if lot_sizes else "pts"

    @property
    def universe(self) -> Universe:
        """The underlying day/expiry/strike index (handy for exploration)."""
        return self._universe

    @property
    def underliers(self) -> list[str]:
        return sorted(self._universe.underliers)

    @property
    def trading_days(self) -> list[date]:
        return self._universe.trading_days

    def front_expiry(self, day: date, underlier: str) -> date:
        return self._universe.front_expiry(day, underlier)

    def strikes(self, day: date, underlier: str, expiry: date) -> np.ndarray:
        return self._universe.strikes(day, underlier, expiry)

    def underlying_prices(self, day: date, underlier: str) -> pd.Series:
        return load_futures_1sec(self._universe.day_dir(day), underlier)

    def option_prices(self, day: date, inst: Instrument) -> pd.Series | None:
        return load_option_1sec(self._universe.day_dir(day), inst)

    def multiplier(self, underlier: str) -> float:
        return float(self._lot_sizes.get(underlier, 1.0))

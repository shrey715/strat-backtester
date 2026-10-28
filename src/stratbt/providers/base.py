# src/stratbt/providers/base.py
from abc import ABC, abstractmethod
from datetime import date, datetime, time

import numpy as np
import pandas as pd

from stratbt.instruments import Instrument


class DataProvider(ABC):
    """
    Everything the engine needs to know about a market: which days exist,
    which expiries/strikes were listed, and a 1-second price series for the
    underlying and for each option.

    Implement this to plug in any market or data vendor — the engine, the
    strategies and the metrics never touch files or vendor APIs directly.
    Every price series returned must be indexed by `session_grid(day)`.
    """

    #: Short identifier shown in logs/UI.
    name: str = "provider"
    #: Session bounds (inclusive, 1-second resolution) in exchange-local time.
    session_open: time = time(9, 15, 0)
    session_close: time = time(15, 29, 59)
    #: Label for PnL numbers when multiplier == 1 ("pts") or currency ("USD").
    unit: str = "pts"

    @property
    @abstractmethod
    def underliers(self) -> list[str]:
        """Underliers this provider can serve, e.g. ["NIFTY", "BANKNIFTY"]."""

    @property
    @abstractmethod
    def trading_days(self) -> list[date]:
        """All available trading days, ascending."""

    @abstractmethod
    def front_expiry(self, day: date, underlier: str) -> date:
        """Nearest expiry on or after `day`. Raises ValueError if no data."""

    @abstractmethod
    def strikes(self, day: date, underlier: str, expiry: date) -> np.ndarray:
        """Sorted int strikes listed for that day/underlier/expiry. Raises ValueError if unknown."""

    @abstractmethod
    def underlying_prices(self, day: date, underlier: str) -> pd.Series:
        """Last-known underlying price at every second of the session; no NaNs."""

    @abstractmethod
    def option_prices(self, day: date, inst: Instrument) -> pd.Series | None:
        """
        Last-known option price at every second of the session (leading NaNs
        allowed before the first quote), or None if the contract does not exist.
        """

    def multiplier(self, underlier: str) -> float:
        """Currency value of one price point per contract (1.0 => PnL in points)."""
        return 1.0

    def session_grid(self, day: date) -> pd.DatetimeIndex:
        """The 1-second timestamps of the trading session on `day`."""
        return pd.date_range(
            datetime.combine(day, self.session_open),
            datetime.combine(day, self.session_close),
            freq="1s",
        )

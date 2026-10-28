# src/stratbt/strategy/fixed_window.py
import numpy as np
import pandas as pd
from datetime import date, time

from stratbt.instruments import Instrument
from stratbt.strategy.base import BaseStrategy, Order, nearest_strike


class FixedWindowStraddleStrategy(BaseStrategy):
    """
    Enter the ATM straddle once, N seconds after open; hold without rolling;
    exit M seconds before close. Zero intraday rolls — isolates the
    entry/exit timing question from the roll-frequency question, and serves
    as a baseline comparison against the always-rolling ATM straddle.
    """

    SESSION_OPEN  = time(9, 15, 0)
    SESSION_CLOSE = time(15, 29, 59)

    def __init__(
        self,
        underlier:     str,
        expiry:        date,
        strikes:       np.ndarray,
        entry_delay_s: int = 300,
        exit_buffer_s: int = 300,
        session_open:  time | None = None,
        session_close: time | None = None,
    ):
        """
        Parameters
        ----------
        underlier : str
            e.g. "NIFTY" or "BANKNIFTY".
        expiry : date
            Front-week expiry this strategy trades for the whole session.
        strikes : np.ndarray
            Sorted strikes available at that expiry (from Universe.strikes).
        entry_delay_s : int, default 300
            Seconds after 09:15:00 to wait before entering (300 = 5 min).
        exit_buffer_s : int, default 300
            Seconds before the session close to force-close (300 = 5 min).
        session_open, session_close : time, optional
            Session bounds; default to the NSE session (09:15:00 / 15:29:59).
        """
        self.SESSION_OPEN  = session_open  or type(self).SESSION_OPEN
        self.SESSION_CLOSE = session_close or type(self).SESSION_CLOSE
        self.underlier     = underlier
        self.expiry        = expiry
        self.strikes       = strikes
        self.entry_delay_s = entry_delay_s
        self.exit_buffer_s = exit_buffer_s
        self._entered      = False
        self._entry_ts     = None

    def _nearest_strike(self, futures_price: float) -> int:
        """
        Parameters
        ----------
        futures_price : float

        Returns
        -------
        int
            The strike in self.strikes closest to futures_price, found via
            binary search. Ties are broken toward the lower strike.
        """
        return nearest_strike(self.strikes, futures_price)

    def on_second(
        self,
        ts:            pd.Timestamp,
        futures_price: float,
        option_price,
        holdings:      dict,
    ) -> list[Order]:
        """
        Parameters
        ----------
        ts : pd.Timestamp
        futures_price : float
        option_price : Callable[[Instrument], float | None]
        holdings : dict[Instrument, tuple[float, int]]

        Returns
        -------
        list[Order]
            SELL orders at the exit deadline if holding; a one-time BUY pair
            when the entry window opens and both legs have a price;
            otherwise empty.
        """
        so, sc         = self.SESSION_OPEN, self.SESSION_CLOSE
        session_open   = ts.replace(hour=so.hour, minute=so.minute, second=so.second, microsecond=0)
        session_close  = ts.replace(hour=sc.hour, minute=sc.minute, second=sc.second, microsecond=0)
        entry_deadline = session_open  + pd.Timedelta(seconds=self.entry_delay_s)
        exit_deadline  = session_close - pd.Timedelta(seconds=self.exit_buffer_s)

        # force-close
        if ts >= exit_deadline and holdings:
            return [Order(inst, "SELL") for inst in holdings]

        # already holding — do not roll, just hold
        if holdings or self._entered:
            return []

        # only enter inside the window [entry_deadline, exit_deadline)
        if ts < entry_deadline or ts >= exit_deadline:
            return []

        # enter once
        strike = self._nearest_strike(futures_price)
        ce     = Instrument(self.underlier, self.expiry, strike, "CE")
        pe     = Instrument(self.underlier, self.expiry, strike, "PE")

        if option_price(ce) is None or option_price(pe) is None:
            return []

        self._entered  = True
        self._entry_ts = ts
        return [Order(ce, "BUY"), Order(pe, "BUY")]

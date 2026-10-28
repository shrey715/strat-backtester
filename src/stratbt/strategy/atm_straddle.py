# src/stratbt/strategy/atm_straddle.py
import numpy as np
import pandas as pd
from datetime import date, time
from typing import Callable
from stratbt.instruments import Instrument
from stratbt.strategy.base import BaseStrategy, Order, nearest_strike

class ATMStraddleStrategy(BaseStrategy):
    """
    Dynamically-rolling ATM straddle on the front-week expiry.

    Every second: find the strike nearest the current futures price. If
    flat, enter that straddle. If holding the wrong strike, roll (sell old
    legs, buy new ones) in the same second — provided both new legs have a
    price; otherwise wait rather than force a trade with no fill price.
    Force-closes everything at DAY_CLOSE_TIME.
    """

    DAY_CLOSE_TIME = time(15, 29, 59) # 3:29:59 PM

    def __init__(self, underlier: str, expiry: date, strikes: np.ndarray,
                 session_close: time | None = None):
        """
        Parameters
        ----------
        underlier : str
            e.g. "NIFTY" or "BANKNIFTY".
        expiry : date
            Front-week expiry this strategy trades for the whole session.
        strikes : np.ndarray
            Sorted strikes available at that expiry (from the provider).
        session_close : time, optional
            Last second of the session (positions are force-closed then).
            Defaults to DAY_CLOSE_TIME (15:29:59, the NSE close).
        """
        self.underlier = underlier
        self.expiry = expiry
        self.strikes = strikes
        self.DAY_CLOSE_TIME = session_close or type(self).DAY_CLOSE_TIME

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

    def _make_legs(self, strike: int) -> tuple[Instrument, Instrument]:
        """
        Parameters
        ----------
        strike : int

        Returns
        -------
        tuple[Instrument, Instrument]
            (call, put) at this strike, at self.expiry.
        """
        call = Instrument(self.underlier, self.expiry, strike, 'CE')
        put = Instrument(self.underlier, self.expiry, strike, 'PE')
        return call, put

    def _held_strike(self, holdings: dict[Instrument, tuple[float, int]]) -> int | None:
        """
        Parameters
        ----------
        holdings : dict[Instrument, tuple[float, int]]

        Returns
        -------
        int or None
            The strike currently held (CE and PE always share one strike,
            so the first instrument found is sufficient), or None if flat.
        """
        for inst in holdings:
            return inst.strike
        return None

    def on_second(self, ts: pd.Timestamp, futures_price: float, option_price: Callable[[Instrument], float | None], holdings: dict[Instrument, tuple[float, int]],) -> list[Order]:
        # force-close before the market closes
        if ts.time() >= self.DAY_CLOSE_TIME:
            if holdings:
                return [Order(inst, "SELL") for inst in holdings]
            return []

        # find target strike
        target = self._nearest_strike(futures_price)
        target_ce, target_pe = self._make_legs(target)

        ce_price = option_price(target_ce)
        pe_price = option_price(target_pe)

        if ce_price is None or pe_price is None:
            # cannot trade if either leg has no price
            return []

        held = self._held_strike(holdings)
        if held is None:
            # flat → enter target straddle
            return [Order(target_ce, "BUY"), Order(target_pe, "BUY")]

        if held == target:
            # holding correct strike → hold
            return []

        old_ce, old_pe = self._make_legs(held)
        return [Order(old_ce, "SELL"), Order(old_pe, "SELL"), Order(target_ce, "BUY"), Order(target_pe, "BUY")]

# src/stratbt/providers/modelled.py
"""
Base class for providers whose option prices are *modelled*: the underlying
series comes from somewhere (a simulator, Yahoo Finance, ...) and options are
priced with Black-Scholes at every second.

Time to expiry runs on the same clock the underlying's volatility accrues on
— trading time (252 sessions a year) — not calendar time. Mixing the two
makes options decay several times too slowly during the session relative to
the moves actually simulated, which hands free money to long-gamma strategies.
"""
from abc import abstractmethod
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.special import ndtr

from stratbt.instruments import Instrument
from stratbt.providers.base import DataProvider

TRADING_DAYS_PER_YEAR = 252


def bs_price(spot: np.ndarray, strike: float, t_years: np.ndarray, vol: float,
             rate: float, is_call: bool) -> np.ndarray:
    """Vectorised Black-Scholes European option price."""
    t = np.maximum(t_years, 1e-9)
    sqrt_t = np.sqrt(t)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc = np.exp(-rate * t)
    if is_call:
        return spot * ndtr(d1) - strike * disc * ndtr(d2)
    return strike * disc * ndtr(-d2) - spot * ndtr(-d1)


def nice_strike_step(price: float) -> int:
    """Roughly 0.25% of the price, snapped to a conventional strike increment."""
    steps = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
    target = max(price * 0.0025, 1.0)
    return min(steps, key=lambda s: abs(np.log(s / target)))


class ModelPricedProvider(DataProvider):
    """Subclasses supply the underlying path and an implied-vol assumption; this class does the rest."""

    expiry_weekday: int = 4          # 0 = Mon … 4 = Fri
    rate: float = 0.04
    price_tick: float = 0.01
    max_expiry_weeks: int = 6

    # ---- hooks ----------------------------------------------------------
    @abstractmethod
    def _has(self, day: date, underlier: str) -> bool:
        """Whether there is data for this day/underlier."""

    @abstractmethod
    def _spot(self, day: date, underlier: str) -> pd.Series:
        """Underlying price on the full 1-second session grid, no NaNs."""

    @abstractmethod
    def _iv(self, day: date, underlier: str) -> float:
        """Annualised implied volatility to price options with on `day`."""

    @abstractmethod
    def _strike_step(self, underlier: str) -> int:
        """Distance between listed strikes."""

    # ---- DataProvider ----------------------------------------------------
    def _check(self, day: date, underlier: str) -> None:
        if not self._has(day, underlier):
            raise ValueError(f"No data for day {day} and underlier {underlier}")

    def _expiries(self, day: date) -> list[date]:
        first = day + timedelta(days=(self.expiry_weekday - day.weekday()) % 7)
        return [first + timedelta(weeks=i) for i in range(self.max_expiry_weeks)]

    def front_expiry(self, day: date, underlier: str) -> date:
        self._check(day, underlier)
        return self._expiries(day)[0]

    def strikes(self, day: date, underlier: str, expiry: date) -> np.ndarray:
        self._check(day, underlier)
        if expiry not in self._expiries(day):
            raise ValueError(f"No data for day {day}, underlier {underlier} and expiry {expiry}")
        step = self._strike_step(underlier)
        path = self._spot(day, underlier)
        lo = int(np.floor(path.min() * 0.95 / step)) * step
        hi = int(np.ceil(path.max() * 1.05 / step)) * step
        return np.arange(lo, hi + step, step, dtype=int)

    def underlying_prices(self, day: date, underlier: str) -> pd.Series:
        self._check(day, underlier)
        return self._spot(day, underlier).copy()

    def option_prices(self, day: date, inst: Instrument) -> pd.Series | None:
        if not self._has(day, inst.underlier) or inst.expiry not in self._expiries(day):
            return None
        if inst.strike not in self.strikes(day, inst.underlier, inst.expiry):
            return None
        spot = self._spot(day, inst.underlier)
        n = len(spot)
        left_today = n - np.arange(n)                     # session seconds remaining
        future_days = int(np.busday_count(day + timedelta(days=1), inst.expiry + timedelta(days=1)))
        t_years = (left_today + future_days * n) / (TRADING_DAYS_PER_YEAR * n)
        px = bs_price(spot.to_numpy(), float(inst.strike), t_years, self._iv(day, inst.underlier),
                      self.rate, inst.option_type == "CE")
        tick = self.price_tick
        return pd.Series(np.maximum(np.round(px / tick) * tick, tick), index=spot.index, name=inst.filename)

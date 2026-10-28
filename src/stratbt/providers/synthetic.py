# src/stratbt/providers/synthetic.py
"""
Self-contained market simulator so the package works with zero downloads.

The underlying follows a geometric Brownian motion sampled every second;
options are priced with Black-Scholes at the matching second. It is a
sandbox for developing and testing strategies and plumbing — results say
nothing about real markets.
"""
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
from scipy.special import ndtr

from stratbt.instruments import Instrument
from stratbt.providers.base import DataProvider

# underlier -> (starting spot, strike step)
DEFAULT_UNDERLIERS: dict[str, tuple[float, int]] = {
    "SPY": (450.0, 1),
    "QQQ": (380.0, 1),
}
_SECONDS_PER_YEAR = 365.25 * 86_400
_TRADING_DAYS_PER_YEAR = 252
_MAX_EXPIRY_WEEKS = 6


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


class SyntheticProvider(DataProvider):
    """
    Parameters
    ----------
    n_days : int
        Number of weekdays to simulate, starting at `start`.
    start : date
        First trading day (rolled forward to a weekday).
    seed : int
        Same seed => identical market, every time.
    annual_vol : float
        Realised annualised volatility of the underlying.
    iv_premium : float
        Implied vol = annual_vol * (1 + iv_premium). > 0 makes options
        systematically rich (long-vol strategies bleed, short-vol earn).
    rate : float
        Risk-free rate used in pricing.
    underliers : dict, optional
        name -> (spot, strike_step); defaults to SPY and QQQ lookalikes.
    expiry_weekday : int
        0=Mon … 4=Fri; options expire at the close of the next such day.
    """

    name = "synthetic"
    session_open = time(9, 30, 0)
    session_close = time(15, 59, 59)
    unit = "USD"

    def __init__(
        self,
        n_days: int = 21,
        start: date = date(2024, 1, 2),
        seed: int = 7,
        annual_vol: float = 0.20,
        iv_premium: float = 0.0,
        rate: float = 0.04,
        underliers: dict[str, tuple[float, int]] | None = None,
        expiry_weekday: int = 4,
    ):
        if n_days < 1:
            raise ValueError("n_days must be >= 1")
        self.seed, self.annual_vol, self.iv_premium = seed, annual_vol, iv_premium
        self.rate, self.expiry_weekday = rate, expiry_weekday
        self._specs = dict(underliers or DEFAULT_UNDERLIERS)

        days: list[date] = []
        d = start
        while len(days) < n_days:
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
        self._days = days

        self._paths: dict[tuple[date, str], pd.Series] = {}
        self._simulate()

    # ---- simulation ----------------------------------------------------
    def _simulate(self) -> None:
        n = len(self.session_grid(self._days[0]))
        dt = 1.0 / (_TRADING_DAYS_PER_YEAR * n)
        for k, (ul, (spot0, _step)) in enumerate(sorted(self._specs.items())):
            rng = np.random.default_rng([self.seed, k])
            spot = spot0
            for day in self._days:
                day_vol = self.annual_vol * float(np.exp(rng.normal(0.0, 0.25)))
                spot *= float(np.exp(rng.normal(0.0, 0.3 * day_vol / np.sqrt(_TRADING_DAYS_PER_YEAR))))
                z = rng.standard_normal(n)
                logret = (-0.5 * day_vol**2) * dt + day_vol * np.sqrt(dt) * z
                path = np.round(spot * np.exp(np.cumsum(logret)), 2)
                spot = float(path[-1])
                self._paths[(day, ul)] = pd.Series(path, index=self.session_grid(day), name=f"{ul}_underlying")

    # ---- DataProvider --------------------------------------------------
    @property
    def underliers(self) -> list[str]:
        return sorted(self._specs)

    @property
    def trading_days(self) -> list[date]:
        return list(self._days)

    def multiplier(self, underlier: str) -> float:
        return 100.0

    def _check(self, day: date, underlier: str) -> None:
        if (day, underlier) not in self._paths:
            raise ValueError(f"No data for day {day} and underlier {underlier}")

    def _expiries(self, day: date) -> list[date]:
        first = day + timedelta(days=(self.expiry_weekday - day.weekday()) % 7)
        return [first + timedelta(weeks=i) for i in range(_MAX_EXPIRY_WEEKS)]

    def front_expiry(self, day: date, underlier: str) -> date:
        self._check(day, underlier)
        return self._expiries(day)[0]

    def strikes(self, day: date, underlier: str, expiry: date) -> np.ndarray:
        self._check(day, underlier)
        if expiry not in self._expiries(day):
            raise ValueError(f"No data for day {day}, underlier {underlier} and expiry {expiry}")
        step = self._specs[underlier][1]
        path = self._paths[(day, underlier)]
        lo = int(np.floor(path.min() * 0.95 / step)) * step
        hi = int(np.ceil(path.max() * 1.05 / step)) * step
        return np.arange(lo, hi + step, step, dtype=int)

    def underlying_prices(self, day: date, underlier: str) -> pd.Series:
        self._check(day, underlier)
        return self._paths[(day, underlier)].copy()

    def option_prices(self, day: date, inst: Instrument) -> pd.Series | None:
        if (day, inst.underlier) not in self._paths:
            return None
        if inst.expiry not in self._expiries(day):
            return None
        if inst.strike not in self.strikes(day, inst.underlier, inst.expiry):
            return None
        spot = self._paths[(day, inst.underlier)]
        end = datetime.combine(inst.expiry, self.session_close) + timedelta(seconds=1)
        t_years = (end - spot.index).total_seconds().to_numpy() / _SECONDS_PER_YEAR
        px = bs_price(spot.to_numpy(), float(inst.strike), t_years,
                      self.annual_vol * (1.0 + self.iv_premium), self.rate,
                      inst.option_type == "CE")
        return pd.Series(np.maximum(np.round(px, 2), 0.01), index=spot.index, name=inst.filename)

# src/stratbt/providers/synthetic.py
"""
Self-contained market simulator so the package works with zero downloads.

The underlying follows a geometric Brownian motion sampled every second;
options are priced with Black-Scholes at the matching second. It is a
sandbox for developing and testing strategies and plumbing — results say
nothing about real markets.
"""
from datetime import date, time, timedelta

import numpy as np
import pandas as pd

from stratbt.providers.modelled import TRADING_DAYS_PER_YEAR, ModelPricedProvider, bs_price  # noqa: F401

# underlier -> (starting spot, strike step)
DEFAULT_UNDERLIERS: dict[str, tuple[float, int]] = {
    "SPY": (450.0, 1),
    "QQQ": (380.0, 1),
}


class SyntheticProvider(ModelPricedProvider):
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
        dt = 1.0 / (TRADING_DAYS_PER_YEAR * n)
        for k, (ul, (spot0, _step)) in enumerate(sorted(self._specs.items())):
            rng = np.random.default_rng([self.seed, k])
            spot = spot0
            for day in self._days:
                # lognormal day-to-day vol regime with E[vol^2] = annual_vol^2, so the
                # market is fair on average (no built-in edge for long or short vol)
                day_vol = self.annual_vol * float(np.exp(rng.normal(-0.25**2, 0.25)))
                spot *= float(np.exp(rng.normal(0.0, 0.3 * day_vol / np.sqrt(TRADING_DAYS_PER_YEAR))))
                z = rng.standard_normal(n)
                logret = (-0.5 * day_vol**2) * dt + day_vol * np.sqrt(dt) * z
                path = np.round(spot * np.exp(np.cumsum(logret)), 2)
                spot = float(path[-1])
                self._paths[(day, ul)] = pd.Series(path, index=self.session_grid(day), name=f"{ul}_underlying")

    # ---- ModelPricedProvider hooks ----------------------------------------
    @property
    def underliers(self) -> list[str]:
        return sorted(self._specs)

    @property
    def trading_days(self) -> list[date]:
        return list(self._days)

    def multiplier(self, underlier: str) -> float:
        return 100.0

    def _has(self, day: date, underlier: str) -> bool:
        return (day, underlier) in self._paths

    def _spot(self, day: date, underlier: str) -> pd.Series:
        return self._paths[(day, underlier)]

    def _iv(self, day: date, underlier: str) -> float:
        return self.annual_vol * (1.0 + self.iv_premium)

    def _strike_step(self, underlier: str) -> int:
        return self._specs[underlier][1]

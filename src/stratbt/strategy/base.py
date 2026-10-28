# src/stratbt/strategy/base.py

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from stratbt.instruments import Instrument

@dataclass
class Order:
    """
    An instruction from a strategy to the engine: buy or sell one instrument.
    The strategy never sets a price — the engine looks up the current price
    and executes the order against it.
    """
    instrument: Instrument
    side: str # "BUY" or "SELL"
    qty: int = 1


def nearest_strike(strikes: np.ndarray, price: float) -> int:
    """
    The strike in `strikes` (sorted) closest to `price`, found via binary
    search. Ties are broken toward the lower strike.
    """
    idx = int(np.searchsorted(strikes, price))
    idx = max(1, min(idx, len(strikes) - 1))
    lower, upper = strikes[idx - 1], strikes[idx]
    return int(lower) if abs(price - lower) <= abs(price - upper) else int(upper)


def flatten_orders(holdings: dict[Instrument, tuple[float, int]]) -> list[Order]:
    """Orders that close every open position, long or short."""
    return [
        Order(inst, "SELL" if qty > 0 else "BUY", abs(qty))
        for inst, (_, qty) in holdings.items()
    ]

class BaseStrategy(ABC):
    """
    Abstract base every strategy must implement. The engine calls on_second()
    once per second of the trading session and executes whatever Orders come
    back.

    A strategy must never: read files, compute PnL, or advance the clock.
    Look-ahead safety is enforced architecturally, not by convention — the
    engine only ever passes prices known as of `ts`, so a strategy has no way
    to access future data even if it tried.

    Class attributes
    ----------------
    allow_short : bool
        Set True if the strategy sells options it doesn't hold (short
        premium). The engine then lets its broker open short positions.
    """

    allow_short: bool = False

    @abstractmethod
    def on_second(
        self,
        ts: pd.Timestamp,
        futures_price: float,
        option_price: Callable[[Instrument], float | None],
        holdings: dict[Instrument, tuple[float, int]],
    ) -> list[Order]:
        """
        Decide what to do at one second of the trading session.

        Parameters
        ----------
        ts : pd.Timestamp
            Current timestamp.
        futures_price : float
            Current futures price (never None — the continuous futures
            series is guaranteed liquid at every second of the session).
        option_price : Callable[[Instrument], float | None]
            Last-known price for any instrument as of `ts`; None if that
            instrument hasn't ticked yet.
        holdings : dict[Instrument, tuple[float, int]]
            Current open positions as a read-only snapshot:
            instrument -> (avg_entry_price, qty).

        Returns
        -------
        list[Order]
            Orders to execute this second. An empty list means do nothing.
        """
        ...

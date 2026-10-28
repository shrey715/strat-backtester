# src/stratbt/strategy/structures.py
"""
Classic multi-leg option structures, entered once and held to an exit time.

Every structure is defined by its legs — option type, how many strikes from
the at-the-money strike, and BUY/SELL — so a new one is a few lines:

    class ShortCallSpread(OptionStructureStrategy):
        allow_short = True
        def build_legs(self):
            return (Leg("CE", 0, "SELL"), Leg("CE", self.width, "BUY"))
"""
from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd

from stratbt.instruments import Instrument
from stratbt.strategy.base import BaseStrategy, Order, flatten_orders, nearest_strike


@dataclass(frozen=True)
class Leg:
    """One leg of a structure: `offset` strikes above (+) / below (-) the ATM strike."""
    option_type: str   # "CE" or "PE"
    offset: int
    side: str          # "BUY" or "SELL"


class OptionStructureStrategy(BaseStrategy):
    """
    Enter a fixed set of legs around the ATM strike once the entry window
    opens, hold, and exit before the session close — or earlier on a
    take-profit / stop-loss measured on the structure's PnL in price points
    (per one lot of the structure, before the contract multiplier).

    Parameters
    ----------
    entry_delay_s : int
        Seconds after the open before entering.
    exit_buffer_s : int
        Seconds before the close at which everything is closed.
    width : int
        Distance, in strikes, of the outer legs from ATM (strangle/condor body).
    wing : int
        Extra strikes between a condor's short and protective legs.
    take_profit, stop_loss : float, optional
        Close the structure when its PnL reaches +take_profit / -stop_loss
        (price points; None or 0 disables). Once closed it is not re-entered that day.
    session_open, session_close : time, optional
        Session bounds; default to the NSE session (09:15:00 / 15:29:59).
    """

    SESSION_OPEN = time(9, 15, 0)
    SESSION_CLOSE = time(15, 29, 59)

    def __init__(
        self,
        underlier: str,
        expiry: date,
        strikes: np.ndarray,
        entry_delay_s: int = 300,
        exit_buffer_s: int = 300,
        width: int = 2,
        wing: int = 2,
        take_profit: float | None = None,
        stop_loss: float | None = None,
        session_open: time | None = None,
        session_close: time | None = None,
    ):
        self.underlier = underlier
        self.expiry = expiry
        self.strikes = strikes
        self.entry_delay_s = entry_delay_s
        self.exit_buffer_s = exit_buffer_s
        self.width = width
        self.wing = wing
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.SESSION_OPEN = session_open or type(self).SESSION_OPEN
        self.SESSION_CLOSE = session_close or type(self).SESSION_CLOSE
        self._done = False   # entered once (or tried and stopped out): never re-enter
        self._legs = self.build_legs()

    def build_legs(self) -> tuple[Leg, ...]:
        """The legs of this structure. Override in subclasses."""
        raise NotImplementedError

    def _instruments(self, atm: int) -> list[tuple[Instrument, Leg]] | None:
        """Resolve legs to instruments around `atm`; None if a strike isn't listed."""
        i = int(np.searchsorted(self.strikes, atm))
        out = []
        for leg in self._legs:
            j = i + leg.offset
            if not 0 <= j < len(self.strikes):
                return None
            out.append((Instrument(self.underlier, self.expiry, int(self.strikes[j]), leg.option_type), leg))
        return out

    def _pnl(self, option_price, holdings) -> float:
        """Open PnL of the structure in price points (unpriced legs count as zero)."""
        total = 0.0
        for inst, (entry, qty) in holdings.items():
            px = option_price(inst)
            if px is not None:
                total += (px - entry) * qty
        return total

    def on_second(self, ts: pd.Timestamp, futures_price: float, option_price, holdings) -> list[Order]:
        so, sc = self.SESSION_OPEN, self.SESSION_CLOSE
        open_ts = ts.replace(hour=so.hour, minute=so.minute, second=so.second, microsecond=0)
        close_ts = ts.replace(hour=sc.hour, minute=sc.minute, second=sc.second, microsecond=0)
        entry_ts = open_ts + pd.Timedelta(seconds=self.entry_delay_s)
        exit_ts = close_ts - pd.Timedelta(seconds=self.exit_buffer_s)

        if holdings:
            if ts >= exit_ts:
                return flatten_orders(holdings)
            pnl = self._pnl(option_price, holdings)
            if (self.take_profit and pnl >= self.take_profit) or \
               (self.stop_loss and pnl <= -self.stop_loss):
                return flatten_orders(holdings)
            return []

        if self._done or not (entry_ts <= ts < exit_ts):
            return []

        legs = self._instruments(nearest_strike(self.strikes, futures_price))
        if legs is None or any(option_price(inst) is None for inst, _ in legs):
            return []

        self._done = True
        return [Order(inst, leg.side) for inst, leg in legs]


class LongStrangle(OptionStructureStrategy):
    """Buy an OTM call and an OTM put `width` strikes away: long volatility, cheaper than a straddle."""
    def build_legs(self):
        return (Leg("CE", self.width, "BUY"), Leg("PE", -self.width, "BUY"))


class ShortStraddle(OptionStructureStrategy):
    """Sell the ATM call and put: collect premium, lose on big moves (unlimited risk)."""
    allow_short = True
    def build_legs(self):
        return (Leg("CE", 0, "SELL"), Leg("PE", 0, "SELL"))


class ShortStrangle(OptionStructureStrategy):
    """Sell an OTM call and put `width` strikes away: wider profit zone than a short straddle (unlimited risk)."""
    allow_short = True
    def build_legs(self):
        return (Leg("CE", self.width, "SELL"), Leg("PE", -self.width, "SELL"))


class IronCondor(OptionStructureStrategy):
    """Short strangle at `width` strikes with protective long wings `wing` strikes further out: defined risk."""
    allow_short = True
    def build_legs(self):
        w, k = self.width, self.width + self.wing
        return (Leg("CE", w, "SELL"), Leg("PE", -w, "SELL"),
                Leg("CE", k, "BUY"), Leg("PE", -k, "BUY"))

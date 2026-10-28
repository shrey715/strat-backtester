# src/stratbt/position.py
import pandas as pd
from dataclasses import dataclass, field
from stratbt.instruments import Instrument

@dataclass
class Trade:
    """One executed transaction; one row in the trade log."""
    timestamp: pd.Timestamp
    instrument: Instrument
    side: str  # 'BUY' or 'SELL'
    price: float
    qty: int = 1
    fee: float = 0.0  # commission/fees charged on this fill, in currency
    slippage: float = 0.0  # cost of crossing the spread vs the mid price, in currency

class Position:
    """
    Accounting ledger for one trading session (one underlier, one day).

    Tracks open holdings at average cost, realized PnL from closed trades,
    fees, and the full trade history — nothing here knows about the clock
    or market data; it only reacts to buy()/sell() calls from the engine.

    Holdings are signed: positive qty is long, negative is short. Shorting
    (selling what you don't hold, or more than you hold) is rejected unless
    `allow_short=True`.

    Parameters
    ----------
    multiplier : float, default 1.0
        Currency value of one price point per contract. PnL (realized and
        unrealized) is price difference * qty * multiplier.
    allow_short : bool, default False
    """

    def __init__(self, multiplier: float = 1.0, allow_short: bool = False):
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")
        self.multiplier = multiplier
        self.allow_short = allow_short
        self._holdings: dict[Instrument, tuple[float, int]] = {}
        self._realized: float = 0.0
        self._fees: float = 0.0
        self._slippage: float = 0.0
        self._trades: list[Trade] = []

    def _apply(self, inst: Instrument, signed_qty: int, price: float) -> None:
        """
        Move the position in `inst` by signed_qty at `price`: scale in on the
        same side (average cost), realise PnL on the part that closes the
        opposite side, and open the remainder (if any) the other way round.
        """
        entry, held = self._holdings.get(inst, (0.0, 0))
        if held == 0 or (held > 0) == (signed_qty > 0):
            new_qty = held + signed_qty
            new_avg = (entry * abs(held) + price * abs(signed_qty)) / abs(new_qty)
            self._holdings[inst] = (new_avg, new_qty)
            return
        closing = min(abs(signed_qty), abs(held))
        direction = 1 if held > 0 else -1
        self._realized += (price - entry) * direction * closing * self.multiplier
        remaining = held + signed_qty
        if remaining == 0:
            del self._holdings[inst]
        elif (remaining > 0) == (held > 0):
            self._holdings[inst] = (entry, remaining)   # partial close: cost basis unchanged
        else:
            self._holdings[inst] = (price, remaining)   # flipped through zero

    def buy(self, ts: pd.Timestamp, inst: Instrument, price: float, qty: int = 1,
            fee: float = 0.0, slippage: float = 0.0) -> Trade:
        """
        Open a new long position or scale into an existing one.

        Parameters
        ----------
        ts : pd.Timestamp
            Execution time, recorded on the resulting Trade.
        inst : Instrument
            Contract being bought.
        price : float
            Fill price.
        qty : int, default 1
            Lots to buy. Must be positive.
        fee : float, default 0.0
            Fees charged on this fill (tracked in `fees`, not in `realized`).
        slippage : float, default 0.0
            Currency cost of this fill versus the mid price. Informational:
            it is already inside `price`, and only used for cost reporting.

        Returns
        -------
        Trade
            The trade just recorded (also appended to self.trades).

        Raises
        ------
        ValueError
            If qty <= 0.

        Notes
        -----
        If inst is already held long, updates the average cost basis:
        new_avg = (old_avg * old_qty + price * qty) / (old_qty + qty).
        If it is held short, the buy covers the short (realizing PnL).
        """
        if qty<=0:
            raise ValueError("Quantity must be positive")
        self._apply(inst, qty, price)
        self._fees += fee
        self._slippage += slippage
        trade = Trade(timestamp=ts, instrument=inst, side='BUY', price=price, qty=qty, fee=fee, slippage=slippage)
        self._trades.append(trade)
        return trade

    def sell(self, ts: pd.Timestamp, inst: Instrument, price: float, qty: int = 1,
             fee: float = 0.0, slippage: float = 0.0) -> Trade:
        """
        Close all or part of an existing long position.

        Parameters
        ----------
        ts : pd.Timestamp
            Execution time, recorded on the resulting Trade.
        inst : Instrument
            Contract being sold. Must currently be held.
        price : float
            Fill price.
        qty : int, default 1
            Lots to sell; may be less than the full held quantity (partial
            close). Must be positive and, unless `allow_short`, not exceed
            the held quantity.
        fee : float, default 0.0
            Fees charged on this fill (tracked in `fees`, not in `realized`).
        slippage : float, default 0.0
            Currency cost of this fill versus the mid price. Informational:
            it is already inside `price`, and only used for cost reporting.

        Returns
        -------
        Trade
            The trade just recorded (also appended to self.trades).

        Raises
        ------
        ValueError
            If qty <= 0, or (with allow_short=False) if inst isn't currently
            held or qty exceeds the held quantity.

        Notes
        -----
        Realizes PnL = (price - avg_entry_price) * qty * multiplier on the
        closed part, added to self.realized. The average entry price is
        unchanged by a partial close — only the remaining quantity shrinks.
        With allow_short, selling beyond the holding opens/extends a short.
        """
        if qty<=0:
            raise ValueError("Quantity must be positive")
        if not self.allow_short:
            if inst not in self._holdings:
                raise ValueError(f"Cannot sell {inst} — not held")
            held_qty = self._holdings[inst][1]
            if qty > held_qty:
                raise ValueError(f"Cannot sell {qty} of {inst} — only {held_qty} held")
        self._apply(inst, -qty, price)
        self._fees += fee
        self._slippage += slippage
        trade = Trade(timestamp=ts, instrument=inst, side='SELL', price=price, qty=qty, fee=fee, slippage=slippage)
        self._trades.append(trade)
        return trade

    def unrealized(self, prices: dict[Instrument, float]) -> float:
        """
        Mark-to-market PnL on current holdings at the given prices.

        Parameters
        ----------
        prices : dict[Instrument, float]
            Must contain an entry for every instrument currently held —
            unlike the engine's own option_price callable, this takes a
            plain, already-complete price snapshot and will KeyError on a
            missing instrument rather than skip it.

        Returns
        -------
        float
            sum((price - avg_entry_price) * qty * multiplier) over all held instruments;
            0.0 if flat. Short positions (negative qty) gain when price falls.
        """
        return sum(
            (prices[inst] - entry_price) * qty * self.multiplier
            for inst, (entry_price, qty) in self._holdings.items()
        )

    def mtm(self, prices: dict[Instrument, float]) -> float:
        """
        Parameters
        ----------
        prices : dict[Instrument, float]
            See unrealized() — must cover every held instrument.

        Returns
        -------
        float
            realized + unrealized(prices): gross mark-to-market equity
            (before fees; see `net_realized` / `fees`).
        """
        return self._realized + self.unrealized(prices)

    @property
    def realized(self) -> float:
        """Cumulative realized PnL from closed trades so far."""
        return self._realized

    @property
    def fees(self) -> float:
        """Cumulative fees paid so far."""
        return self._fees

    @property
    def slippage(self) -> float:
        """Cumulative slippage cost (spread paid vs mid), already reflected in fill prices."""
        return self._slippage

    @property
    def net_realized(self) -> float:
        """Realized PnL after fees."""
        return self._realized - self._fees

    @property
    def holdings(self) -> dict[Instrument, tuple[float, int]]:
        """A copy of current holdings: instrument -> (avg_entry_price, signed qty)."""
        return self._holdings.copy()

    @property
    def is_empty(self) -> bool:
        """True if nothing is currently held."""
        return len(self._holdings) == 0

    @property
    def trades(self) -> list[Trade]:
        """A copy of every trade executed so far, in chronological order."""
        return self._trades.copy()

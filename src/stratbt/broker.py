# src/stratbt/broker.py
"""
The seam between a strategy's Orders and a market.

`Broker` is the interface the engine talks to. `PaperBroker` is the
simulated implementation used by backtests and paper trading: it applies the
`ExecutionModel` (slippage, fees) and enforces `RiskLimits`. A live adapter
(e.g. for a retail brokerage API) implements the same interface, so a
strategy that backtests under a PaperBroker runs unchanged against it.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from stratbt.execution import ExecutionModel
from stratbt.instruments import Instrument
from stratbt.position import Position
from stratbt.risk import RiskLimits
from stratbt.strategy.base import Order

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fill:
    """One executed order."""
    timestamp: pd.Timestamp
    instrument: Instrument
    side: str
    qty: int
    price: float
    fee: float
    slippage: float = 0.0


@dataclass(frozen=True)
class Rejection:
    """An order the broker refused, with the reason."""
    timestamp: pd.Timestamp
    order: Order
    reason: str


class Broker(ABC):
    """
    Executes Orders and reports what is held. A live adapter implements
    `submit`, `holdings` and `equity`; the rest has working defaults.
    """

    halted: bool = False                 # set True to refuse new risk (e.g. daily loss limit hit)

    def check_risk(self, net_equity: float) -> None:
        """Called every second with current net PnL; brokers with loss limits latch `halted` here."""

    def flatten_orders(self) -> list[Order]:
        """Orders that close every open position."""
        return [Order(inst, "SELL" if qty > 0 else "BUY", abs(qty)) for inst, (_, qty) in self.holdings.items()]

    @abstractmethod
    def submit(self, order: Order, ts: pd.Timestamp, ref_price: float | None) -> Fill | None:
        """
        Try to execute `order`. `ref_price` is the current market price
        (simulated brokers fill against it; live brokers may ignore it).
        Returns the Fill, or None if the order was rejected.
        """

    @property
    @abstractmethod
    def holdings(self) -> dict[Instrument, tuple[float, int]]:
        """instrument -> (avg_entry_price, signed qty)."""

    @abstractmethod
    def equity(self, prices: dict[Instrument, float]) -> float:
        """Net PnL (after fees) marked at `prices`."""


class PaperBroker(Broker):
    """
    Simulated broker backed by a Position ledger.

    Parameters
    ----------
    multiplier : float
        Currency value of one price point per contract.
    execution : ExecutionModel, optional
    risk : RiskLimits, optional
    allow_short : bool
        Whether opening short positions is permitted.
    """

    def __init__(
        self,
        multiplier: float = 1.0,
        execution: ExecutionModel | None = None,
        risk: RiskLimits | None = None,
        allow_short: bool = False,
    ):
        self.execution = execution or ExecutionModel()
        self.risk = risk or RiskLimits()
        self.position = Position(multiplier=multiplier, allow_short=allow_short)
        self.halted = False
        self.rejections: list[Rejection] = []

    # ---- Broker --------------------------------------------------------
    @property
    def holdings(self) -> dict[Instrument, tuple[float, int]]:
        return self.position.holdings

    def equity(self, prices: dict[Instrument, float]) -> float:
        """Net PnL: realized - fees + unrealized on instruments present in `prices`."""
        p = self.position
        unrealized = sum(
            (prices[i] - entry) * qty * p.multiplier
            for i, (entry, qty) in p.holdings.items() if i in prices
        )
        return p.net_realized + unrealized

    def reduces_exposure(self, order: Order) -> bool:
        """True if the order moves an existing position toward zero."""
        held = self.position.holdings.get(order.instrument)
        if held is None:
            return False
        qty = held[1]
        return (order.side == "SELL" and qty > 0) or (order.side == "BUY" and qty < 0)

    def submit(self, order: Order, ts: pd.Timestamp, ref_price: float | None) -> Fill | None:
        closing = self.reduces_exposure(order)

        if ref_price is None:
            if closing:
                raise RuntimeError(
                    f"No fill price for {order.side} order on {order.instrument.filename} at {ts}. "
                    "Position would be left open — this is a data or loader bug. "
                    "Check that forward-fill is working correctly."
                )
            return self._reject(ts, order, "no price available")

        reason = self._risk_reason(order, closing)
        if reason:
            return self._reject(ts, order, reason)

        price = self.execution.fill_price(order.side, ref_price)
        fee = self.execution.fee(price, order.qty, self.position.multiplier)
        slip = abs(price - ref_price) * order.qty * self.position.multiplier
        if order.side == "BUY":
            self.position.buy(ts, order.instrument, price, order.qty, fee=fee, slippage=slip)
        else:
            self.position.sell(ts, order.instrument, price, order.qty, fee=fee, slippage=slip)
        return Fill(ts, order.instrument, order.side, order.qty, price, fee, slip)

    # ---- risk ----------------------------------------------------------
    def _risk_reason(self, order: Order, closing: bool) -> str | None:
        if closing:
            return None  # reducing risk is always allowed
        if self.halted:
            return "trading halted (daily loss limit)"
        r = self.risk
        if r.max_order_qty is not None and order.qty > r.max_order_qty:
            return f"order qty {order.qty} > max_order_qty {r.max_order_qty}"
        if r.max_open_contracts is not None:
            held = self.position.holdings
            after = sum(abs(q) for _, q in held.values()) + order.qty
            if after > r.max_open_contracts:
                return f"would hold {after} contracts > max_open_contracts {r.max_open_contracts}"
        return None

    def check_risk(self, net_equity: float) -> None:
        """Latch `halted` once the daily loss limit is breached."""
        limit = self.risk.max_daily_loss
        if limit is not None and not self.halted and net_equity <= -limit:
            self.halted = True
            logger.warning("Daily loss limit %.2f breached (equity %.2f): flattening and halting",
                           limit, net_equity)

    def _reject(self, ts: pd.Timestamp, order: Order, reason: str) -> None:
        logger.warning("Rejected %s %s at %s: %s", order.side, order.instrument.filename, ts, reason)
        self.rejections.append(Rejection(ts, order, reason))
        return None

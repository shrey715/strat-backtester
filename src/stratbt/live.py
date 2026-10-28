# src/stratbt/live.py
"""
Run a strategy against a live (or replayed) stream of prices through any Broker.

This is the same decision loop as the backtester — `strategy.on_second(...)`
once per second, orders filled after the configured delay, risk limits
enforced by the broker — but driven by a `QuoteFeed` instead of a
pre-loaded day, with the extras production needs:

* **reconciliation** on start: the positions the broker holds must match what
  the runner last saved, or it refuses to trade;
* an append-only **audit log** (JSON lines) of every decision, fill, rejection
  and safety event;
* a **kill switch**: create a file (or call `kill()`) and the runner flattens
  everything and stops;
* **state persistence** after every fill so a restart can pick up where it left off.

Paper trading is `PaperBroker` + a feed. A real brokerage needs two adapters:
a `QuoteFeed` (streaming prices) and a `Broker` (order routing). Nothing
here talks to a real brokerage.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd

from stratbt.broker import Broker
from stratbt.execution import ExecutionModel
from stratbt.instruments import Instrument
from stratbt.providers.base import DataProvider
from stratbt.strategy.base import BaseStrategy, Order

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    """Everything known at one second."""
    ts: pd.Timestamp
    underlying: float
    option_price: Callable[[Instrument], float | None]
    is_last: bool = False        # final tick of the session: unfilled orders execute now, positions are flattened


class QuoteFeed(ABC):
    """A source of per-second snapshots. Implement this for a live market-data connection."""

    underlier: str
    expiry: date
    strikes: "object"            # np.ndarray of listed strikes (what the strategy factory needs)

    @abstractmethod
    def snapshots(self) -> Iterator[Snapshot]:
        """Yield one Snapshot per second until the session ends."""


class ReplayFeed(QuoteFeed):
    """Replays one historical day from any DataProvider as if it were arriving live."""

    def __init__(self, provider: DataProvider, day: date, underlier: str):
        self.provider, self.day, self.underlier = provider, day, underlier
        self.expiry = provider.front_expiry(day, underlier)
        self.strikes = provider.strikes(day, underlier, self.expiry)

    def snapshots(self) -> Iterator[Snapshot]:
        under = self.provider.underlying_prices(self.day, self.underlier)
        values, grid, cache = under.to_numpy(), under.index, {}
        idx = 0

        def option_price(inst: Instrument) -> float | None:
            if inst not in cache:
                s = self.provider.option_prices(self.day, inst)
                cache[inst] = s.to_numpy() if s is not None else None
            arr = cache[inst]
            if arr is None or arr[idx] != arr[idx]:      # missing or NaN
                return None
            return float(arr[idx])

        last = len(grid) - 1
        for idx, ts in enumerate(grid):
            yield Snapshot(ts, float(values[idx]), option_price, is_last=idx == last)


class AuditLog:
    """Append-only JSON-lines record of everything the runner decides and does."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.events: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, ts=None, **data) -> None:
        rec = {"at": datetime.now().isoformat(timespec="milliseconds"),
               "ts": str(ts) if ts is not None else None, "event": event, **data}
        self.events.append(rec)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")


class KillSwitch:
    """Stops the runner when `kill()` is called or the trigger file exists."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._flag = False

    def kill(self) -> None:
        self._flag = True

    @property
    def triggered(self) -> bool:
        return self._flag or bool(self.path and self.path.exists())


class ReconciliationError(RuntimeError):
    """The broker's positions do not match the runner's saved state."""


def _encode(holdings: dict) -> list[dict]:
    return [{"underlier": i.underlier, "expiry": i.expiry.isoformat(), "strike": i.strike,
             "option_type": i.option_type, "entry": e, "qty": q} for i, (e, q) in holdings.items()]


@dataclass
class SessionSummary:
    net_pnl: float
    n_orders: int
    n_fills: int
    n_rejections: int
    halted: bool
    killed: bool
    audit: AuditLog = field(repr=False, default_factory=AuditLog)
    open_positions: int = 0          # non-zero means the session ended with positions still held


class LiveRunner:
    """
    Parameters
    ----------
    strategy : BaseStrategy
    feed : QuoteFeed
    broker : Broker
    execution : ExecutionModel
        Only `delay_s` is used here: how long after a decision an order is
        sent to the broker (the broker applies costs itself).
    audit : AuditLog, optional
    kill : KillSwitch, optional
    state_path : path, optional
        JSON file holding the positions last known to the runner.
    adopt_existing : bool
        If True, start even when the broker holds positions the saved state doesn't
        know about (they are adopted, and logged).
    """

    def __init__(self, strategy: BaseStrategy, feed: QuoteFeed, broker: Broker,
                 execution: ExecutionModel | None = None, audit: AuditLog | None = None,
                 kill: KillSwitch | None = None, state_path: str | Path | None = None,
                 adopt_existing: bool = False):
        self.strategy, self.feed, self.broker = strategy, feed, broker
        self.delay_s = (execution or ExecutionModel()).delay_s
        self.audit = audit or AuditLog()
        self.kill = kill or KillSwitch()
        self.state_path = Path(state_path) if state_path else None
        self.adopt_existing = adopt_existing
        self.n_orders = self.n_fills = 0

    # ---- safety ----------------------------------------------------------
    def reconcile(self) -> None:
        """Broker positions must equal the saved state (or be empty with no state)."""
        actual = _encode(self.broker.holdings)
        saved = json.loads(self.state_path.read_text()).get("holdings", []) if self.state_path and self.state_path.exists() else []
        key = lambda rows: sorted(json.dumps(r, sort_keys=True) for r in rows)
        if key(actual) != key(saved):
            self.audit.write("reconcile_mismatch", broker=actual, saved=saved)
            if not self.adopt_existing:
                raise ReconciliationError(
                    f"Broker holds {len(actual)} position(s) but saved state has {len(saved)}. "
                    "Resolve the difference or start with adopt_existing=True.")
            self.audit.write("adopted_positions", positions=actual)
        else:
            self.audit.write("reconciled", positions=len(actual))

    def _save_state(self, ts) -> None:
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({"saved_at": str(ts), "holdings": _encode(self.broker.holdings)}, indent=1))

    # ---- loop ------------------------------------------------------------
    def run(self) -> SessionSummary:
        self.audit.write("start", strategy=type(self.strategy).__name__, underlier=self.feed.underlier,
                         delay_s=self.delay_s)
        self.broker.sync()               # live brokers load the venue's true positions here
        self.reconcile()
        pending: list[tuple[pd.Timestamp, Order]] = []
        killed = False
        equity = 0.0
        try:
            self._loop(pending, killed, equity)
        except BaseException as e:
            held = _encode(self.broker.holdings)
            self.audit.write("aborted", error=f"{type(e).__name__}: {e}", open_positions=held)
            if held:
                logger.error("ABORTED WITH %d OPEN POSITION(S): %s", len(held), held)
            raise
        return self._summary

    def _loop(self, pending, killed, equity) -> None:
        for snap in self.feed.snapshots():
            ts = snap.ts
            # 1. send orders whose delay has elapsed (all of them on the last tick)
            if pending:
                due = [o for t, o in pending if t <= ts or snap.is_last]
                pending = [(t, o) for t, o in pending if not (t <= ts or snap.is_last)]
                for order in due:
                    self._submit(order, ts, snap)

            # 2. decide (not while earlier orders are still in flight)
            if not pending:
                if self.kill.triggered and not killed:
                    killed = True
                    self.audit.write("kill_switch", ts, holdings=len(self.broker.holdings))
                if killed or self.broker.halted:
                    orders = self.broker.flatten_orders()
                else:
                    orders = self.strategy.on_second(ts, snap.underlying, snap.option_price, self.broker.holdings)
                    if orders:
                        self.audit.write("decision", ts, orders=[(o.instrument.filename, o.side, o.qty) for o in orders])
                if snap.is_last:
                    orders = list(orders) + [o for o in self.broker.flatten_orders()
                                             if not any(x.instrument == o.instrument for x in orders)]
                for order in orders:
                    self.n_orders += 1
                    if self.delay_s == 0 or snap.is_last:
                        self._submit(order, ts, snap)
                    else:
                        pending.append((ts + pd.Timedelta(seconds=self.delay_s), order))

            # 3. mark to market and apply loss limits
            prices = {i: p for i in self.broker.holdings if (p := snap.option_price(i)) is not None}
            equity = self.broker.equity(prices)
            was_halted = self.broker.halted
            self.broker.check_risk(equity)
            if self.broker.halted and not was_halted:
                self.audit.write("risk_halt", ts, equity=equity)
            if killed and not self.broker.holdings and not pending:
                break

        rejections = len(getattr(self.broker, "rejections", []))
        held = len(self.broker.holdings)
        if held:
            self.audit.write("ended_with_open_positions", positions=_encode(self.broker.holdings))
        self.audit.write("end", equity=equity, fills=self.n_fills, rejections=rejections, killed=killed)
        self._summary = SessionSummary(net_pnl=equity, n_orders=self.n_orders, n_fills=self.n_fills,
                                       n_rejections=rejections, halted=self.broker.halted, killed=killed,
                                       audit=self.audit, open_positions=held)

    def _submit(self, order: Order, ts: pd.Timestamp, snap: Snapshot) -> None:
        fill = self.broker.submit(order, ts, snap.option_price(order.instrument))
        if fill is None:
            self.audit.write("rejected", ts, order=(order.instrument.filename, order.side, order.qty))
            return
        self.n_fills += 1
        self.audit.write("fill", ts, instrument=order.instrument.filename, side=fill.side, qty=fill.qty,
                         price=fill.price, fee=fill.fee)
        self._save_state(ts)

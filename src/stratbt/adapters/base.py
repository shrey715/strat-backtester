# src/stratbt/adapters/base.py
"""
Shared machinery for connecting stratbt to a real venue.

A venue adapter is a pair:

* a **feed** (`PollingFeed` subclass) that turns the venue's quotes into the per-second
  snapshots a strategy consumes, and
* a **broker** (`LiveBroker` subclass) that turns the strategy's Orders into venue orders.

Everything that must be identical across venues lives here: the three trading modes, the
real-money confirmation, size and loss caps, quote sanity checks, marketable-limit pricing,
fill tracking (wait, cancel on timeout, accept partial fills) and shadow bookkeeping.

SAFETY MODEL
------------
mode="dry-run"  Real quotes, real decisions, **no order is ever sent**. Fills are simulated at the limit
                price (venues that offer a pre-flight/validation call also get one).
mode="paper"    Orders go to the venue's own paper/simulated account.
mode="live"     Real money. Refused unless you pass the confirmation token (or set
                STRATBT_CONFIRM_REAL_MONEY) AND a daily-loss limit. Size caps default to one contract.
"""
from __future__ import annotations

import logging
import os
import time as _time
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, time
from math import ceil, floor
from zoneinfo import ZoneInfo

import pandas as pd

from stratbt.broker import Fill, PaperBroker
from stratbt.execution import ExecutionModel
from stratbt.instruments import Instrument
from stratbt.live import QuoteFeed, Snapshot
from stratbt.risk import RiskLimits
from stratbt.strategy.base import Order

logger = logging.getLogger(__name__)

MODES = ("dry-run", "paper", "live")
REAL_MONEY_TOKEN = "I_ACCEPT_REAL_MONEY_TRADING_RISK"
REAL_MONEY_ENV = "STRATBT_CONFIRM_REAL_MONEY"


class LiveTradingNotConfirmed(PermissionError):
    """Real-money mode was requested without the explicit confirmation."""


class FeedError(RuntimeError):
    """The quote feed failed repeatedly; trading cannot continue safely."""


# ── symbology ────────────────────────────────────────────────────────────

def osi_symbol(inst: Instrument) -> str:
    """OCC/OSI option symbol as used by Alpaca and Public: ROOT + YYMMDD + C|P + strike*1000 (8 digits), no padding."""
    cp = "C" if inst.option_type == "CE" else "P"
    return f"{inst.underlier}{inst.expiry:%y%m%d}{cp}{int(round(inst.strike * 1000)):08d}"


def parse_osi(symbol: str, underlier: str | None = None) -> Instrument | None:
    """Inverse of `osi_symbol`; returns None if it isn't a plain option symbol (e.g. a stock)."""
    s = symbol.replace(" ", "")
    if len(s) < 16:
        return None
    tail, root = s[-15:], s[:-15]
    try:
        expiry = datetime.strptime(tail[:6], "%y%m%d").date()
        strike = int(tail[7:]) / 1000.0
    except ValueError:
        return None
    if tail[6] not in "CP" or not root or (underlier and root != underlier):
        return None
    return Instrument(root, expiry, int(strike) if strike == int(strike) else strike, "CE" if tail[6] == "C" else "PE")


# ── data classes ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Quote:
    bid: float | None
    ask: float | None
    last: float | None = None
    ts: float = 0.0                       # epoch seconds when it was fetched

    @property
    def mid(self) -> float | None:
        if self.bid and self.ask and self.ask >= self.bid > 0:
            return (self.bid + self.ask) / 2.0
        return self.last if self.last else None


@dataclass
class FillReport:
    """What a venue says happened to one order."""
    filled_qty: float
    avg_price: float
    status: str
    fee: float = 0.0
    order_id: str | None = None


@dataclass(frozen=True)
class LiveSafety:
    """Conservative defaults; loosen them deliberately."""
    mode: str = "dry-run"
    max_order_qty: int = 1
    max_open_contracts: int = 4
    max_spread_pct: float = 0.30         # refuse to trade a market wider than 30% of its mid
    quote_max_age_s: float = 15.0        # refuse to trade on a quote older than this
    fill_timeout_s: float = 8.0          # cancel an order that has not filled by then
    poll_s: float = 0.5                  # how often to check an open order
    price_at: str = "touch"              # "touch" = buy the ask / sell the bid (marketable); "mid" = join the mid
    tick: float = 0.01
    confirm_real_money: str | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.price_at not in ("touch", "mid"):
            raise ValueError("price_at must be 'touch' or 'mid'")
        if self.max_order_qty < 1 or self.max_open_contracts < 1:
            raise ValueError("size caps must be at least 1")


def require_real_money_confirmation(safety: LiveSafety, risk: RiskLimits) -> None:
    if safety.mode == "dry-run":
        return
    if risk.max_daily_loss is None:
        raise ValueError(f"mode='{safety.mode}' requires a daily loss limit (RiskLimits.max_daily_loss).")
    if safety.mode == "live":
        token = safety.confirm_real_money or os.environ.get(REAL_MONEY_ENV)
        if token != REAL_MONEY_TOKEN:
            raise LiveTradingNotConfirmed(
                "Real-money trading is off. To enable it, pass confirm_real_money='"
                f"{REAL_MONEY_TOKEN}' or set {REAL_MONEY_ENV} to that value. "
                "Start with mode='dry-run', then 'paper', and size caps of one contract.")


# ── broker ───────────────────────────────────────────────────────────────

class LiveBroker(PaperBroker):
    """
    A Broker that routes orders to a real venue. Inherits PaperBroker's position ledger,
    risk limits, daily-loss halt and rejection log, and replaces order execution.

    Subclasses implement `_place` (and optionally `_preflight`) plus `positions`.
    """

    name = "live"

    def __init__(self, multiplier: float = 100.0, risk: RiskLimits | None = None, safety: LiveSafety | None = None,
                 allow_short: bool = True, clock=_time.time, sleep=_time.sleep):
        self.safety = safety or LiveSafety()
        risk = risk or RiskLimits()
        require_real_money_confirmation(self.safety, risk)
        capped = RiskLimits(
            max_daily_loss=risk.max_daily_loss,
            max_open_contracts=min(risk.max_open_contracts or self.safety.max_open_contracts, self.safety.max_open_contracts),
            max_order_qty=min(risk.max_order_qty or self.safety.max_order_qty, self.safety.max_order_qty))
        super().__init__(multiplier, ExecutionModel(delay_s=0), capped, allow_short=allow_short)
        self.clock, self.sleep = clock, sleep
        self.quote_source = None             # callable(Instrument) -> Quote | None, set by `attach`
        self.sent: list[dict] = []           # every order actually sent to the venue (audit / tests)

    def attach(self, feed: "PollingFeed") -> "LiveBroker":
        """Use the feed's latest quotes to price orders."""
        self.quote_source = feed.quote
        return self

    # -- to implement ------------------------------------------------------
    @abstractmethod
    def _place(self, order: Order, limit_price: float, client_id: str) -> FillReport:
        """Send one limit order, wait up to safety.fill_timeout_s, cancel what is left, report the outcome."""

    @abstractmethod
    def positions(self) -> dict[Instrument, tuple[float, int]]:
        """The venue's true open option positions: instrument -> (avg entry, signed qty)."""

    def _track(self, fetch, cancel, terminal: frozenset[str]) -> FillReport:
        """
        Follow one working order: poll `fetch() -> FillReport` until its status is terminal. If it has not
        filled by safety.fill_timeout_s, call `cancel()` once and give the venue the same time again to
        confirm; whatever was filled by then (possibly a partial fill) is what we report.
        """
        deadline = self.clock() + self.safety.fill_timeout_s
        cancelled = False
        while True:
            rep = fetch()
            if rep.status in terminal:
                return rep
            if self.clock() > deadline:
                if cancelled:
                    logger.error("order %s still %s after cancel; treating %s as filled", rep.order_id, rep.status, rep.filled_qty)
                    rep.status = f"unconfirmed:{rep.status}"
                    return rep
                cancel()
                cancelled = True
                deadline = self.clock() + self.safety.fill_timeout_s
            self.sleep(self.safety.poll_s)

    def _preflight(self, order: Order, limit_price: float) -> None:
        """Optional venue-side validation used in dry-run (raise to refuse). Default: none."""

    # -- Broker ------------------------------------------------------------
    def sync(self) -> None:
        self.position.load_holdings(self.positions())

    def submit(self, order: Order, ts: pd.Timestamp, ref_price: float | None) -> Fill | None:
        if ref_price is None:                # live: a missing price is a retry next tick, never a crash
            return self._reject(ts, order, "no price available")
        return super().submit(order, ts, ref_price)

    def _limit_price(self, order: Order) -> tuple[float | None, str]:
        q = self.quote_source(order.instrument) if self.quote_source else None
        if q is None:
            return None, "no quote"
        if self.clock() - q.ts > self.safety.quote_max_age_s:
            return None, f"stale quote ({self.clock() - q.ts:.0f}s old)"
        if not (q.bid and q.ask and q.ask >= q.bid > 0):
            return None, "no two-sided market"
        mid = (q.bid + q.ask) / 2.0
        spread = (q.ask - q.bid) / mid
        if spread > self.safety.max_spread_pct:
            return None, f"spread too wide ({spread:.0%} of mid)"
        tick = self.safety.tick
        if self.safety.price_at == "touch":
            px = q.ask if order.side == "BUY" else q.bid
        else:
            px = mid
        px = ceil(px / tick - 1e-9) * tick if order.side == "BUY" else floor(px / tick + 1e-9) * tick
        return round(max(px, tick), 4), ""

    def _execute(self, order: Order, ts: pd.Timestamp, ref_price: float) -> Fill | None:
        limit, why = self._limit_price(order)
        if limit is None:
            return self._reject(ts, order, why)
        client_id = uuid.uuid4().hex
        try:
            if self.safety.mode == "dry-run":
                self._preflight(order, limit)
                report = FillReport(order.qty, limit, "simulated (dry-run: nothing sent)")
            else:
                q = self.quote_source(order.instrument)
                self.sent.append({"client_id": client_id, "symbol": order.instrument.filename, "side": order.side,
                                  "qty": order.qty, "limit": limit, "mode": self.safety.mode,
                                  "bid": q.bid, "ask": q.ask})                  # the market the price was derived from
                report = self._place(order, limit, client_id)
        except Exception as e:               # venue/network errors: refuse and retry next tick
            logger.exception("venue error on %s", order)
            return self._reject(ts, order, f"venue error: {type(e).__name__}: {e}")
        qty = int(round(report.filled_qty))
        if qty <= 0:
            return self._reject(ts, order, f"not filled ({report.status})")
        slip = abs(report.avg_price - ref_price) * qty * self.position.multiplier
        buy = self.position.buy if order.side == "BUY" else self.position.sell
        buy(ts, order.instrument, report.avg_price, qty, fee=report.fee, slippage=slip)
        return Fill(ts, order.instrument, order.side, qty, report.avg_price, report.fee, slip)

    # helper for adapters: opening or closing?
    def is_closing(self, order: Order) -> bool:
        return self.reduces_exposure(order)


# ── feed ─────────────────────────────────────────────────────────────────

class PollingFeed(QuoteFeed):
    """
    Turns a venue's quote API into per-second `Snapshot`s.

    The strategy asks for option prices lazily (`snapshot.option_price(inst)`); the first ask for an
    instrument returns None and registers it, and from the next tick on it is fetched in one batched
    call together with the underlying. Subclasses implement `fetch_underlying` and `fetch_options`.
    """

    MAX_WATCHED = 60

    def __init__(self, underlier: str, expiry: date, strikes, session_open: time = time(9, 30),
                 session_close: time = time(15, 59, 59), tz: str = "America/New_York", poll_interval_s: float = 1.0,
                 quote_max_age_s: float = 15.0, max_failures: int = 5, clock=_time.time, sleep=_time.sleep):
        self.underlier, self.expiry, self.strikes = underlier, expiry, strikes
        self.session_open, self.session_close = session_open, session_close
        self.tz, self.poll_s, self.max_age = ZoneInfo(tz), poll_interval_s, quote_max_age_s
        self.max_failures, self.clock, self.sleep = max_failures, clock, sleep
        self._quotes: dict[Instrument, Quote] = {}
        self._watch: dict[Instrument, int] = {}
        self._under: Quote | None = None
        self._tick = 0
        self._failures = 0

    @abstractmethod
    def fetch_underlying(self) -> Quote:
        """Latest underlying quote."""

    @abstractmethod
    def fetch_options(self, instruments: list[Instrument]) -> dict[Instrument, Quote]:
        """Latest quotes for these option contracts (one batched call if the venue allows)."""

    # -- strategy-facing ---------------------------------------------------
    def quote(self, inst: Instrument) -> Quote | None:
        return self._quotes.get(inst)

    def _price(self, inst: Instrument) -> float | None:
        self._watch[inst] = self._tick
        q = self._quotes.get(inst)
        if q is None or self.clock() - q.ts > self.max_age:
            return None
        return q.mid

    def _refresh(self) -> None:
        wanted = [i for i, t in self._watch.items() if self._tick - t < 120][-self.MAX_WATCHED:]
        try:
            self._under = self.fetch_underlying()
            if wanted:
                self._quotes.update(self.fetch_options(wanted))
            self._failures = 0
        except Exception as e:
            self._failures += 1
            logger.warning("quote fetch failed (%d/%d): %s", self._failures, self.max_failures, e)
            if self._failures >= self.max_failures:
                raise FeedError(f"Quote feed failed {self._failures} times in a row: {e}") from e

    def snapshots(self):
        while True:
            now = self.clock()
            local = datetime.fromtimestamp(now, self.tz)
            if local.time() < self.session_open:
                self.sleep(min(30.0, max(0.5, self.poll_s)))
                continue
            self._tick += 1
            self._refresh()
            under = self._under.mid if self._under else None
            is_last = local.time() >= self.session_close
            if under is not None:
                yield Snapshot(pd.Timestamp(local.replace(tzinfo=None, microsecond=0)), float(under), self._price, is_last)
            if is_last:
                return
            self.sleep(max(0.0, self.poll_s - (self.clock() - now)))

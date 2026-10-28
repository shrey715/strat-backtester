# tests/venue_fakes.py
"""A simulated market and fake broker-SDK clients, so adapters can be exercised end to end with no network."""
import uuid
from datetime import datetime, time
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from stratbt import SyntheticProvider
from stratbt.adapters.base import osi_symbol, parse_osi
from stratbt.instruments import Instrument
from stratbt.strategy.structures import IronCondor

NY = ZoneInfo("America/New_York")
OPEN, CLOSE = time(9, 30), time(9, 49, 59)        # a 20-minute session keeps tests fast


class MarketSim:
    """Quotes for one synthetic day, advanced by a fake clock (sleep() moves time forward)."""

    def __init__(self, seed: int = 3, half_spread: float = 0.02):
        self.provider = SyntheticProvider(n_days=2, seed=seed)
        self.day = self.provider.trading_days[0]
        self.ticker = "SPY"
        self.half = half_spread
        self.expiry = self.provider.front_expiry(self.day, self.ticker)
        self.strikes = [int(k) for k in self.provider.strikes(self.day, self.ticker, self.expiry)]
        self.now = datetime.combine(self.day, time(9, 29, 55), NY).timestamp()
        self._under = self.provider.underlying_prices(self.day, self.ticker).to_numpy()
        self._opt = {}
        self.wide_spread = False
        self.frozen = False

    def clock(self) -> float:
        return self.now

    def sleep(self, secs: float = 0.02) -> bool:
        self.now += max(float(secs), 0.001)
        return True

    def _idx(self) -> int:
        local = datetime.fromtimestamp(self.now, NY)
        return max(0, min(int((local - datetime.combine(self.day, time(9, 30), NY)).total_seconds()), len(self._under) - 1))

    def underlying(self) -> float:
        return float(self._under[self._idx()])

    def option_mid(self, inst: Instrument) -> float | None:
        if inst not in self._opt:
            s = self.provider.option_prices(self.day, inst)
            self._opt[inst] = None if s is None else s.to_numpy()
        arr = self._opt[inst]
        return None if arr is None else float(arr[self._idx()])

    def bid_ask(self, inst: Instrument) -> tuple[float, float] | None:
        mid = self.option_mid(inst)
        if mid is None:
            return None
        h = mid * 0.6 if self.wide_spread else self.half
        return round(max(mid - h, 0.01), 2), round(mid + h, 2)

    def condor(self, **kw) -> IronCondor:
        return IronCondor(self.ticker, self.expiry, self.strikes, entry_delay_s=60, exit_buffer_s=60,
                          session_open=OPEN, session_close=CLOSE, **kw)


def feed_kwargs(sim: MarketSim) -> dict:
    return dict(session_open=OPEN, session_close=CLOSE, tz="America/New_York", poll_interval_s=1.0, clock=sim.clock, sleep=sim.sleep)


# ── Alpaca ───────────────────────────────────────────────────────────────

class FakeAlpacaTrading:
    def __init__(self, sim: MarketSim, fill: str = "full", positions=None, fill_qty: int = 2):
        self.sim, self.fill, self.fill_qty = sim, fill, fill_qty
        self.submitted, self.cancelled, self._orders, self._positions = [], [], {}, positions or []
        self.quote_at_submit = []

    def submit_order(self, req):
        if self.fill == "error":
            raise RuntimeError("503 service unavailable")
        oid = uuid.uuid4()
        self.submitted.append(req)
        self.quote_at_submit.append(self.sim.bid_ask(parse_osi(req.symbol)))    # what the market was when the order went out
        self._orders[str(oid)] = {"req": req, "cancelled": False}
        return SimpleNamespace(id=oid)

    def get_order_by_id(self, oid):
        from alpaca.trading.enums import OrderStatus
        rec = self._orders[str(oid)]
        req, qty = rec["req"], int(rec["req"].qty)
        if self.fill == "full":
            status, filled = OrderStatus.FILLED, qty
        elif self.fill == "partial":
            status, filled = (OrderStatus.CANCELED if rec["cancelled"] else OrderStatus.PARTIALLY_FILLED), min(self.fill_qty, qty)
        else:                                             # "never"
            status, filled = (OrderStatus.CANCELED if rec["cancelled"] else OrderStatus.NEW), 0
        return SimpleNamespace(id=oid, status=status, filled_qty=str(filled),
                               filled_avg_price=str(req.limit_price) if filled else None)

    def cancel_order_by_id(self, oid):
        self.cancelled.append(str(oid))
        self._orders[str(oid)]["cancelled"] = True

    def get_all_positions(self):
        return self._positions

    def get_option_contracts(self, req):
        s = self.sim
        return SimpleNamespace(option_contracts=[SimpleNamespace(symbol=osi_symbol(Instrument(s.ticker, s.expiry, k, t)),
                                                                 expiration_date=s.expiry, strike_price=float(k), tradable=True)
                                                 for k in s.strikes for t in ("CE", "PE")], next_page_token=None)


class FakeAlpacaStock:
    def __init__(self, sim):
        self.sim = sim

    def get_stock_latest_quote(self, req):
        p = self.sim.underlying()
        return {req.symbol_or_symbols: SimpleNamespace(bid_price=p - 0.01, ask_price=p + 0.01)}


class FakeAlpacaOptions:
    def __init__(self, sim):
        self.sim, self.requests = sim, []

    def get_option_latest_quote(self, req):
        self.requests.append(list(req.symbol_or_symbols))
        out = {}
        for sym in req.symbol_or_symbols:
            ba = self.sim.bid_ask(parse_osi(sym))
            if ba:
                out[sym] = SimpleNamespace(bid_price=ba[0], ask_price=ba[1])
        return out


# ── Public ───────────────────────────────────────────────────────────────

class FakePublic:
    def __init__(self, sim: MarketSim, fill: str = "full", positions=None, preflight_error: Exception | None = None):
        self.sim, self.fill, self._positions, self.preflight_error = sim, fill, positions or [], preflight_error
        self.placed, self.cancelled, self.preflights, self._orders = [], [], [], {}

    def _q(self, symbol, bid, ask, last=None):
        return SimpleNamespace(instrument=SimpleNamespace(symbol=symbol), bid=Decimal(str(bid)), ask=Decimal(str(ask)),
                               last=Decimal(str(last if last is not None else (bid + ask) / 2)))

    def get_quotes(self, instruments):
        out = []
        for i in instruments:
            if str(getattr(i.type, "value", i.type)) == "EQUITY":
                p = self.sim.underlying()
                out.append(self._q(i.symbol, p - 0.01, p + 0.01))
            else:
                ba = self.sim.bid_ask(parse_osi(i.symbol))
                if ba:
                    out.append(self._q(i.symbol, *ba))
        return out

    def get_option_expirations(self, req):
        return SimpleNamespace(expirations=[self.sim.expiry.isoformat()])

    def get_option_chain(self, req):
        s = self.sim
        mk = lambda t: [self._q(osi_symbol(Instrument(s.ticker, s.expiry, k, t)), 1, 1.1) for k in s.strikes]
        return SimpleNamespace(calls=mk("CE"), puts=mk("PE"))

    def perform_preflight_calculation(self, req):
        if self.preflight_error:
            raise self.preflight_error
        self.preflights.append(req)
        return SimpleNamespace(estimated_commission=0)

    def place_order(self, req):
        if self.fill == "error":
            raise RuntimeError("503 service unavailable")
        self.placed.append(req)
        self._orders[req.order_id] = {"req": req, "cancelled": False}
        return SimpleNamespace(order_id=req.order_id)

    def get_order(self, oid):
        from public_api_sdk import OrderStatus
        rec = self._orders[oid]
        qty = int(rec["req"].quantity)
        if self.fill == "full":
            st, filled = OrderStatus.FILLED, qty
        else:
            st, filled = (OrderStatus.CANCELLED if rec["cancelled"] else OrderStatus.NEW), 0
        return SimpleNamespace(status=st, filled_quantity=Decimal(filled), average_price=rec["req"].limit_price if filled else None)

    def cancel_order(self, oid):
        self.cancelled.append(oid)
        self._orders[oid]["cancelled"] = True

    def get_portfolio(self):
        return SimpleNamespace(positions=self._positions)


# ── Interactive Brokers ──────────────────────────────────────────────────

class FakeTicker:
    def __init__(self, sim, contract):
        self.sim, self.contract = sim, contract

    def _ba(self):
        c = self.contract
        if c.secType == "STK":
            p = self.sim.underlying()
            return p - 0.01, p + 0.01
        inst = Instrument(c.symbol, datetime.strptime(c.lastTradeDateOrContractMonth, "%Y%m%d").date(), int(c.strike),
                          "CE" if c.right == "C" else "PE")
        return self.sim.bid_ask(inst) or (-1, -1)

    bid = property(lambda s: s._ba()[0])
    ask = property(lambda s: s._ba()[1])
    last = property(lambda s: float("nan"))
    close = property(lambda s: float("nan"))


class FakeIB:
    def __init__(self, sim: MarketSim, accounts=("DU1234567",), fill: str = "full", positions=None):
        self.sim, self._accounts, self.fill, self._positions = sim, list(accounts), fill, positions or []
        self.orders, self.cancelled, self.subscriptions, self.trades = [], [], [], []
        self._next = 100

    def managedAccounts(self):
        return self._accounts

    def sleep(self, secs=0.02):
        return self.sim.sleep(secs)

    def qualifyContracts(self, *contracts):
        for c in contracts:
            c.conId = 12345
        return list(contracts)

    def reqMarketDataType(self, n):
        pass

    def reqSecDefOptParams(self, sym, ex, kind, conId):
        s = self.sim
        return [SimpleNamespace(exchange="SMART", expirations=[s.expiry.strftime("%Y%m%d")], strikes=[float(k) for k in s.strikes])]

    def reqMktData(self, contract, *a, **k):
        self.subscriptions.append(contract)
        return FakeTicker(self.sim, contract)

    def cancelMktData(self, contract):
        pass

    def placeOrder(self, contract, order):
        self._next += 1
        order.orderId = self._next
        self.orders.append((contract, order))
        qty = int(order.totalQuantity)
        filled = qty if self.fill == "full" else 0
        status = "Filled" if self.fill == "full" else "Submitted"
        trade = SimpleNamespace(order=order, contract=contract,
                                orderStatus=SimpleNamespace(status=status, filled=filled, avgFillPrice=order.lmtPrice if filled else 0.0))
        self.trades.append(trade)
        return trade

    def cancelOrder(self, order):
        self.cancelled.append(order.orderId)
        for t in self.trades:
            if t.order is order:
                t.orderStatus.status = "Cancelled"

    def positions(self):
        return self._positions

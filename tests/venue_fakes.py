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

    def __init__(self, seed: int = 3, half_spread: float = 0.02, ticker: str = "SPY", tz: ZoneInfo = NY, underliers: dict | None = None):
        self.provider = SyntheticProvider(n_days=2, seed=seed, underliers=underliers)
        self.day = self.provider.trading_days[0]
        self.ticker, self.tz = ticker, tz
        self.half = half_spread
        self.expiry = self.provider.front_expiry(self.day, self.ticker)
        self.strikes = [int(k) for k in self.provider.strikes(self.day, self.ticker, self.expiry)]
        self.now = datetime.combine(self.day, time(9, 29, 55), self.tz).timestamp()
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
        local = datetime.fromtimestamp(self.now, self.tz)
        return max(0, min(int((local - datetime.combine(self.day, time(9, 30), self.tz)).total_seconds()), len(self._under) - 1))

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
    return dict(session_open=OPEN, session_close=CLOSE, tz=sim.tz.key, poll_interval_s=1.0, clock=sim.clock, sleep=sim.sleep)


IST = ZoneInfo("Asia/Kolkata")


def nse_sim(**kw) -> MarketSim:
    """A NIFTY-like market (strikes every 50) on India time."""
    return MarketSim(ticker="NIFTY", tz=IST, underliers={"NIFTY": (24000.0, 50)}, half_spread=0.10, **kw)


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


# ── Zerodha Kite ─────────────────────────────────────────────────────────

def _tick(x: float, t: float = 0.05) -> float:
    return round(round(x / t) * t, 2)


class FakeKite:
    """Mimics the parts of KiteConnect the adapter uses; records every call."""

    def __init__(self, sim: MarketSim, lot_size: int = 75, fill: str = "full", positions=None, margin_error: Exception | None = None,
                 fill_lots: int = 2, fill_units: int | None = None):
        self.sim, self.lot, self.fill, self._positions, self.margin_error, self.fill_lots = sim, lot_size, fill, positions or [], margin_error, fill_lots
        self.fill_units = fill_units
        self.placed, self.cancelled, self.margin_checks, self.quote_calls, self._orders = [], [], [], [], {}
        self._n = 0
        self.symbols = {}

    def profile(self):
        return {"user_id": "AB1234"}

    def instruments(self, exchange=None):
        s = self.sim
        rows = []
        for k in s.strikes:
            for t in ("CE", "PE"):
                sym = f"{s.ticker}{s.expiry:%y%b}{k}{t}".upper()
                self.symbols[sym] = (k, t)
                rows.append(dict(instrument_token=len(rows) + 1, tradingsymbol=sym, name=s.ticker, expiry=s.expiry, strike=float(k),
                                 tick_size=0.05, lot_size=self.lot, instrument_type=t, segment="NFO-OPT", exchange="NFO"))
        rows.append(dict(instrument_token=999, tradingsymbol=f"{s.ticker}FUT", name=s.ticker, expiry=s.expiry, strike=0.0, tick_size=0.05,
                         lot_size=self.lot, instrument_type="FUT", segment="NFO-FUT", exchange="NFO"))      # decoys the filter must ignore
        rows.append(dict(instrument_token=998, tradingsymbol="BANKNIFTY99XXX1CE", name="BANKNIFTY", expiry=s.expiry, strike=1.0,
                         tick_size=0.05, lot_size=15, instrument_type="CE", segment="NFO-OPT", exchange="NFO"))
        return rows

    def _inst(self, tradingsymbol):
        k, t = self.symbols[tradingsymbol]
        return Instrument(self.sim.ticker, self.sim.expiry, k, t)

    def quote(self, *keys):
        self.quote_calls.append(keys)
        out = {}
        for key in keys:
            ex, _, sym = key.partition(":")
            if sym == "NIFTY 50":
                out[key] = {"last_price": self.sim.underlying(), "depth": {"buy": [{"price": 0, "quantity": 0, "orders": 0}], "sell": [{"price": 0}]}}
            elif sym in self.symbols:
                ba = self.sim.bid_ask(self._inst(sym))
                if ba:
                    out[key] = {"last_price": _tick(sum(ba) / 2), "depth": {"buy": [{"price": _tick(ba[0]), "quantity": 750, "orders": 4}],
                                                                           "sell": [{"price": _tick(ba[1]), "quantity": 750, "orders": 4}]}}
        return out

    def order_margins(self, params):
        self.margin_checks.append(params)
        if self.margin_error:
            raise self.margin_error
        return [{"total": 1000.0}]

    def place_order(self, **kw):
        if self.fill == "error":
            raise RuntimeError("503 gateway")
        self._n += 1
        oid = f"2610{self._n:08d}"
        self.placed.append(kw)
        self._orders[oid] = {"kw": kw, "cancelled": False}
        return oid

    def order_history(self, oid):
        rec = self._orders[oid]
        kw = rec["kw"]
        qty = kw["quantity"]
        if self.fill == "full":
            return [{"status": "OPEN"}, {"status": "COMPLETE", "filled_quantity": qty, "pending_quantity": 0, "average_price": kw["price"]}]
        if self.fill == "partial":
            filled = self.fill_units if self.fill_units is not None else min(self.fill_lots * self.lot, qty)
            return [{"status": "CANCELLED" if rec["cancelled"] else "OPEN", "filled_quantity": filled, "average_price": kw["price"]}]
        return [{"status": "CANCELLED" if rec["cancelled"] else "OPEN", "filled_quantity": 0, "average_price": 0}]    # never fills

    def cancel_order(self, variety, order_id, parent_order_id=None):
        self.cancelled.append((variety, order_id))
        self._orders[order_id]["cancelled"] = True

    def positions(self):
        return {"net": self._positions, "day": []}

    def login_url(self):
        return "https://kite.zerodha.com/connect/login?v=3&api_key=KEY"

    def generate_session(self, request_token, api_secret):
        self.session_args = (request_token, api_secret)
        return {"access_token": "TOKEN_FOR_TODAY"}
